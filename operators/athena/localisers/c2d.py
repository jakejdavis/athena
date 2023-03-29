from typing import Iterable

import numpy as np
from keras.engine.sequential import Sequential
from keras.layers import Normalization
from numpy import ndarray
from tensorflow.keras.models import Model
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tqdm import tqdm

from .localiser import Localiser


class C2DLocaliser(Localiser):
    """
    Localiser for Conv2D layers.
    """

    def __init__(
        self,
        model: Sequential,
        layer_index: int,
        layer_weights: ResourceVariable,
        norm_scaler: Normalization,
    ) -> None:
        super().__init__(model, layer_index, layer_weights, norm_scaler)

    def compute_fi_gl(self, inputs_outputs: ndarray):
        """
        Calculates the forward impact and gradient loss for a C2D layer.

        This function is based on https://github.com/coinse/arachne/blob/cc1523ce4a9ad6dbbecdf87e1f5712a72e48a393/arachne/run_localise.py#L355

        :param inputs_outputs: inputs and outputs to use for calculating FL, GL
        :return: forward impact and gradient loss
        """

        inputs, _ = inputs_outputs
        layer_config = self.model.layers[self.layer_index].get_config()

        is_channel_first = layer_config["data_format"] == "channels_first"
        if self.layer_index == 0:
            new_shape = inputs.shape + (1,)
            inputs_reshaped = inputs.reshape(new_shape)

            prev_output_v = inputs_reshaped
            prev_output = inputs_reshaped
        else:
            target_model = Model(
                inputs=self.model.input,
                outputs=self.model.layers[self.layer_index - 1].output,
            )
            prev_output_v = target_model.predict(inputs)
            prev_output = prev_output_v

        tr_prev_output_v = (
            np.moveaxis(prev_output_v, [1, 2, 3], [3, 1, 2])
            if is_channel_first
            else prev_output_v
        )

        kernel_shape = self.layer_weights.shape[:2]
        strides = layer_config["strides"]
        padding_type = layer_config["padding"]
        if padding_type == "valid":
            paddings = [0, 0]
        else:
            if padding_type == "same":
                true_ws_shape = [
                    self.layer_weights.shape[0],
                    self.layer_weights.shape[-1],
                ]
                paddings = [
                    int(
                        (
                            (strides[i] - 1) * true_ws_shape[i]
                            - strides[i]
                            + kernel_shape[i]
                        )
                        / 2
                    )
                    for i in range(2)
                ]
            elif not isinstance(padding_type, str) and isinstance(
                padding_type, Iterable
            ):
                paddings = list(padding_type)
                if len(paddings) == 1:
                    paddings = [paddings[0], paddings[0]]
            else:
                print("padding type: {} not supported".format(padding_type))
                paddings = [0, 0]
                assert False

            # Add padding
            if is_channel_first:
                paddings_per_axis = [
                    [0, 0],
                    [0, 0],
                    [paddings[0], paddings[0]],
                    [paddings[1], paddings[1]],
                ]
            else:
                paddings_per_axis = [
                    [0, 0],
                    [paddings[0], paddings[0]],
                    [paddings[1], paddings[1]],
                    [0, 0],
                ]

            tr_prev_output_v = np.pad(
                tr_prev_output_v, paddings_per_axis, mode="constant", constant_values=0
            )

        if is_channel_first:
            num_kernels = int(prev_output.shape[1])
        else:
            assert layer_config["data_format"] == "channels_last", layer_config[
                "data_format"
            ]
            num_kernels = int(prev_output.shape[-1])
        assert num_kernels == self.layer_weights.shape[2], "{} vs {}".format(
            num_kernels, self.layer_weights.shape[2]
        )

        if is_channel_first:
            input_shape = [int(v) for v in prev_output.shape[2:]]
        else:
            input_shape = [int(v) for v in prev_output.shape[1:-1]]

        n_mv_0 = int(
            (input_shape[0] - kernel_shape[0] + 2 * paddings[0]) / strides[0] + 1
        )
        n_mv_1 = int(
            (input_shape[1] - kernel_shape[1] + 2 * paddings[1]) / strides[1] + 1
        )

        n_output_channel = self.layer_weights.shape[-1]
        from_front = []

        for idx_ol in tqdm(range(n_output_channel)):
            for i in range(n_mv_0):
                for j in range(n_mv_1):
                    curr_prev_output_slice = tr_prev_output_v[
                        :, i * strides[0] : i * strides[0] + kernel_shape[0], :, :
                    ]
                    curr_prev_output_slice = curr_prev_output_slice[
                        :, :, j * strides[1] : j * strides[1] + kernel_shape[1], :
                    ]
                    output = (
                        curr_prev_output_slice * self.layer_weights[:, :, :, idx_ol]
                    )
                    sum_output = np.sum(np.abs(output))
                    output = output / sum_output
                    sum_output = np.nan_to_num(output, posinf=0.0)
                    output = np.mean(output, axis=0)
                    from_front.append(output)

        from_front = np.asarray(from_front)
        if is_channel_first:
            from_front = from_front.reshape(
                (
                    n_output_channel,
                    n_mv_0,
                    n_mv_1,
                    kernel_shape[0],
                    kernel_shape[1],
                    int(prev_output.shape[1]),
                )
            )
        else:
            from_front = from_front.reshape(
                (
                    n_mv_0,
                    n_mv_1,
                    n_output_channel,
                    kernel_shape[0],
                    kernel_shape[1],
                    int(prev_output.shape[-1]),
                )
            )

        from_front = np.moveaxis(from_front, [0, 1, 2], [3, 4, 5])
        from_behind = self.compute_gradient_to_output(inputs_outputs[0])

        forward_impacts = from_front * from_behind

        if is_channel_first:
            forward_impacts = np.sum(np.sum(forward_impacts, axis=-1), axis=-1)
        else:
            forward_impacts = np.sum(np.sum(forward_impacts, axis=-2), axis=-2)

        grad = self.compute_gradient_to_loss(inputs_outputs)

        pairs = np.asarray([grad.flatten(), forward_impacts.flatten()]).T
        return {"shape": forward_impacts.shape, "costs": pairs}
