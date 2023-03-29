import logging

import numpy as np
import tensorflow as tf
from keras.engine.sequential import Sequential
from numpy import ndarray
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Model
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tqdm import tqdm

from .localiser import Localiser


class FCLocaliser(Localiser):
    """
    Localiser for Dense layers.
    """

    def __init__(
        self,
        model: Sequential,
        layer_index: int,
        layer_weights: ResourceVariable,
        norm_scaler: Normalizer,
    ) -> None:
        super().__init__(model, layer_index, layer_weights, norm_scaler)

    def compute_fi_gl(self, inputs_outputs: ndarray):
        """
        Calculates the forward impact and gradient loss for a Dense layer.

        This function is based on https://github.com/coinse/arachne/blob/cc1523ce4a9ad6dbbecdf87e1f5712a72e48a393/arachne/run_localise.py#L325

        :param inputs_outputs: inputs and outputs to use for calculating FL, GL
        :return: forward impact and gradient loss
        """

        logging.info("Calculating forward impact for FC layer")
        logging.info(f"- Layer index: {self.layer_index}")
        logging.info(
            f"- Input shape: {self.model.layers[self.layer_index].input_shape}"
        )
        logging.info(
            f"- Output shape: {self.model.layers[self.layer_index].output_shape}"
        )

        inputs, _ = inputs_outputs
        if self.layer_index == 0:
            prev_output = inputs
        else:
            target_model = Model(
                inputs=self.model.input,
                outputs=self.model.layers[self.layer_index - 1].output,
            )
            prev_output = target_model.predict(inputs, verbose=0)

        if len(prev_output.shape) == 3:
            prev_output = prev_output.reshape(
                prev_output.shape[0], prev_output.shape[-1]
            )

        logging.debug(f"Previous output shape: {prev_output.shape}")
        assert (
            int(prev_output.shape[-1]) == self.layer_weights.shape[0]
        ), "{} vs {}".format(int(prev_output.shape[-1]), self.layer_weights.shape[0])

        from_front = []
        for idx in tqdm(range(self.layer_weights.shape[-1])):
            assert (
                int(prev_output.shape[-1]) == self.layer_weights.shape[0]
            ), "{} vs {}".format(
                int(prev_output.shape[-1]), self.layer_weights.shape[0]
            )

            output = np.multiply(prev_output, self.layer_weights[:, idx])

            output = np.abs(output)
            output = self.norm_scaler.fit_transform(output)
            output = np.mean(output, axis=0)
            from_front.append(output)

        from_front = np.asarray(from_front)
        from_front = from_front.T

        logging.info("Calculating gradient from behind")
        from_behind = self.compute_gradient_to_output(inputs_outputs[0])

        forward_impacts = from_front * from_behind
        grad = self.compute_gradient_to_loss(inputs_outputs)

        pairs = np.asarray([grad.flatten(), forward_impacts.flatten()]).T

        return {"shape": forward_impacts.shape, "costs": pairs}
