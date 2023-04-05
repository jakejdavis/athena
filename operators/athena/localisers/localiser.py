import logging
import sys
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from keras.engine.sequential import Sequential
from keras.layers import Normalization
from keras.losses import binary_crossentropy, categorical_crossentropy, mae, mse
from numpy import ndarray
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import Model
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tqdm import tqdm


class Localiser(ABC):
    """
    Abstract class for localisers.
    """

    def __init__(
        self,
        model: Sequential,
        layer_index: int,
        layer_weights: ResourceVariable,
        norm_scaler: Normalization,
    ) -> None:
        self.model = model
        self.loss_func = model.loss
        self.layer_index = layer_index
        self.layer_weights = layer_weights
        self.norm_scaler = norm_scaler

    def compute_gradient_to_output(
        self, inputs: ndarray, on_weight: bool = False, by_batch: bool = True
    ) -> ndarray:
        """
        Calculates the gradient of the output of a layer with respect to the input.

        This function is based on https://github.com/coinse/arachne/blob/cc1523ce4a9ad6dbbecdf87e1f5712a72e48a393/arachne/run_localise.py#LL60

        :param inputs: inputs to use for calculating the gradient
        :param on_weight: whether to calculate the gradient on the weights or the output
        :param by_batch: whether to calculate the gradient by batch
        :return: gradient of the output of a layer with respect to the input
        """
        norm_scaler = Normalizer(norm="l1")

        input_dim = inputs.shape[0]

        if not on_weight:
            target = self.model.layers[self.layer_index].output
        else:  # on weights
            target = self.model.layers[self.layer_index].weights[
                :-1
            ]  # exclude the bias

        # since this might cause OOM error, divide them
        num = inputs.shape[0]
        if by_batch:
            batch_size = 64
            num_split = int(np.round(num / batch_size))
            if num_split == 0:
                num_split += 1
            chunks = np.array_split(np.arange(num), num_split)
        else:
            chunks = [np.arange(num)]

        if not on_weight:
            output_shape = tuple([input_dim] + [int(v) for v in target.shape[1:]])
            gradient = np.zeros(output_shape)

            layer_output = target
            new_model = Model(
                inputs=self.model.input, outputs=[self.model.output, target]
            )
            for chunk in chunks:
                input_chunk = tf.convert_to_tensor(inputs[chunk])

                with tf.GradientTape() as tape:
                    tape.watch(input_chunk)
                    final_output, layer_output = new_model(input_chunk)

                gradients = tape.gradient(final_output, layer_output)
                gradient[chunk] = gradients.numpy()

            gradient = np.abs(gradient)
            reshaped_gradient = gradient.reshape(gradient.shape[0], -1)
            norm_gradient = norm_scaler.fit_transform(reshaped_gradient)
            mean_gradient = np.mean(norm_gradient, axis=0)
            ret_gradient = mean_gradient.reshape(gradient.shape[1:])

            return ret_gradient
        else:
            raise NotImplementedError

    def compute_gradient_to_loss(
        self,
        inputs_outputs: ndarray,
        by_batch: bool = True,
    ) -> ndarray:
        """
        Calculates the gradient of the loss with respect to the input.

        This function is based on https://github.com/coinse/arachne/blob/cc1523ce4a9ad6dbbecdf87e1f5712a72e48a393/arachne/run_localise.py#L127

        :param inputs_outputs: inputs and outputs to use for calculating the gradient
        :param by_batch: whether to calculate the gradient by batch
        :return: gradient of the loss with respect to the input
        """

        inputs, outputs = inputs_outputs
        outputs = outputs.astype(np.int32)
        targets = self.model.layers[self.layer_index].weights[:-1]

        # is multi label
        if len(self.model.output.shape) == 3:
            y_tensor = tf.keras.Input(
                shape=(self.model.output.shape[-1],), name="labels"
            )
        else:
            y_tensor = tf.keras.Input(
                shape=list(self.model.output.shape)[1:], name="labels"
            )

        num = inputs.shape[0]
        if by_batch:
            batch_size = 64
            num_split = int(np.round(num / batch_size))
            if num_split == 0:
                num_split += 1
            chunks = np.array_split(np.arange(num), num_split)
        else:
            chunks = [np.arange(num)]

        if self.model.loss == categorical_crossentropy:
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_tensor, logits=self.model.output, name="per_label_loss"
            )
        elif self.model.loss == binary_crossentropy:
            loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_tensor, logits=self.model.output, name="per_label_loss"
            )
        else:
            logging.error("Loss function not implemented")
            sys.exit(1)

        new_model = Model(
            inputs=[self.model.input, y_tensor],
            outputs=[self.model.output, loss_tensor],
        )

        gradients = [[] for _ in range(len(targets))]
        for chunk in chunks:
            input_chunk = tf.convert_to_tensor(inputs[chunk])
            output_chunk = tf.convert_to_tensor(outputs[chunk])

            with tf.GradientTape() as tape:
                tape.watch(input_chunk)
                tape.watch(output_chunk)
                _, loss = new_model([input_chunk, output_chunk])

            curr_gradients = tape.gradient(loss, targets)

            for i, _gradient in enumerate(curr_gradients):
                gradients[i].append(_gradient)

        for i, gradients_p_chunk in enumerate(gradients):
            gradients[i] = np.abs(np.sum(np.asarray(gradients_p_chunk), axis=0))

        logging.debug("Gradients shape: {}".format(np.array(gradients).shape))

        return np.array(gradients[0]) if len(gradients) == 1 else np.array(gradients)

    @abstractmethod
    def compute_fi_gl(self, inputs: ndarray) -> int:
        pass
