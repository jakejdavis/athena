from .searcher import Searcher
from scipy.optimize import differential_evolution
import logging
import numpy as np
import random
import tensorflow as tf
import keras


class DE(Searcher):
    """
    Use differential evolution to search for a patch of weights which improves the models
    fitness on negative inputs and keeps the fitness on positive inputs the same.
    """

    def __init__(self, model, pos_neg, weights_to_target):
        super().__init__(model)

        (self.i_neg, self.o_neg), (self.i_pos, self.o_pos) = pos_neg
        self.o_neg = keras.utils.to_categorical(self.o_neg)
        self.o_pos = keras.utils.to_categorical(self.o_pos)

        self.weights_to_target = weights_to_target
        self.alpha = 0.5

        logging.info(f"Starting with fitness {self.fitness([])}")

    def apply_patch(self, patched_weights):
        new_model = self.model

        # Ignoring bias, using tf.tensor_scatter_nd_update
        for i, (layer_index, (neuron_index, weight_index)) in enumerate(
            self.weights_to_target
        ):
            new_model.weights[layer_index].assign(
                tf.tensor_scatter_nd_update(
                    new_model.weights[layer_index],
                    [[neuron_index, weight_index]],
                    [patched_weights[i]],
                )
            )

        return new_model

    def search(self):
        # Set bounds to +/- 100% of the original weights
        bounds = []
        for i, (layer_index, (neuron_index, weight_index)) in enumerate(
            self.weights_to_target
        ):
            lower = (
                self.model.layers[layer_index].get_weights()[0][neuron_index][
                    weight_index
                ]
                * -1
            )
            higher = (
                self.model.layers[layer_index].get_weights()[0][neuron_index][
                    weight_index
                ]
                * 1
            )

            if higher < lower:
                higher, lower = lower, higher
            bounds.append((lower, higher))

        result = differential_evolution(
            self.fitness,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=False,
            polish=True,
            init="latinhypercube",
            atol=0,
        )

        return result

    def score(self, model, inputs_outputs):
        inputs, outputs = inputs_outputs

        predictions = model.predict(inputs)

        _score = 0

        for i, prediction in enumerate(predictions):
            if np.argmax(prediction) == np.argmax(outputs[i]):
                _score += 1
            else:
                # TODO: change to support other loss functions
                # Categorical cross entropy
                loss = np.sum(
                    -outputs[i] * np.log(prediction + 1e-9), axis=-1, keepdims=True
                )
                _score += 1 / (1 + loss)

        return _score

    def fitness(self, weights):
        if len(weights) > 0:
            self.new_model = self.apply_patch(weights)
        else:
            logging.debug("No weights to apply")
            self.new_model = self.model

        neg_fitness = self.score(self.new_model, (self.i_neg, self.o_neg))
        pos_fitness = self.score(self.new_model, (self.i_pos, self.o_pos))

        total_fitness = pos_fitness + self.alpha * neg_fitness

        logging.debug(f"Fitness with weights {weights}: {total_fitness}")

        return total_fitness
