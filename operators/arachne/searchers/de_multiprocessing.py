import logging
from typing import List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from keras.engine.sequential import Sequential
from numpy import int64, ndarray
from scipy.optimize import differential_evolution

from utils.config import get_config_val

from .searcher import Searcher

Nfeval = 1


def score(
    model: Sequential, inputs_outputs: Tuple[ndarray, ndarray]
) -> Union[ndarray, int]:
    inputs, outputs = inputs_outputs

    predictions = model(inputs)

    _score = 0

    for i, prediction in enumerate(predictions):
        if np.argmax(prediction) == np.argmax(outputs[i]):
            _score += 1
        else:
            # TODO: change to support other loss functions
            loss = keras.losses.categorical_crossentropy(
                outputs[i], prediction, from_logits=False
            )
            _score += 1 / (1 + loss)

    return _score


def fitness(
    weights: ndarray,
    model: Sequential,
    pos,
    neg,
    pos_trivial,
    neg_trivial,
    weights_to_target: List[Tuple[int64, Tuple[int64, int64]]],
    trivial_weighting: float,
    alpha: float,
) -> ndarray:
    if len(weights) > 0:
        new_model = apply_patch(model, weights, weights_to_target)
    else:
        logging.debug("No weights to apply")
        new_model = model

    neg_fitness = score(new_model, (neg[0], neg[1]))
    pos_fitness = score(new_model, (pos[0], pos[1]))

    total_fitness = pos_fitness + alpha * neg_fitness

    # Maxmimise fitness of outputs not being targetted
    if pos_trivial is not None:
        neg_trivial_fitness = score(new_model, (neg_trivial[0], neg_trivial[1]))
        pos_trivial_fitness = score(new_model, (pos_trivial[0], pos_trivial[1]))

        total_fitness += (
            -1 * trivial_weighting * (neg_trivial_fitness + alpha * pos_trivial_fitness)
        )

    return total_fitness


def apply_patch(model, patched_weights: ndarray, weights_to_target) -> Sequential:
    new_model = model

    # Ignoring bias, using tf.tensor_scatter_nd_update
    for i, (layer_index, (neuron_index, weight_index)) in enumerate(weights_to_target):
        new_model.layers[layer_index].weights[0].assign(
            tf.tensor_scatter_nd_update(
                new_model.layers[layer_index].weights[0],
                [[neuron_index, weight_index]],
                [patched_weights[i]],
            )
        )

    return new_model


class DE(Searcher):
    """
    Use differential evolution to search for a patch of weights which improves the models
    fitness on negative inputs and keeps the fitness on positive inputs the same.
    """

    def __init__(
        self,
        model: Sequential,
        data: Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]],
        trivial_data: Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]],
        weights_to_target: List[Tuple[int, Tuple[int64, int64]]],
        additional_config: dict = None,
        workers: int = -1,
    ) -> None:
        super().__init__(model)

        self.pos, self.neg = data
        self.trivial_pos, self.trivial_neg = trivial_data
        self.weights_to_target = weights_to_target
        self.workers = workers
        self.additional_config = additional_config
        self.trivial_weighting = get_config_val(
            self.additional_config,
            "operator.searcher.fitness.trivial_weighting",
            0.8,
            float,
        )
        self.alpha = get_config_val(
            self.additional_config, "operator.searcher.fitness.alpha", 0.8, float
        )

    def callback_print(self, Xi, convergence):
        global Nfeval
        xi_fitness = fitness(
            Xi,
            self.model,
            self.pos,
            self.neg,
            self.trivial_pos,
            self.trivial_neg,
            self.weights_to_target,
            self.trivial_weighting,
            self.alpha,
        )
        logger = logging.getLogger("athena")

        if Nfeval == 1:
            x_is = ""
            for i in range(len(Xi)):
                x_is += f"\tw{i}\t"

            logger.info(f"i {x_is}\tfitness")

        x_is = ""
        for x in Xi:
            x_is += f"\t{x: 3.6f}"

        logger.info(f"{Nfeval} {x_is}\t{xi_fitness}")

        Nfeval += 1

    def search(self):
        bounds_dist = get_config_val(
            self.additional_config, "operator.searcher.bounds_dist", 1, float
        )
        bounds = [
            (-bounds_dist, bounds_dist) for _ in range(len(self.weights_to_target))
        ]

        # Re-compile model to remove other metric functions
        self.model.compile(
            optimizer=self.model.optimizer, loss=self.model.loss, metrics=["accuracy"]
        )

        return differential_evolution(
            fitness,
            bounds,
            maxiter=get_config_val(
                self.additional_config, "operator.searcher.maxiter", 100, int
            ),
            popsize=get_config_val(
                self.additional_config, "operator.searcher.popsize", 100, int
            ),
            tol=get_config_val(
                self.additional_config, "operator.searcher.tol", 0.001, float
            ),
            mutation=(0.25, 1.5),
            recombination=0.7,
            init="latinhypercube",
            workers=self.workers,
            updating="deferred",
            callback=self.callback_print,
            args=(
                self.model,
                self.pos,
                self.neg,
                self.trivial_pos,
                self.trivial_neg,
                self.weights_to_target,
                self.trivial_weighting,
                self.alpha,
            ),
        )
