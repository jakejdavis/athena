import logging
import os
from multiprocessing import Queue
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.engine.sequential import Sequential
from numpy import int64, ndarray
from scipy.optimize import differential_evolution

from utils.config import get_config_val
from utils.model_utils import is_classification

from .fitness_plotter import FitnessPlotter
from .searcher import Searcher

Nfeval = 1


def score(
    model: Sequential, inputs_outputs: Tuple[ndarray, ndarray], variance=None
) -> float:
    """
    Score for model on inputs and outputs.

    :param model: Model to score
    :param inputs_outputs: Inputs and outputs to score on
    :param variance: Used in regression, variance of a correct prediction
    """
    inputs, outputs = inputs_outputs

    predictions = model(inputs)

    _score = 0.0

    for i, prediction in enumerate(predictions):
        prediction_correct = (
            is_classification(model) and np.argmax(prediction) == np.argmax(outputs[i])
        ) or (not is_classification(model) and abs(prediction - outputs[i]) < variance)

        if prediction_correct:
            _score += 1.0
        else:
            _score += 1.0 / (1.0 + model.loss(outputs[i], prediction))

    return _score


def fitness_score(
    patched_model: Sequential,
    pos,
    neg,
    variance: float,
    alpha: float,
) -> float:
    """
    Fitness score for a model.

    :param patched_model: Model to score
    :param pos: Positive inputs and outputs
    :param neg: Negative inputs and outputs
    :param variance: Used in regression, variance of a correct prediction
    :param alpha: Weighting of negative fitness
    """
    neg_fitness = score(patched_model, (neg[0], neg[1]), variance)
    pos_fitness = score(patched_model, (pos[0], pos[1]), variance)

    return pos_fitness + alpha * neg_fitness


def fitness(
    weights: ndarray,
    model: Union[Sequential, str],
    pos,
    neg,
    pos_trivial,
    neg_trivial,
    weights_to_target: List[Tuple[int64, Tuple[int64, int64]]],
    trivial_weighting: float,
    alpha: float,
    variance: float,
) -> ndarray:
    """
    Fitness function for differential evolution.

    :param weights: Weights to apply to the model
    :param model: Model to apply weights to
    :param pos: Positive inputs and outputs
    :param neg: Negative inputs and outputs
    :param pos_trivial: Positive trivial inputs and outputs
    :param neg_trivial: Negative trivial inputs and outputs
    :param weights_to_target: Weights to target
    :param trivial_weighting: Weighting of trivial fitness
    :param alpha: Weighting of negative fitness
    :param variance: Used in regression, variance of a correct prediction
    """

    if isinstance(model, str):
        # For multiprocessing, load model from file
        model = tf.keras.models.load_model(model)
        model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=["accuracy"],
        )

    if len(weights) > 0:
        new_model = apply_patch(model, weights, weights_to_target)
    else:
        logging.debug("No weights to apply")
        new_model = model

    total_fitness = fitness_score(new_model, pos, neg, variance, alpha)

    # Maxmimise fitness of outputs not being targetted
    if pos_trivial is not None:
        total_fitness += (
            -1
            * trivial_weighting
            * fitness_score(new_model, pos_trivial, neg_trivial, variance, alpha)
        )

    return total_fitness


def apply_patch(model, patched_weights: ndarray, weights_to_target) -> Sequential:
    for i, (layer_index, neuron) in enumerate(weights_to_target):
        model.layers[layer_index].weights[0].assign(
            tf.tensor_scatter_nd_update(
                model.layers[layer_index].weights[0],
                [neuron],
                [patched_weights[i]],
            )
        )

    return model


class DE(Searcher):
    """
    Use differential evolution to search for a patch of weights which improves the models
    fitness on negative inputs and keeps the fitness on positive inputs the same.
    """

    TEMP_MODEL_PATH = "cache/model_mp.h5"

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

        self.alpha = get_config_val(
            self.additional_config, "operator.searcher.fitness.alpha", 0.8, float
        )
        self.variance = get_config_val(
            config=self.additional_config, key="correct_variance", default=1, type=float
        )

        self.trivial_weighting = get_config_val(
            self.additional_config,
            "operator.searcher.fitness.trivial_weighting",
            None,
            float,
        )

        if self.trivial_weighting is None:
            logging.info("Weighting trivial fitness for initial fitness ~= 0")
            standard_fitness_score = fitness_score(
                self.model, self.pos, self.neg, self.variance, self.alpha
            )

            trivial_fitness_score = fitness_score(
                self.model,
                self.trivial_pos,
                self.trivial_neg,
                self.variance,
                self.alpha,
            )

            self.trivial_weighting = standard_fitness_score / trivial_fitness_score
            logging.info(f"Trivial weighting: {self.trivial_weighting}")

        self.plot = get_config_val(
            self.additional_config, "operator.searcher.plot.show", False, bool
        )

        if self.plot:
            self.fitness_plot = FitnessPlotter(Queue())
            self.fitness_plot.start()

            # Initial plot is by default disabled as there will usually
            # be a significant difference between the initial fitness and the
            # fitness after the first iteration
            if get_config_val(
                self.additional_config,
                "operator.searcher.plot.initial_fitness",
                False,
                bool,
            ):
                initial_fitness = fitness(
                    np.array([]),
                    self.model,
                    self.pos,
                    self.neg,
                    self.trivial_pos,
                    self.trivial_neg,
                    self.weights_to_target,
                    self.trivial_weighting,
                    self.alpha,
                    self.variance,
                )

                logging.info(f"Starting with fitness {initial_fitness}")
                self.fitness_plot.update(initial_fitness)

    def __del__(self) -> None:
        if self.plot:
            self.fitness_plot.update(None)
            self.fitness_plot.join()

        os.remove(self.TEMP_MODEL_PATH)

    def callback_print(self, Xi, convergence):
        """
        Callback function to print current best fitness and weights.

        :param Xi: Current best weights
        :param convergence: Current convergence
        """
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
            self.variance,
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

        if self.plot:
            self.fitness_plot.update(xi_fitness)
        Nfeval += 1

    def search(self):
        """
        Perform the search for a patch of weights which minimizes the models fitness.
        """
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

        if self.workers != 1:
            if os.path.exists(self.TEMP_MODEL_PATH):
                os.remove(self.TEMP_MODEL_PATH)
            else:
                os.makedirs(os.path.dirname(self.TEMP_MODEL_PATH), exist_ok=True)
            self.model.save(self.TEMP_MODEL_PATH)

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
                self.TEMP_MODEL_PATH if self.workers != 1 else self.model,
                self.pos,
                self.neg,
                self.trivial_pos,
                self.trivial_neg,
                self.weights_to_target,
                self.trivial_weighting,
                self.alpha,
                self.variance,
            ),
        )
