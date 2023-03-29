import logging
from multiprocessing import Process, Queue
from typing import List, Tuple, Union

import keras
import numpy as np
import tensorflow as tf
from keras.engine.sequential import Sequential
from numpy import int64, ndarray
from scipy.optimize import differential_evolution
from scipy.optimize._optimize import OptimizeResult

from utils.config import get_config_val
from utils.model_utils import is_classification

from .searcher import Searcher


class FitnessPlotter(Process):
    def __init__(self, queue: Queue) -> None:
        super().__init__()
        self.queue = queue
        self.fitnesses = []

    def run(self) -> None:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness over time")
        plt.show()

        while True:
            fitness = self.queue.get()
            if fitness is None:
                break
            self.fitnesses.append(fitness)
            ax.plot(self.fitnesses, "b-")
            fig.canvas.draw()
            fig.canvas.flush_events()

    def update(self, fitness: List[float]) -> None:
        self.queue.put(fitness)


Nfeval = 1


class DE(Searcher):
    """
    Use differential evolution to search for a patch of weights which improves the models
    fitness on negative inputs and keeps the fitness on positive inputs the same.
    """

    def __init__(
        self,
        model: Sequential,
        pos_neg: Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]],
        pos_neg_trivial: Tuple[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray]],
        weights_to_target: List[Tuple[int, Tuple[int64, int64]]],
        additional_config: dict = {},
    ) -> None:
        super().__init__(model)

        (self.i_neg, self.o_neg), (self.i_pos, self.o_pos) = pos_neg

        if pos_neg_trivial[0] is not None:
            (self.i_neg_trivial, self.o_neg_trivial), (
                self.i_pos_trivial,
                self.o_pos_trivial,
            ) = pos_neg_trivial

        self.weights_to_target = weights_to_target

        self.additional_config = additional_config
        self.variance = get_config_val(
            config=self.additional_config, key="correct_variance", default=1, type=float
        )
        self.trivial_weighting = get_config_val(
            config=self.additional_config,
            key="operator.searcher.fitness.trivial_weighting",
            default=0.8,
            type=float,
        )
        self.alpha = get_config_val(
            config=self.additional_config,
            key="operator.searcher.fitness.alpha",
            default=0.8,
            type=float,
        )

        initial_fitness = self.fitness([])
        logging.info(f"Starting with fitness {initial_fitness}")

        self.fitness_plot = FitnessPlotter(Queue())
        self.fitness_plot.start()

        self.fitness_plot.update(initial_fitness)

    def __del__(self) -> None:
        self.fitness_plot.update(None)
        self.fitness_plot.join()

    def apply_patch(self, patched_weights: ndarray) -> Sequential:
        new_model = self.model

        # Ignoring bias, using tf.tensor_scatter_nd_update
        for i, (layer_index, (neuron_index, weight_index)) in enumerate(
            self.weights_to_target
        ):
            new_model.layers[layer_index].weights[0].assign(
                tf.tensor_scatter_nd_update(
                    new_model.layers[layer_index].weights[0],
                    [[neuron_index, weight_index]],
                    [patched_weights[i]],
                )
            )

        return new_model

    def search(self) -> OptimizeResult:
        # Set bounds to +/- 100% of the original weights
        bounds = []

        for i, (layer_index, (neuron_index, weight_index)) in enumerate(
            self.weights_to_target
        ):
            bounds.append((-10, 10))

        def callback_print(Xi, convergence):
            global Nfeval
            xi_fitness = self.fitness(Xi)
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

            self.fitness_plot.update(xi_fitness)
            Nfeval += 1

        return differential_evolution(
            self.fitness,
            bounds,
            maxiter=100,
            popsize=15,
            tol=0.001,
            mutation=(0.5, 1),
            recombination=0.7,
            callback=callback_print,
            init="latinhypercube",
        )

    def score(
        self, model: Sequential, inputs_outputs: Tuple[ndarray, ndarray]
    ) -> float:
        inputs, outputs = inputs_outputs

        predictions = model(inputs)

        _score = 0.0

        for i, prediction in enumerate(predictions):
            prediction_correct = (
                is_classification(model)
                and np.argmax(prediction) == np.argmax(outputs[i])
            ) or (
                not is_classification(model)
                and abs(prediction - outputs[i]) < self.variance
            )

            if prediction_correct:
                _score += 1.0
            else:
                _score += 1.0 / (1.0 + model.loss(outputs[i], prediction))

        return _score

    def fitness(self, weights: ndarray) -> ndarray:
        if len(weights) > 0:
            self.new_model = self.apply_patch(weights)
        else:
            logging.debug("No weights to apply")
            self.new_model = self.model

        neg_fitness = self.score(self.new_model, (self.i_neg, self.o_neg))
        pos_fitness = self.score(self.new_model, (self.i_pos, self.o_pos))

        total_fitness = pos_fitness + self.alpha * neg_fitness

        # Maxmimise fitness of outputs not being targetted
        if hasattr(self, "i_neg_trivial"):
            neg_trivial_fitness = self.score(
                self.new_model, (self.i_neg_trivial, self.o_neg_trivial)
            )
            pos_trivial_fitness = self.score(
                self.new_model, (self.i_pos_trivial, self.o_pos_trivial)
            )

            total_fitness += (
                -1
                * self.trivial_weighting
                * (neg_trivial_fitness + self.alpha * pos_trivial_fitness)
            )

        return total_fitness
