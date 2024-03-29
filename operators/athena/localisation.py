import logging
from typing import Callable, Dict, List, Tuple, TypedDict

import numpy as np
from keras.engine.sequential import Sequential
from numpy import int64, ndarray
from sklearn.preprocessing import Normalizer

import utils
import utils.model_utils

from . import localisers


class LayerFIGL(TypedDict):
    costs: ndarray
    shape: Tuple[int, ...]


def extract_pareto(
    layer_fi_gl_pos: Dict[int, LayerFIGL],
    layer_fi_gl_neg: Dict[int, LayerFIGL],
) -> Tuple[List[Tuple[int, Tuple[int64, int64]]], List[Tuple[List[int], ndarray]]]:
    """
    Extracts the pareto front from forward impacts and gradient losses.

    This function is heavily based on https://github.com/coinse/arachne/blob/cc1523ce4a9ad6dbbecdf87e1f5712a72e48a393/arachne/run_localise.py#L790

    :param layer_fi_gl_pos: forward impacts and gradient losses for inputs correctly classified
    :param layer_fi_gl_neg: forward impacts and gradient losses for inputs incorrectly classified
    """

    shapes = {}
    costs_by_keys = []
    indicies_to_nodes = []
    layer_indicies = list(layer_fi_gl_neg.keys())
    for layer_index in layer_indicies:
        cost_pos = layer_fi_gl_pos[layer_index]["costs"]
        cost_neg = layer_fi_gl_neg[layer_index]["costs"]
        shapes[layer_index] = layer_fi_gl_pos[layer_index]["shape"]

        combined_costs = cost_neg / (1.0 + cost_pos)

        for i, cost in enumerate(combined_costs):
            costs_by_keys.append(([layer_index, i], cost))
            indicies_to_nodes.append(
                [layer_index, np.unravel_index(i, shapes[layer_index])]
            )

    costs = np.asarray([cost for _, cost in costs_by_keys])
    _costs = costs.copy()
    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0
    while next_point_index < len(_costs):
        nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        _costs = _costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    pareto_front = [
        tuple(efficient_point)
        for efficient_point in np.asarray(indicies_to_nodes, dtype=object)[is_efficient]
    ]

    return pareto_front, costs_by_keys


def bidirectional_localisation(model: Sequential, pos: tuple, neg: tuple) -> None:
    """
    Uses the bidirectional algorithm for patch localisation (see https://arxiv.org/abs/1912.12463).

    :param model: a keras model
    :param pos: a set of inputs that do not reveal the fault
    :param neg: a set of inputs that reveal the fault
    """

    if not isinstance(model.loss, Callable):
        logging.warning("Loss function is not callable")

    logging.debug("Input shape: {}".format(neg[0].shape))

    norm_scaler = Normalizer(norm="l1")

    fi_gl_neg = {}
    fi_gl_pos = {}

    for layer_index, layer in enumerate(model.layers):
        layer_type = utils.model_utils.get_layer_type(layer.name)
        logging.debug(f"Layer {layer_index} type: {layer_type}")

        # Skip layers with no weights (Dropout, Flatten, etc.)
        if len(layer.weights) == 0:
            logging.debug("Skipping layer with 0 weights: {}".format(layer.name))
            continue

        layer_weights = layer.weights[0]
        logging.debug("Layer weights (kernel) shape: {}".format(layer_weights.shape))

        # Use the localiser for the layer type
        localiser_class = localisers.get_localiser(layer_type)
        if localiser_class is None:
            logging.debug("Skipping layer with no localiser: {}".format(layer.name))
            continue
        localiser = localiser_class(model, layer_index, layer_weights, norm_scaler)

        layer_fi_gl_neg = localiser.compute_fi_gl(neg)
        layer_fi_gl_pos = localiser.compute_fi_gl(pos)

        fi_gl_neg[layer_index] = layer_fi_gl_neg
        fi_gl_pos[layer_index] = layer_fi_gl_pos

    return extract_pareto(fi_gl_pos, fi_gl_neg)


def random_localisation(model: Sequential, pos: tuple, neg: tuple) -> None:
    """
    Returns a random list of weights, [layer_index, (i, j)]

    :param model: a keras model
    :param pos: a set of inputs that do not reveal the fault
    :param neg: a set of inputs that reveal the fault
    """

    if not isinstance(model.loss, Callable):
        logging.warning("Loss function is not callable")

    logging.debug("Input shape: {}".format(neg[0].shape))

    N = 10

    nodes = []

    while len(nodes) < N:
        random_layer = np.random.randint(0, len(model.layers))

        layer = model.layers[random_layer]
        layer_type = utils.model_utils.get_layer_type(layer.name)

        if layer_type != "FC":
            continue

        if len(layer.weights) == 0:
            continue

        layer_weights = layer.weights[0]

        random_indexes = np.random.randint(0, layer_weights.shape[0], 2)

        nodes.append([random_layer, tuple(random_indexes)])

    return nodes
