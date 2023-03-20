import logging
import sys

import numpy as np
import tensorflow as tf
from keras.engine.compile_utils import LossesContainer
from sklearn.preprocessing import Normalizer

sys.path.append("../")
from typing import Callable, Dict, List, Optional, Tuple, TypedDict, Union

from keras.engine.sequential import Sequential
from numpy import int64, ndarray
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tqdm import tqdm

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
    M: a keras model
    i_neg: a set of inputs that reveal the fault
    i_pos: a set of inputs that do not reveal the fault
    """

    assert isinstance(model.loss, Callable), "Loss function is not callable"

    logging.debug("Input shape: {}".format(neg[0].shape))

    norm_scaler = Normalizer(norm="l1")

    fi_gl_neg = {}
    fi_gl_pos = {}

    for layer_index, layer in enumerate(model.layers):
        layer_type = utils.model_utils.get_layer_type(layer.name)
        logging.debug(f"Layer {layer_index} type: {layer_type}")

        if len(layer.weights) == 0:
            logging.debug("Skipping layer with 0 weights: {}".format(layer.name))
            continue

        layer_weights = layer.weights[0]
        logging.debug("Layer weights (kernel) shape: {}".format(layer_weights.shape))

        # TODO: add other layer types
        if layer_type == "C2D":
            logging.debug("Skipping layer C2D (NI): {}".format(layer_type))
            continue

        localiser = localisers.get_localiser(layer_type)(
            model, layer_index, layer_weights, norm_scaler
        )

        layer_fi_gl_neg = localiser.compute_fi_gl(neg)
        layer_fi_gl_pos = localiser.compute_fi_gl(pos)

        fi_gl_neg[layer_index] = layer_fi_gl_neg
        fi_gl_pos[layer_index] = layer_fi_gl_pos

    return extract_pareto(fi_gl_pos, fi_gl_neg)
