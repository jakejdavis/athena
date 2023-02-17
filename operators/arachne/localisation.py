from keras.engine.compile_utils import LossesContainer
import numpy as np
import logging
import tensorflow as tf
from keras.layers import Normalization

import sys
sys.path.append("../")
import utils
import utils.model_utils

from . import localisers

from keras.engine.sequential import Sequential
from numpy import ndarray
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from typing import Callable, List, Union

def compute_gradient_loss(weight: ResourceVariable, M: Sequential, inputs: ndarray, loss_func: Callable) -> int:
    with tf.GradientTape() as tape:
        tape.watch(weight)
        loss = loss_func(M(inputs), M(inputs))

    return tape.gradient(loss, weight)

def compute_forward_impact(weight: ResourceVariable, M: Sequential, inputs: ndarray) -> int:
    return 0        

def extract_pareto(pool: List[Union[ResourceVariable, float]]) -> None:
    """function which extracts the pareto front from a pool of solutions"""
    # pareto_pool = [tuple(v) for v in np.asarray(pool, dtype = object)]
    # using utils.is_pareto_efficient
    pass

def bidirectional_localisation(model: Sequential, i_neg: ndarray, i_pos: ndarray) -> None:
    """
        M: a keras model 
        i_neg: a set of inputs that reveal the fault
        i_pos: a set of inputs that do not reveal the fault
    """

    assert isinstance(model.loss, Callable), "Loss function is not callable"
    loss_func: Callable = model.loss.__call__
    
    pool = {}
   
    i_pos_indices = np.random.choice(len(i_pos), len(i_neg), replace=False)
    i_pos = i_pos[i_pos_indices]

    logging.debug("Input shape: {}".format(i_neg.shape))

    norm_scaler = Normalization()

    for layer_index, layer_weights in enumerate(model.weights):
        layer_type = utils.model_utils.get_layer_type(layer_weights.name)
        logging.debug("Layer type: {}".format(layer_type))

        if layer_type == "C2D": 
            logging.debug("Skipping layer: {}".format(layer_type))
            continue

        localiser = localisers.get_localiser(layer_type)(model, layer_index, layer_weights, norm_scaler)

        fwd_imp_neg = localiser.compute_forward_impact(i_neg)
        fwd_imp_pos = localiser.compute_forward_impact(i_pos)

        fwd_imp = fwd_imp_neg / (1 + fwd_imp_pos)

        logging.info("Forward impact: {}".format(fwd_imp))
        
        grad_loss_neg = -localiser.compute_gradient_loss(i_neg)
        grad_loss_pos = -localiser.compute_gradient_loss(i_pos)

        grad_loss = grad_loss_neg / (1 + grad_loss_pos)

        logging.info("Gradient loss: {}".format(grad_loss))


        pool[layer_weights.name] = (grad_loss, fwd_imp)

        """grad_loss_neg = -compute_gradient_loss(weight, M, i_neg, loss_func)
        grad_loss_pos = -compute_gradient_loss(weight, M, i_pos, loss_func)

        grad_loss = grad_loss_neg / (1 + grad_loss_pos)

        logging.debug("Gradient loss: {}".format(grad_loss))

        fwd_imp_neg = compute_forward_impact(weight, M, i_neg)
        fwd_imp_pos = compute_forward_impact(weight, M, i_pos)

        fwd_imp = fwd_imp_neg / (1 + fwd_imp_pos)

        pool[weight] = (grad_loss, fwd_imp)"""

    return None#extract_pareto(pool)
