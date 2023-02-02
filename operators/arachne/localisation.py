import numpy as np
import logging

import sys
sys.path.append("../")
import utils
import utils.model_utils

def compute_gradient_loss(weight, M, inputs, loss_func):
    return 0
    #return utils.partial_derivative(loss_func, inputs, weight)

def compute_forward_impact(weight, M, inputs):
    return 0        

def extract_pareto(pool):
    """function which extracts the pareto front from a pool of solutions"""
    # pareto_pool = [tuple(v) for v in np.asarray(pool, dtype = object)]
    pass

def bidirectional_localisation(M, i_neg, i_pos):
    """
        M: a keras model 
        i_neg: a set of inputs that reveal the fault
        i_pos: a set of inputs that do not reveal the fault
    """

    loss_func = M.loss
    
    pool = []
   
    i_pos_indices = np.random.choice(len(i_pos), len(i_neg), replace=False)
    i_pos = i_pos[i_pos_indices]

    for weight in M.weights:
        grad_loss_neg = - compute_gradient_loss(weight, M, i_neg, loss_func)
        grad_loss_pos = - compute_gradient_loss(weight, M, i_pos, loss_func)

        grad_loss = grad_loss_neg / (1 + grad_loss_pos)

        fwd_imp_neg = compute_forward_impact(weight, M, i_neg)
        fwd_imp_pos = compute_forward_impact(weight, M, i_pos)

        fwd_imp = fwd_imp_neg / (1 + fwd_imp_pos)

        pool += (weight, grad_loss, fwd_imp)

    return extract_pareto(pool)
