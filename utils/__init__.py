import keras.datasets.mnist
import numpy as np
from numpy import ndarray
from typing import Tuple

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    From https://stackoverflow.com/a/40239615/8853580
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
        
def partial_derivative(func, x, var_index, epsilon=1e-5):
    shape = x.shape
    var = x[var_index]
    delta = np.zeros(shape, dtype=var.dtype)
    delta[var_index] = epsilon
    return (func(*(x + delta)) - func(*(x - delta))) / (2 * epsilon)

def generate_mnist_inputs(output: int, n: int = 100) -> Tuple[ndarray, ndarray]:
    mnist = keras.datasets.mnist.load_data()

    i_pos = np.random.choice(np.where(mnist[1][1] == output)[0], n, replace=False)
    i_neg = np.random.choice(np.where(mnist[1][1] != output)[0], n, replace=False)

    return (mnist[0][0][i_pos], mnist[0][0][i_neg])

def generate_inputs(name: str, output: int) -> Tuple[ndarray, ndarray]:
    if name == "mnist_conv":
        return generate_mnist_inputs(output)

    assert False, "Unknown dataset"
 
