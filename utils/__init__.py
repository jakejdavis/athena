from keras import Sequential
import keras.datasets.mnist
import numpy as np
from numpy import ndarray
from typing import Tuple
import logging
import keras.engine.sequential

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

    while next_point_index < len(costs):
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

def generate_mnist_inputs(model: keras.engine.sequential.Sequential, dataset: Tuple[ndarray, ndarray], n: int = 100) -> Tuple[ndarray, ndarray]:
    i_pos = []
    i_neg = []

    inputs, outputs = dataset

    (img_rows, img_cols) = (28, 28)
    num_classes = 10

    inputs = inputs.reshape(inputs.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    inputs = inputs.astype('float32')
    inputs /= 255

    while len(i_pos) < n and len(i_neg) < n:
        # if model predicts random input correctly add to i_pos
        random_index = np.random.randint(0, len(inputs))
        data = inputs[random_index].reshape(1, img_rows, img_cols, 1)
        prediction = model.predict(data, verbose=0)

        if np.argmax(prediction) == np.argmax(outputs[random_index]):
            if len(i_pos) < n:
                i_pos.append(inputs[random_index])
        else:
            if len(i_neg) < n:
                i_neg.append(inputs[random_index])

    logging.info("Generated %d positive and negative examples" % len(i_neg))

    return np.array(i_pos), np.array(i_neg)

def generate_inputs(name: str, model: Sequential, dataset: None = None) -> Tuple[ndarray, ndarray]:
    if name == "mnist_conv":
        _, test_dataset = keras.datasets.mnist.load_data()
        return generate_mnist_inputs(model, test_dataset)

    assert False, "Unknown dataset"
 
