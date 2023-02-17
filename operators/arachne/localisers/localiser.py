from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np

from keras.engine.sequential import Sequential
from numpy import ndarray
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from keras.layers import Normalization

class Localiser(ABC):
    def __init__(self, model: Sequential, layer_index: int, layer_weights: ResourceVariable, norm_scaler: Normalization) -> None:
        self.model = model
        self.loss_func = model.loss.__call__
        self.layer_index = layer_index
        self.layer_weights = layer_weights
        self.norm_scaler = norm_scaler

    def compute_gradient_to_output(self) -> ndarray:
        return np.ones((self.layer_weights.shape[0], self.layer_weights.shape[1]))

    @abstractmethod
    def compute_gradient_loss(self, inputs: ndarray) -> int:
        pass

    @abstractmethod
    def compute_forward_impact(self, inputs: ndarray) -> int:
        pass