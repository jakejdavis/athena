from .localiser import Localiser

from keras.engine.sequential import Sequential
from numpy import ndarray
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from keras.layers import Normalization

class C2DLocaliser(Localiser):
    def __init__(self, model: Sequential, layer_index: int, layer_weights: ResourceVariable, norm_scaler: Normalization) -> None:
        super().__init__(model, layer_index, layer_weights)

    def compute_gradient_loss(self, inputs: ndarray) -> int:
        pass

    def compute_fi_gl(self, inputs: ndarray) -> int:
        pass
