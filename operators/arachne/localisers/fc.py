from .localiser import Localiser

from tqdm import tqdm
import numpy as np
import logging

from tensorflow.keras.models import Model
from keras.engine.sequential import Sequential
from numpy import ndarray
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from keras.layers import Normalization

class FCLocaliser(Localiser):
    def __init__(self, model: Sequential, layer_index: int, layer_weights: ResourceVariable, norm_scaler: Normalization) -> None:
        super().__init__(model, layer_index, layer_weights, norm_scaler)

    def compute_gradient_loss(self, inputs: ndarray) -> int:
        return 0

    def compute_forward_impact(self, inputs: ndarray) -> int:
        if self.layer_index == 0 or self.layer_index == len(self.model.layers) - 1:
            prev_output = inputs
        else:
            target_model = Model(inputs=self.model.input, outputs=self.model.layers[self.layer_index].output)
            prev_output = target_model.predict(inputs, verbose=0)
        
        if len(prev_output.shape) == 3:
            prev_output = prev_output.reshape(prev_output.shape[0], prev_output.shape[-1])
			
        from_front = []
        for idx in tqdm(range(self.layer_weights.shape[-1])):
            assert int(prev_output.shape[-1]) == self.layer_weights.shape[0], "{} vs {}".format(
                int(prev_output.shape[-1]), self.layer_weights.shape[0])
                
            output = np.multiply(prev_output, self.layer_weights[:,idx])
            output = np.abs(output)
            self.norm_scaler.adapt(output)
            output = self.norm_scaler(output) 
            output = np.mean(output, axis = 0)
            from_front.append(output) 
        
        from_front = np.asarray(from_front)
        from_front = from_front.T
        from_behind = self.compute_gradient_to_output()

        logging.debug("FI shape: {} vs {}".format(from_front.shape, from_behind.shape))

        return from_front * from_behind