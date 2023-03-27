from abc import abstractmethod

import numpy as np
from numpy import ndarray
from tensorflow.keras.models import Sequential


class TestSet:
    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config

    @abstractmethod
    def run(self, model: Sequential, X: ndarray, Y: ndarray) -> float:
        pass
