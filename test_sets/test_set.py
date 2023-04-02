from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray
from tensorflow.keras.models import Sequential


class TestSet(ABC):
    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config

    @abstractmethod
    def run(self, model: Sequential, X: ndarray, Y: ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def test_passed(self, test_result: float) -> bool:
        raise NotImplementedError
