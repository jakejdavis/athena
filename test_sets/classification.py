import numpy as np
from numpy import ndarray
from tensorflow.keras.models import Sequential

import utils.config

from .test_set import TestSet


class ClassificationTestSet(TestSet):
    def __init__(self, additional_config) -> None:
        super().__init__(additional_config)

    def run(self, model: Sequential, X: ndarray, Y: ndarray) -> float:
        return model.evaluate(X, Y, verbose=0)[1]

    def test_passed(self, test_result: float) -> bool:
        return test_result >= float(
            utils.config.get_config_val(
                self.additional_config, "test_set.accuracy_threshold", 0.8
            )
        )
