import numpy as np
from numpy import ndarray
from tensorflow.keras.models import Sequential

import utils.config

from .test_case import TestCase, TestSet


class ClassificationTestCase(TestCase):
    name = "Example classification test case"

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


class ClassificationTestSet(TestSet):
    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config
        self.test_cases = [ClassificationTestCase(additional_config)]
