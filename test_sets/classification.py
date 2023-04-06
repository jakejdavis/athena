from numpy import ndarray
from tensorflow.keras.models import Sequential

import utils.config

from .test_case import TestCase, TestSet


class ClassificationTestCase(TestCase):
    """
    Test case for classification models.

    :param additional_config: Additional configuration for test case.
    """

    name = "Example classification test case"

    def __init__(self, additional_config) -> None:
        super().__init__(additional_config)

    def run(self, model: Sequential, X: ndarray, Y: ndarray) -> float:
        """
        Returns the accuracy of the model on the given inputs and outputs.

        :param model: The model to test.
        :param X: The inputs to test the model with.
        :param Y: The outputs to test the model with.
        """
        return model.evaluate(X, Y, verbose=0)[1]

    def test_passed(self, test_result: float) -> bool:
        """
        Checks whether the test case passed given the output of self.run().

        :param test_result: The output of self.run().
        :return: Whether the test case passed.
        """
        return test_result >= float(
            utils.config.get_config_val(
                self.additional_config, "test_set.accuracy_threshold", 0.8
            )
        )


class ClassificationTestSet(TestSet):
    """
    Test set for classification models.
    """

    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config
        self.test_cases = [ClassificationTestCase(additional_config)]
