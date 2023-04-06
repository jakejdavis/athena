from abc import ABC, abstractmethod

from numpy import ndarray
from tensorflow.keras.models import Sequential


class TestCase(ABC):
    """
    Abstract class for test cases.
    """

    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config

    @abstractmethod
    def run(self, model: Sequential, X: ndarray, Y: ndarray):
        """
        Runs a test case on model with inputs X and outputs Y.

        :param model: The model to test.
        :param X: The inputs to test the model with.
        :param Y: The outputs to test the model with.
        :return Output of test case
        """
        raise NotImplementedError

    @abstractmethod
    def test_passed(self, test_result) -> bool:
        """
        Checks whether a test case passed given the output of self.run().

        :param test_result: The output of self.run().
        :return: Whether the test case passed.
        """
        raise NotImplementedError


class TestSet(ABC):
    """
    Abstract class for test sets.
    """

    def __init__(self, additional_config, test_cases) -> None:
        self.additional_config = additional_config
        self.test_cases = test_cases

    def get_test_cases(self) -> list:
        return self.test_cases
