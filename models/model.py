from abc import ABC, abstractmethod

from keras.engine.sequential import Sequential


class Model(ABC):
    """
    Abstract class for models.
    """

    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config

    @abstractmethod
    def train(self) -> Sequential:
        """
        Trains the model.

        :return: The trained model.
        """
        pass

    @abstractmethod
    def generate_inputs_outputs(
        self,
        model: Sequential,
        n: int = 20,
        specific_output: int = None,
        generic: bool = False,
    ):
        """
        Generates inputs and outputs for the model.

        :param model: The model to generate inputs and outputs for.
        :param n: The number of inputs to generate.
        :param specific_output: The specific output to generate inputs for.
        :param generic: Whether to generate generic inputs.
        """

        pass

    @abstractmethod
    def generate_evaluation_data(
        self, specific_output: int = None, generic: bool = False
    ):
        """
        Generates evaluation data for testing the model.

        :param specific_output: The specific output to generate inputs for.
        :param generic: Whether to generate generic inputs.
        """

        pass
