from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract class for models.
    """

    def __init__(self, additional_config) -> None:
        self.additional_config = additional_config

    @abstractmethod
    def train(self) -> Sequential:
        pass

    @abstractmethod
    def generate_inputs_outputs(self) -> None:
        pass

    @abstractmethod
    def generate_evaluation_data(self) -> None:
        pass
