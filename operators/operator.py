from abc import ABC, abstractmethod

from keras.engine.sequential import Sequential


class Operator(ABC):
    """
    Abstract class for operators.
    """

    def __init__(self, model: Sequential, additional_config) -> None:
        self.model = model
        self.additional_config = additional_config

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Sequential:
        raise NotImplementedError
