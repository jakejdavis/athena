from abc import ABC, abstractmethod

from keras.engine.sequential import Sequential


class Model(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def generate_inputs_outputs(self) -> None:
        pass
