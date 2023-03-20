from abc import ABC, abstractmethod

import numpy as np
from keras.engine.sequential import Sequential


class Searcher(ABC):
    def __init__(self, model: Sequential) -> None:
        self.model = model

    @abstractmethod
    def search(self) -> None:
        pass
