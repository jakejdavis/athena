import logging
import sys
from typing import Type

from .boston import BostonModel
from .fashionmnist import FashionMNISTModel
from .mnist_conv import MNISTConvModel


def get_model(name: str) -> Type[FashionMNISTModel]:
    if name.lower() == "fashionmnist":
        return FashionMNISTModel
    if name.lower() == "mnist_conv":
        return MNISTConvModel
    if name.lower() == "boston":
        return BostonModel
    else:
        logging.error(f"Model {name} not found")
        sys.exit(1)
