import importlib
import logging
import sys
from typing import Type

from .boston import BostonModel
from .fashionmnist import FashionMNISTModel
from .mnist_conv import MNISTConvModel
from .model import Model


def get_model(name: str) -> Type[Model]:
    """
    Get concrete model by name.

    :param name: Name of the model
    """

    if name.lower() == "fashionmnist":
        return FashionMNISTModel
    if name.lower() == "mnist_conv":
        return MNISTConvModel
    if name.lower() == "boston":
        return BostonModel

    logging.debug(f"Model {name} not found, trying to import dynamically")
    return get_model_dynamic(name)


def get_model_dynamic(name: str) -> Type[Model]:
    """
    Get concrete model by name dynamically using importlib.
    Expects file models/<name>.py to exist with a class which inherits from Model.

    :param name: Name of the model
    """

    # Use importlib to dynamically import the model
    module = importlib.import_module("models." + name.lower())

    # Return class in module of type Model
    for _class in module.__dict__.values():
        if isinstance(_class, type) and issubclass(_class, Model) and _class != Model:
            return _class

    # If no class was found, raise error
    logging.error(f"Model {name} not found")
    sys.exit(1)
