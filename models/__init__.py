from typing import Type
from .fashionmnist import FashionMNISTModel
from .mnist_conv import MNISTConvModel


def get_model(name: str) -> Type[FashionMNISTModel]:
    if name.lower() == "fashionmnist":
        return FashionMNISTModel
    if name.lower() == "mnist_conv":
        return MNISTConvModel
    else:
        raise NotImplementedError("Model for {} not implemented".format(name))
