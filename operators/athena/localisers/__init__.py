from typing import Type

from .c2d import C2DLocaliser
from .fc import FCLocaliser
from .localiser import Localiser


def get_localiser(name: str) -> Type[Localiser]:
    """
    Get concrete localiser by layer type.

    :param name: Type of layer
    """

    if name == "FC":
        return FCLocaliser
    elif name == "C2D":
        return C2DLocaliser
    else:
        return None
