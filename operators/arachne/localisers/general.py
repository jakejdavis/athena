from typing import Type

from .c2d import C2DLocaliser
from .fc import FCLocaliser


def get_localiser(name: str) -> Type[FCLocaliser]:
    if name == "FC":
        return FCLocaliser
    elif name == "C2D":
        return C2DLocaliser
    else:
        raise NotImplementedError("Localiser for {} not implemented".format(name))
