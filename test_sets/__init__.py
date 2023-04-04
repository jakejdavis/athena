import importlib
import logging
import sys
from typing import Type

from test_sets.test_case import TestCase, TestSet

from .classification import ClassificationTestSet


def get_test_set(test_set_name) -> Type[TestSet]:
    """
    Get test set by name.

    :param name: Name of set set
    """
    if test_set_name == "classification":
        return ClassificationTestSet

    logging.debug(f"Test set {test_set_name} not found, trying to import dynamically")
    return get_test_set_dynamic(test_set_name)


def get_test_set_dynamic(name: str) -> Type[TestSet]:
    """
    Get concrete test set by name dynamically using importlib.
    Expects file test_sets/<name>.py to exist with a class which inherits from TestSet.

    :param name: Name of the test set
    """

    # Use importlib to dynamically import the test set
    module = importlib.import_module("test_sets." + name.lower())

    # Return class in module of type TestSet
    for _class in module.__dict__.values():
        if (
            isinstance(_class, type)
            and issubclass(_class, TestSet)
            and _class != TestSet
        ):
            return _class

    # If no class was found, raise error
    logging.error(f"Test set {name} not found")
    sys.exit(1)
