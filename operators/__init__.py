import importlib
import logging
import sys

from .athena_operator import AthenaOperator
from .operator import Operator


def get_operator(operator_name):
    """
    Get concrete operator by name.

    :param operator_name: Name of the operator
    :return: Operator class
    """

    if operator_name == "athena":
        return AthenaOperator

    logging.debug(f"Operator {operator_name} not found, trying to import dynamically")
    return get_operator_dynamic(operator_name)


def get_operator_dynamic(operator_name):
    """
    Get concrete operator by name dynamically using importlib.
    Expects file operators/<name>.py OR operators/<name>_operator.py to exist containing class which inherits from Model.

    :param name: Name of the operator
    :return: Operator class
    """

    # Use importlib to dynamically import the operator
    try:
        module = importlib.import_module("operators." + operator_name.lower())
    except ModuleNotFoundError:
        logging.warn(f"Operator module operators.{operator_name} not found")
        try:
            module = importlib.import_module(
                "operators." + operator_name.lower() + "_operator"
            )
        except ModuleNotFoundError:
            logging.error(
                f"Operator module operators.{operator_name}_operator not found"
            )
            sys.exit(1)

    # Return class in module of type Operator
    for _class in module.__dict__.values():
        if (
            isinstance(_class, type)
            and issubclass(_class, Operator)
            and _class != Operator
        ):
            return _class

    # If no class was found, raise error
    logging.error(f"Operator {operator_name} not found")
    sys.exit(1)
