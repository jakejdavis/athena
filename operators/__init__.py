import logging
import sys

from .arachne_operators import operator_arachne


def get_operator(operator_name):
    if operator_name == "arachne":
        return operator_arachne
    else:
        logging.error(f"Operator {operator_name} not implemented")
        sys.exit(1)
