import logging
import sys

from .classification import ClassificationTestSet


def get_test_set(test_set_name):
    if test_set_name == "classification":
        return ClassificationTestSet
    else:
        logging.error(f"Test set {test_set_name} not implemented")
        sys.exit(1)
