import os
import re

import tensorflow as tf
from keras.models import Sequential

layer_regexes = {
    "FC": re.compile("dense*"),
    "C2D": re.compile(".*conv2d*"),
    "LSTM": re.compile(".*LSTM*"),
    "Input": re.compile("InputLayer"),
}


def get_layer_type(name: str) -> str:
    """
    Get layer type by name.

    :param name: Name of layer given by Keras
    :return: Type of layer
    """
    for k, v in layer_regexes.items():
        if v.match(name):
            return k


TEMP_MODEL_PATH = "cache/safe_clone_model.h5"


def safe_clone_model(model: Sequential) -> Sequential:
    """
    Clone a model and its weights by saving it to disk and loading it again.

    :param model: Model to clone
    :return: Cloned model
    """
    model.save(TEMP_MODEL_PATH)

    cloned_model = tf.keras.models.load_model(TEMP_MODEL_PATH)
    cloned_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics,
    )
    os.remove(TEMP_MODEL_PATH)

    return cloned_model
