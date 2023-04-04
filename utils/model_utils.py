import os
import re

import tensorflow as tf
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.models import Sequential

layer_regexes = {
    "FC": re.compile("dense*"),
    "C2D": re.compile(".*conv2d*"),
    "LSTM": re.compile(".*LSTM*"),
    "Input": re.compile("InputLayer"),
}


def get_layer_type(name: str) -> str:
    for k, v in layer_regexes.items():
        if v.match(name):
            return k


def is_classification(model):
    return model.loss in [binary_crossentropy, categorical_crossentropy]


TEMP_MODEL_PATH = "cache/safe_clone_model.h5"


def safe_clone_model(model: Sequential) -> Sequential:
    """
    Clone a model and its weights by saving it to disk and loading it again.
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
