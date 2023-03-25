import re

from keras.losses import binary_crossentropy, categorical_crossentropy

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
