import re
from keras.losses import binary_crossentropy, categorical_crossentropy
import logging

layer_regexes = {
    "FC": re.compile("dense*"),
    # must detect 'conv2d/kernel:0'
    "C2D": re.compile(".*conv2d*"),
    "LSTM": re.compile(".*LSTM*"),
    "Input": re.compile("InputLayer")
}

def get_layer_type(name: str) -> str:
    for k, v in layer_regexes.items():
        if v.match(name):
            return k

def get_loss_func(is_multi_label = True):
	return binary_crossentropy if is_multi_label else categorical_crossentropy 
