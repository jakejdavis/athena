import re

layer_regexes = {
    "FC": re.compile("Dense*"),
    "C2D": re.compile("Conv2D"),
    "LSTM": re.compile(".*LSTM*"),
    "Input": re.compile("InputLayer")
}

def get_layer_type(name):
    for k, v in layer_regexes.items():
        if v.match(name):
            return k

def get_loss_func(is_multi_label = True):
	"""
	here, we will only return either cross_entropy or binary_crossentropy
	"""
	loss_func = 'categorical_cross_entropy' if is_multi_label else 'binary_crossentropy'
	return loss_func 
