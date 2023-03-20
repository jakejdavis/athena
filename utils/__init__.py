from keras import Sequential
import keras.datasets.mnist
import numpy as np
from numpy import ndarray
from typing import Tuple
import logging
import keras.engine.sequential
import pickle


def generate_mnist_inputs(
    model: keras.engine.sequential.Sequential,
    dataset: Tuple[ndarray, ndarray],
    n: int = 100,
) -> Tuple[ndarray, ndarray]:
    i_pos = []
    i_neg = []

    inputs, outputs = dataset

    (img_rows, img_cols) = (28, 28)
    num_classes = 10

    inputs = inputs.reshape(inputs.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    inputs = inputs.astype("float32")
    inputs /= 255

    while len(i_pos) < n and len(i_neg) < n:
        # if model predicts random input correctly add to i_pos
        random_index = np.random.randint(0, len(inputs))
        data = inputs[random_index].reshape(1, img_rows, img_cols, 1)
        prediction = model.predict(data, verbose=0)

        if np.argmax(prediction) == np.argmax(outputs[random_index]):
            if len(i_pos) < n:
                i_pos.append(inputs[random_index])
        else:
            if len(i_neg) < n:
                i_neg.append(inputs[random_index])

    logging.info("Generated %d positive and negative examples" % len(i_neg))

    return np.array(i_pos), np.array(i_neg)


def generate_mnist_inputs_outputs(
    model: keras.engine.sequential.Sequential,
    dataset: Tuple[ndarray, ndarray],
    n: int = 100,
) -> Tuple[ndarray, ndarray]:
    logging.info("Generating %d positive and negative examples..." % n)
    inputs, outputs = dataset

    (img_rows, img_cols) = (28, 28)

    inputs = inputs.reshape(inputs.shape[0], img_rows, img_cols, 1)
    inputs = inputs.astype("float32")
    inputs /= 255

    i_pos = []
    i_neg = []
    o_pos = []
    o_neg = []

    while len(i_pos) < n or len(i_neg) < n:
        random_index = np.random.randint(0, len(inputs))
        data = inputs[random_index].reshape(1, img_rows, img_cols, 1)
        prediction = model.predict(data, verbose=0)

        if np.argmax(prediction) == outputs[random_index]:
            if len(i_pos) < n:
                i_pos.append(inputs[random_index])
                o_pos.append(outputs[random_index])
                logging.debug(
                    "Generated %d positive and %d negative examples"
                    % (len(i_pos), len(i_neg))
                )
        else:
            if len(i_neg) < n:
                i_neg.append(inputs[random_index])
                o_neg.append(outputs[random_index])
                logging.debug(
                    "Generated %d positive and %d negative examples"
                    % (len(i_pos), len(i_neg))
                )

    logging.info("Done generating examples!")

    return (np.array(i_pos), np.array(o_pos)), (np.array(i_neg), np.array(o_neg))


def generate_fashionmnist_inputs_outputs(
    model: keras.engine.sequential.Sequential,
    dataset: Tuple[ndarray, ndarray],
    n: int = 100,
) -> Tuple[ndarray, ndarray]:
    logging.info("Generating %d positive and negative examples..." % n)
    inputs, outputs = dataset

    (img_rows, img_cols) = (28, 28)

    inputs = inputs.reshape(inputs.shape[0], img_rows * img_cols)
    inputs = inputs.astype("float32")
    inputs /= 255

    i_pos = []
    i_neg = []
    o_pos = []
    o_neg = []

    while len(i_pos) < n or len(i_neg) < n:
        random_index = np.random.randint(0, len(inputs))
        data = inputs[random_index].reshape(1, img_rows * img_cols)
        prediction = model.predict(data, verbose=0)[0]

        if np.argmax(prediction) == outputs[random_index]:
            if len(i_pos) < n:
                i_pos.append(inputs[random_index])
                o_pos.append(outputs[random_index])

                logging.debug(
                    "Generated %d positive and %d negative examples"
                    % (len(i_pos), len(i_neg))
                )
        else:
            if len(i_neg) < n:
                i_neg.append(inputs[random_index])
                o_neg.append(outputs[random_index])

                logging.debug(
                    "Generated %d positive and %d negative examples"
                    % (len(i_pos), len(i_neg))
                )

    logging.info("Done generating examples!")

    return (np.array(i_pos), np.array(o_pos)), (np.array(i_neg), np.array(o_neg))


def generate_inputs(
    name: str, model: Sequential, dataset: None = None
) -> Tuple[ndarray, ndarray]:
    if name == "mnist_conv":
        _, test_dataset = keras.datasets.mnist.load_data()
        return generate_mnist_inputs(model, test_dataset)

    assert False, "Unknown dataset"


def generate_inputs_outputs(
    name: str, model: Sequential, dataset: None = None
) -> Tuple[ndarray, ndarray]:
    if name == "mnist_conv":
        _, test_dataset = keras.datasets.mnist.load_data()

        # load mnist_input_outputs.pkl if exists else generate
        try:
            with open("mnist_input_outputs.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logging.info("mnist_input_outputs.pkl not found, generating...")
            inputs_outputs = generate_mnist_inputs_outputs(model, test_dataset)
            with open("mnist_input_outputs.pkl", "wb") as f:
                pickle.dump(inputs_outputs, f)
                logging.info("mnist_input_outputs.pkl saved")
            return inputs_outputs

    if name == "fashionmnist":
        _, test_dataset = keras.datasets.fashion_mnist.load_data()

        # load fashionmnist_input_outputs.pkl if exists else generate
        try:
            with open("fashionmnist_input_outputs.pkl", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            logging.info("fashionmnist_input_outputs.pkl not found, generating...")
            inputs_outputs = generate_fashionmnist_inputs_outputs(model, test_dataset)
            with open("fashionmnist_input_outputs.pkl", "wb") as f:
                pickle.dump(inputs_outputs, f)
                logging.info("fashionmnist_input_outputs.pkl saved")
            return inputs_outputs

    assert False, "Unknown dataset"
