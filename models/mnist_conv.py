import logging

import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.engine.sequential import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers.legacy import Adam

from .model import Model


class MNISTConvModel(Model):
    def __init__(self, additional_config) -> None:
        super().__init__(additional_config)

        (self.img_rows, self.img_cols) = (28, 28)
        self.input_shape = (self.img_rows * self.img_cols,)
        self.num_classes = 10

    def generate_training_data(self):
        ((x_train, y_train), (x_test, y_test)) = mnist.load_data()

        if K.image_data_format() == "channels_first":
            x_train = x_train.reshape(x_train.shape[0], 1, self.img_rows, self.img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return (x_train, y_train), (x_test, y_test)

    def train(self) -> Sequential:

        (x_train, y_train), (x_test, y_test) = self.generate_training_data()

        batch_size = 2048
        epochs = 20

        model = Sequential()
        model.add(
            Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=self.input_shape
            )
        )
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))
        model.compile(
            optimizer=Adam(),
            loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
        )

        score = model.evaluate(x_train, y_train, verbose=0)

        logging.info("Train loss: %s", score[0])
        logging.info("Train accuracy: %s", score[1])

        return model

    def generate_inputs_outputs(
        self,
        model: Sequential,
        n: int = 20,
        specific_output: int = None,
        generic: bool = False,
    ):
        logging.info("Generating %d positive and negative examples..." % n)

        # TODO: compare between training/test data (better to use test but is there enough for a specific output?)
        inputs, outputs = keras.datasets.mnist.load_data()[1]

        inputs = inputs.reshape(inputs.shape[0], self.img_rows, self.img_cols)
        inputs = inputs.astype("float32")
        inputs /= 255

        i_pos = []
        i_neg = []
        o_pos = []
        o_neg = []

        if specific_output is not None:
            if generic:
                inputs = inputs[outputs != specific_output]
                outputs = outputs[outputs != specific_output]
            else:
                inputs = inputs[outputs == specific_output]
                outputs = outputs[outputs == specific_output]

        batch_size = 128

        while len(i_pos) < n or len(i_neg) < n:
            random_indexes = np.random.randint(0, len(inputs), batch_size)

            data = inputs[random_indexes]
            predictions = model.predict(data, verbose=0)

            for i, prediction in enumerate(predictions):
                random_index = random_indexes[i]
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

        o_pos = keras.utils.to_categorical(o_pos, self.num_classes)
        o_neg = keras.utils.to_categorical(o_neg, self.num_classes)

        return (np.array(i_pos), np.array(o_pos)), (np.array(i_neg), np.array(o_neg))

    def generate_evaluation_data(
        self, specific_output: int = None, generic: bool = False
    ):
        logging.info("Generating evaluation data...")

        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

        if specific_output is not None:
            if generic:
                x_test = x_test[y_test != specific_output]
                y_test = y_test[y_test != specific_output]
            else:
                x_test = x_test[y_test == specific_output]
                y_test = y_test[y_test == specific_output]

        if K.image_data_format() == "channels_first":
            x_test = x_test.reshape(x_test.shape[0], 1, self.img_rows, self.img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)

        x_test = x_test.astype("float32")
        x_test /= 255

        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return x_test, y_test
