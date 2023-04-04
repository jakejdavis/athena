import logging

import keras
import numpy as np
from keras import backend as K
from keras.datasets import imdb
from keras.engine.sequential import Sequential
from keras.layers import Conv2D, Dense, Dropout, Embedding, Flatten
from tensorflow.keras.optimizers.legacy import Adam

from .model import Model

NUM_WORDS = 10000
MAX_LEN = 20


class IMDBModel(Model):
    def __init__(self, additional_config) -> None:
        super().__init__(additional_config)

    def generate_training_data(self):
        ((x_train, y_train), (x_test, y_test)) = imdb.load_data(num_words=NUM_WORDS)

        x_train = keras.utils.pad_sequences(x_train, maxlen=MAX_LEN)
        x_test = keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

        return (x_train, y_train), (x_test, y_test)

    def train(self) -> Sequential:
        (x_train, y_train), (x_test, y_test) = self.generate_training_data()

        batch_size = 2048
        epochs = 20

        model = Sequential()
        model.add(Embedding(NUM_WORDS, 128, input_length=MAX_LEN))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=Adam(),
            loss=keras.losses.binary_crossentropy,
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
        # model.save(os.path.join("trained_models", model_name + "_trained.h5"))
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
        inputs, outputs = keras.datasets.imdb.load_data(num_words=NUM_WORDS)[1]

        inputs = keras.utils.pad_sequences(inputs, maxlen=MAX_LEN)

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
                if (
                    outputs[random_index] == 1
                    and prediction > 0.5
                    or outputs[random_index] == 0
                    and prediction < 0.5
                ):
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

    def generate_evaluation_data(
        self, specific_output: int = None, generic: bool = False
    ):
        logging.info("Generating evaluation data...")

        (_, _), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)
        x_test = keras.utils.pad_sequences(x_test, maxlen=MAX_LEN)

        if specific_output is not None:
            if generic:
                x_test = x_test[y_test != specific_output]
                y_test = y_test[y_test != specific_output]
            else:
                x_test = x_test[y_test == specific_output]
                y_test = y_test[y_test == specific_output]

        return x_test, y_test
