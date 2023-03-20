import logging

import keras
import numpy as np
from keras.datasets import fashion_mnist
from keras.engine.sequential import Sequential
from keras.layers import Activation, Dense
from tensorflow.keras.optimizers.legacy import Adam

from .model import Model


class FashionMNISTModel(Model):
    def __init__(self) -> None:
        super().__init__()

        (self.img_rows, self.img_cols) = (28, 28)
        self.input_shape = (self.img_rows * self.img_cols,)

    def generate_training_data(self):
        ((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

        num_classes = 10

        x_train = x_train.reshape(x_train.shape[0], self.img_rows * self.img_cols)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows * self.img_cols)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return (x_train, y_train), (x_test, y_test)

    def train(self, model_name: str) -> Sequential:
        logging.info("Training model %s", model_name)

        (x_train, y_train), (x_test, y_test) = self.generate_training_data()

        batch_size = 128
        epochs = 12

        model = Sequential()
        model.add(Dense(100, input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))
        model.compile(
            optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
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

        logging.info("Test loss: %s", score[0])
        logging.info("Test accuracy: %s", score[1])

        return model

    def generate_inputs_outputs(
        self,
        model: keras.engine.sequential.Sequential,
        n: int = 20,
        specific_output: int = None,
    ):
        logging.info("Generating %d positive and negative examples..." % n)

        # TODO: compare between training/test data (better to use test but is there enough for a specific output?)
        inputs, outputs = fashion_mnist.load_data()[0]

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
            if specific_output is not None:
                if outputs[random_index] != specific_output:
                    continue

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
