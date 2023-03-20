import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist

nonmutated_model = keras.models.load_model("trained_models/mnist_conv_trained.h5")
mutated_model = keras.models.load_model("mutants/mnist_conv_patched_0.h5")

img_rows, img_cols = (28, 28)
num_classes = 10


def generate_training_data():
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()

    if K.image_data_format() == "channels_first":
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


ACCURACY_THRESHOLD = 0.8


def test_case(model, X, Y):
    total = len(Y)
    y_pred = model.predict(X)

    y_pred = np.argmax(y_pred, axis=1)
    y_actual = np.argmax(Y, axis=1)

    correct = np.sum(y_pred == y_actual)
    accuracy = correct / total

    return accuracy


(x_train, y_train), (x_test, y_test) = generate_training_data()

filtered_x_test = []
filtered_y_test = []
for i in range(len(x_test)):
    if np.argmax(y_test[i]) == 0:
        filtered_x_test.append(x_test[i])
        filtered_y_test.append(y_test[i])

filtered_x_test = np.array(filtered_x_test)
filtered_y_test = np.array(filtered_y_test)

filtered_nonmutated_accuracy = test_case(
    nonmutated_model, filtered_x_test, filtered_y_test
)

nonmutated_accuracy = test_case(nonmutated_model, x_test, y_test)


filtered_mutated_accuracy = test_case(mutated_model, filtered_x_test, filtered_y_test)


mutated_accuracy = test_case(mutated_model, x_test, y_test)

if filtered_nonmutated_accuracy < ACCURACY_THRESHOLD:
    print(
        f"✗ Nonmutated model has accuracy {filtered_nonmutated_accuracy} on test data with only 0s"
    )
else:
    print(
        f"✓ Nonmutated model has accuracy {filtered_nonmutated_accuracy} on test data with only 0s"
    )

if nonmutated_accuracy < ACCURACY_THRESHOLD:
    print(
        f"✗ Nonmutated model has accuracy {nonmutated_accuracy} on test data excluding 0s"
    )
else:
    print(
        f"✓ Nonmutated model has accuracy {nonmutated_accuracy} on test data excluding 0s"
    )

if filtered_mutated_accuracy < ACCURACY_THRESHOLD:
    print(
        f"✓ Mutated model has accuracy {filtered_mutated_accuracy} on test data with only 0s"
    )
else:
    print(
        f"✗ Mutated model has accuracy {filtered_mutated_accuracy} on test data with only 0s"
    )

if mutated_accuracy < ACCURACY_THRESHOLD:
    print(f"✗ Mutated model has accuracy {mutated_accuracy} on test data excluding 0s")
else:
    print(f"✓ Mutated model has accuracy {mutated_accuracy} on test data excluding 0s")
