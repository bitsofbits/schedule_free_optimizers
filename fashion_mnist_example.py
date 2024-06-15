import numpy as np
from keras import layers
from schedule_free_optimizers import AdamScheduleFree, SGDScheduleFree
from tensorflow import keras

num_classes = 10
input_shape = (28, 28, 1)


def get_training_data():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_trains = []
    padded = np.pad(x_train, [(0, 0), (1, 1), (1, 1)])
    print(padded.shape)
    _, m, n = x_train.shape
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            cropped = padded[:, 1 + di : m + 1 + di, 1 + dj : n + 1 + dj]
            x_trains.append(cropped)
            x_trains.append(cropped[:, ::-1])
    x_train = np.concatenate(x_trains, axis=0)
    y_train = np.concatenate([y_train] * 18, axis=0)

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def build_model():
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.Conv2D(32, kernel_size=(3, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.Conv2D(64, kernel_size=(3, 3)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def test_optimizer(optimizer):
    x_train, y_train, x_test, y_test = get_training_data()

    model = build_model()
    print(model.summary())

    batch_size = 128
    epochs = 10

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )


if __name__ == '__main__':
    print('Keras Adam')
    test_optimizer(
        keras.optimizers.Adam(learning_rate=0.03, amsgrad=True, weight_decay=0.004)
    )
    print('Schedule Free Adam')
    test_optimizer(
        AdamScheduleFree(learning_rate=0.03, amsgrad=True, weight_decay=0.004)
    )
    print("Keras SGD")
    test_optimizer(keras.optimizers.SGD(learning_rate=0.01, weight_decay=0.004))
    print("Schedule Free SGD")
    test_optimizer(SGDScheduleFree(learning_rate=0.1, weight_decay=0.004))
