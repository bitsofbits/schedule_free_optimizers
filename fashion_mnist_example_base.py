import numpy as np
# Always use dataset from keras-3
from tensorflow.keras import datasets

num_classes = 10
input_shape = (28, 28, 1)
batch_size = 64

epochs = 10


def get_training_data():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

    x_trains = []
    x_trains.append(x_train)
    x_trains.append(x_train[:, ::-1])
    x_train = np.concatenate(x_trains, axis=0)
    y_train = np.concatenate([y_train] * 2, axis=0)

    n_test = len(x_test) - (batch_size - len(x_test) % batch_size) % batch_size
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    np.random.seed(99)
    train_indices = np.arange(len(x_train))
    np.random.shuffle(train_indices)
    test_indices = np.arange(len(x_test))[:n_test]
    np.random.shuffle(test_indices)

    return (
        x_train[train_indices],
        y_train[train_indices],
        x_test[test_indices],
        y_test[test_indices],
    )


def build_model(keras, norm=None):
    layers = keras.layers
    if norm is None:
        norm = layers.LayerNormalization
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=3, padding='same'),
            layers.MaxPooling2D(pool_size=2),
            norm(),
            layers.Activation('relu'),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.MaxPooling2D(pool_size=2),
            norm(),
            layers.Activation('relu'),
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.MaxPooling2D(pool_size=2),
            norm(),
            layers.Activation('relu'),
            layers.DepthwiseConv2D(3, 3, activation='relu', depth_multiplier=2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def test_optimizer(optimizer, keras, callbacks=[]):
    x_train, y_train, x_test, y_test = get_training_data()

    model = build_model(keras)
    print(model.summary())

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        jit_compile=True,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )


def test_all(keras, schedule_free):
    print('Schedule Free Adam')
    test_optimizer(
        schedule_free.AdamScheduleFree(
            learning_rate=0.01, weight_decay=0.004, warmup_steps=1000
        ),
        keras,
        callbacks=[schedule_free.ScheduleFreeCallback()],
    )
    print('Keras Adam')
    test_optimizer(keras.optimizers.Adam(learning_rate=0.01, weight_decay=0.004), keras)
    print("Schedule Free SGD")
    test_optimizer(
        schedule_free.SGDScheduleFree(
            learning_rate=0.1, weight_decay=0.004, warmup_steps=1000
        ),
        keras,
        callbacks=[schedule_free.ScheduleFreeCallback()],
    )
    print("Keras SGD")
    test_optimizer(keras.optimizers.SGD(learning_rate=0.1, weight_decay=0.004), keras)
