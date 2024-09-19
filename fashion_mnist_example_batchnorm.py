from fashion_mnist_example import build_model, get_training_data
from schedule_free_optimizers import (AdamScheduleFree, ScheduleFreeCallback,
                                      SGDScheduleFree)
from tf_keras import layers

num_classes = 10
input_shape = (28, 28, 1)


def test_optimizer(optimizer):
    optimizer.exclude_from_weight_decay(var_names=['_wdexclude'])

    x_train, y_train, x_test, y_test = get_training_data()

    model = build_model(norm=layers.BatchNormalization)
    print(model.summary())

    batch_size = 128
    epochs = 20

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

    optimizer.finalize_weights(model)
    optimizer._learning_rate.assign(0.0)

    # Momentum for Keras BatcNormalization defaults to 0.99, so should
    # run for about 100 (1 / (1 - momentum)) batches to fix the
    # statistics. This means that 1 epoch is plenty in this case.
    batchnorm_fix_epochs = 1

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=batchnorm_fix_epochs,
        validation_data=(x_test, y_test),
    )


if __name__ == '__main__':
    print('Schedule Free Adam')
    test_optimizer(
        AdamScheduleFree(learning_rate=0.03, weight_decay=0.004, warmup_steps=10000)
    )
