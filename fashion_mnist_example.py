import tf_keras  # pyright: ignore

import schedule_free_optimizers
from fashion_mnist_example_base import test_all

if __name__ == "__main__":
    test_all(tf_keras, schedule_free_optimizers)
