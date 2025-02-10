import keras

from fashion_mnist_example_base import test_optimizer
from schedule_free_optimizers_keras3 import AdamScheduleFree

if __name__ == "__main__":
    print("Schedule Free Adam")
    test_optimizer(
        AdamScheduleFree(learning_rate=0.01, weight_decay=0.004, warmup_steps=1000),
        keras,
        fix_norms=True,
    )
