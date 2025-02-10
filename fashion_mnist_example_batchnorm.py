import tf_keras  # pyright: ignore

from fashion_mnist_example_base import test_optimizer
from schedule_free_optimizers import AdamScheduleFree

if __name__ == "__main__":
    print("Schedule Free Adam")
    test_optimizer(
        AdamScheduleFree(learning_rate=0.01, weight_decay=0.004, warmup_steps=1000),
        tf_keras,
        fix_norms=True,
    )
