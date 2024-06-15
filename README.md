# Schedule Free Optimizers in Keras

This implements SGD and Adam schedule free optimizers described in
[The Road Less Scheduled]
(https://doi.org/10.48550/arXiv.2405.15682). We use a somewhat naive
implementation of the algorithm described in the paper although we
avoid storing *x* in a manner similar to that used in the [repository]
(https://github.com/facebookresearch/schedule_free/tree/main) for the
paper.

The examples in `fashion_mnist_examples.py` produce the following
training losses at 10 epochs:

|  Optimizer  |    Keras    | Schedule Free |
| -----------:|:-----------:|--------------:|
|         SGD |     0.231   |     0.209     |
|        Adam |     0.315   |     0.175     |

These are run with the same parameters between the Keras and Schedule
Free versions. Note that setting warmup_steps to, e.g., 1000, can
improve the Schedule Free versions performance, but was omitted to make
the runs comparable.