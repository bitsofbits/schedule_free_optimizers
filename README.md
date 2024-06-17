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

N.B.: These need to be fully updated

|  Optimizer  | Keras Train Loss | SF Train Loss| Keras Test Accuracy | SF Test Accuracy |
| -----------:|:----------------:|:------------:|:-------------------:|:----------------:|
|         SGD |       0.282      |     0.182    |        0.866        |     0.927       |
|        Adam |       0.234      |     0.168    |        0.918        |     0.928       |       

These are run with the same parameters between the Keras and Schedule
Free versions. Note that setting warmup_steps to, e.g., 1000, can
improve the Schedule Free versions performance, but was omitted to make
the runs comparable.