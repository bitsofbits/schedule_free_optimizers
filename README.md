# Schedule Free Optimizers in Keras

This implements SGD and Adam schedule free optimizers described in
[The Road Less Scheduled]
(https://doi.org/10.48550/arXiv.2405.15682). We use a somewhat naive
implementation of the algorithm described in the paper although we
avoid storing *x* in a manner similar to that used in the [repository]
(https://github.com/facebookresearch/schedule_free/tree/main) for the
paper.

On the (very small) example I use here, decaying `x` for weight decay
produced better results so I'm using that as the default. The original
behavior can be recovered by setting `decay='x_at_y`` in the
constructor.

The examples in `fashion_mnist_examples.py` produce the following
training losses at 10 epochs:

|  Optimizer   |    Train Loss    | Test Accuracy |
|-------------:|:----------------:|:---------_---:|
|          SGD |       0.249      |     0.909     |
|         Adam |       0.300      |     0.907     |
|       SF-SGD |       0.122      |     0.930     |
|      SF-Adam |       0.115      |     0.931     |   
| SF-Half-Adam |       0.086      |     0.935     |

These are run with the same parameters between the Keras and Schedule
Free versions except that the schedule free versions had warmup steps
equal to about 1 epoch (10000). In all cases the best training loss and
test accuracy were recorded, the best accuracy usually occurred before the
end of training because this model has a tendency to overfit.

Half-Adam uses the schedule free version of Adam, with alpha set to 0.5.
Alpha interpolates between SGD and Adam, and this version fits deeper
in this case.
