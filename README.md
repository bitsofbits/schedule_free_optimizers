# Schedule Free Optimizers in Keras

This implements SGD and Adam schedule free optimizers described in
[The Road Less Scheduled]
(https://doi.org/10.48550/arXiv.2405.15682). We use a somewhat naive
implementation of the algorithm described in the paper although we
avoid storing *x* in a manner similar to that used in the [repository]
(https://github.com/facebookresearch/schedule_free/tree/main) for the
paper.

The optimizer sets the weights to `y` from the paper, but at inference
time the weights should be instead set to `x`. For models that do not
use batch normalization–or other layers that incorporate running
averages–the `ScheduleFreeCallback` can be used as shown in
`fashion_mnist_example.py`. This sets the weights to `x` during each
evaluation and when training is done.

However, if batch normalization is used, the statistics for the
BatchNorm need to be updated when the weights are updated. One approach
is to set the weights to `x` at training completion and run for a few
epochs with the learning rate set to zero as shown in
`fashion_mnist_example_batchnorm.py`. Another approach is to simply not
update the weights, the results are often good enough when using
the `y` values as the final weights.

On the (very small) example I use here, decaying `x` for weight decay
produced better results so I'm using that as the default. The original
behavior can be recovered by setting `decay='x_at_y`` in the
constructor.

The examples in `fashion_mnist_examples.py` produce the following
training losses at 20 epochs:

|  Optimizer   |    Train Loss    | Test Accuracy |
|-------------:|:----------------:|:---------_---:|
|          SGD |       0.341      |     0.873     |
|         Adam |       0.230      |     0.911     |
|       SF-SGD |       0.268      |     0.911     |
|      SF-Adam |       0.172      |     0.919     |   


These are run with the same parameters between the Keras and Schedule
Free versions except that the schedule free versions had warmup steps
equal to about 1 epoch (10000). In all cases the best training loss and
test accuracy were recorded, the best accuracy sometimes occurred
before the end of training.
