# Schedule Free Optimizers in Keras


## Keras2 versus Keras3

There are schedule free optimizers included here for both Keras-2 and
Keras-3. However most of the recent clean up and experimentation I've
done has been directed at the Keras-3 version, so there may be bugs
or inefficiences in the Keras-2 version that are fixed in the Keras-3
version.


## Description

This implements SGD and Adam schedule free optimizers described in
[The Road Less Scheduled]
(https://doi.org/10.48550/arXiv.2405.15682). 

The optimizer sets the weights to `y` from the paper, but at inference
time the weights should be instead set to `x`. For models that do not
use batch normalization–or other layers that incorporate running
averages–the `ScheduleFreeCallback` can be used as shown in
`fashion_mnist_example_keras3.py`. This sets the weights to `x` during
each evaluation, for the Keras-2 version, and when training is done. In
the Keras-3 version, weights are automatically set to their `x` version
at the end of training.

However, if batch normalization is used, the statistics for the
BatchNorm need to be updated when the weights are updated. One approach
is to set the weights to `x` at training completion and run for a few
epochs with the learning rate set to zero as shown in
`fashion_mnist_example_batchnorm.py`. Another approach is to simply not
update the weights, the results are often good enough when using
the `y` values as the final weights.

The examples in `fashion_mnist_example_keras3.py` produce the following
training losses at 20 epochs:

|  Optimizer   |    Train Loss    | Test Accuracy |
|-------------:|:----------------:|:---------_---:|
|          SGD |       0.313      |     0.901     |
|         Adam |       0.182      |     0.923     |
|       SF-SGD |       0.228      |     0.921     |
|      SF-Adam |       0.086      |     0.939     |   


These are run with the same parameters for both the Keras and Schedule
Free versions except that the schedule free versions had warmup steps
equal to about 1 epoch (10000). In all cases the best training loss and
test accuracy were recorded, the best accuracy sometimes occurred
before the end of training.
