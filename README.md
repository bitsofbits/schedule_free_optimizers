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

|  Optimizer  | Keras Train Loss | SF Train Loss| Keras Test Accuracy | SF Test Accuracy |
| -----------:|:----------------:|:------------:|:-------------------:|:----------------:|
|         SGD |       0.182      |     0.147    |        0.920        |      0.932       |
|        Adam |       0.318      |     0.145    |        0.894        |      0.930       |       

These are run with the same parameters between the Keras and Schedule
Free versions. In all cases the best training loss and test accuracy
were recorded.  Adding warmup  to the schedule free versions further
improves the performance (e.g., `warmup_steps=1000` increases the test
accuracy 0.932 and 0.933 for SGD and Adam respectively)
