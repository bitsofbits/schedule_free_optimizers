# Schedule Free Optimizers in Keras

This implements SGD and Adam schedule free optimizers described in 
[The Road Less Scheduled](https://doi.org/10.48550/arXiv.2405.15682). We use a
naive implementation of the algorithm described in the paper and store both
*x* and *z* which differs somewhat from the way this was implemented in
the [repository](https://github.com/facebookresearch/schedule_free/tree/main) 
for the paper.

The examples in `fashion_mnist_examples.py` produce the following
training losses at 10 epochs:

| Optimizer   |    Keras    | Schedule Free |
| ---:        |    :----:   |     ---:      |
| SGD         |     0.231   |     0.210     |
| Adam        |     0.315   |     0.164     |

These are run with the same parameters between the Keras and Schedule
Free versions except that schedule free use a 1000 step warmup. The
schedule free example does give a significant training improvement over
the Keras version.