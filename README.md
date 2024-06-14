# Schedule Free Optimizers in Keras

This implements SGD and Adam schedule free optimizers described in 
[The Road Less Scheduled](https://doi.org/10.48550/arXiv.2405.15682). We use a
naive implementation of the algorithm described in the paper and store both
*x* and *z* which differs somewhat from the way this was implemented in
the [repository](https://github.com/facebookresearch/schedule_free/tree/main) 
for the paper.

