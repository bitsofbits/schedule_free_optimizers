# Copyright 2024 Timothy Hochberg.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


from keras import callbacks, ops, optimizers


class ScheduleFreeCallback(callbacks.Callback):
    def on_test_begin(self, logs=None):
        assert self.model is not None
        self.orig_weights = self.model.optimizer.finalize_weights(self.model)

    def on_test_end(self, logs=None):
        assert self.model is not None
        self.model.optimizer.set_weights(self.orig_weights)

    def on_train_end(self, logs=None):
        assert self.model is not None
        self.orig_weights = self.model.optimizer.finalize_weights(self.model)


# TODO: look at finalize_variable_values in superclass


class BaseScheduleFree(optimizers.Optimizer):
    schedule_weight_exponent = 2.0

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        decay_type: str = "z_at_y",
        name: str = "BaseScheduleFree",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=0.0, name=name, **kwargs
        )
        self.warmup_steps = warmup_steps
        self.sf_weight_decay = weight_decay
        self.decay_type = decay_type

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._weight_sums = []
        self._zs = []
        for var in var_list:
            self._zs.append(
                self.add_variable_from_reference(reference_variable=var, name="z")
            )
            self._weight_sums.append(
                self.add_variable(shape=(), dtype=var.dtype, name="weight_sum")
            )

        self._weight_sum = self.add_variable(
            shape=(), dtype="float32", name="weight_sum"
        )
        self._schedule = self.add_variable((), dtype="float32")
        self._c = self.add_variable((), dtype="float32")

    def setup_update_steps(self, trainable_variables):
        schedule, c = self._get_schedule_and_c()
        self.assign(self._schedule, schedule)
        self.assign(self._c, c)

    def _backend_apply_gradients(self, grads, trainable_variables):
        self.setup_update_steps(trainable_variables)
        super()._backend_apply_gradients(grads, trainable_variables)

    def finalize_weights(self, model):
        old_weights = []
        new_weights = []
        for var in model.trainable_weights:
            y = var.numpy()
            z = self._zs[self._get_variable_index(var)].numpy()
            beta = self.get_beta()
            x = (y - (1 - beta) * z) / beta
            new_weights.append((var, x))
            old_weights.append((var, y))
        self.set_weights(new_weights)
        return old_weights

    def set_weights(self, new_weights):
        for var, weights in new_weights:
            self.assign(var, weights)

    def get_beta(self) -> float:
        raise NotImplementedError()

    def sf_step(self, *, y, z, gradient, c, gamma):
        # y = beta * x + (1 - beta) * z =>
        beta = ops.cast(self.get_beta(), y.dtype)
        # beta_x = y - (1 - beta) * z
        beta_x = ops.subtract(y, ops.multiply(ops.subtract(1, beta), z))

        lambda_ = self.get_weight_decay(y)
        scaled_lambda = ops.multiply(gamma, lambda_)
        if self.decay_type == "z_at_y":
            # Default approach suggested in the paper because "equivalent to L2"
            self.assign_sub(z, ops.multiply(scaled_lambda, y))
        elif self.decay_type == "z_at_z":
            assert False
            # Alternative approach mentioned in paper
            self.assign_sub(z, ops.multiply(scaled_lambda, z))
        else:
            raise ValueError(f'unknown value for `decay`: "{self.decay_type}"')

        self.assign_sub(z, ops.multiply(gamma, gradient))
        # beta_x = (1 - c) * beta_x + c * beta * z
        beta_z = ops.multiply(beta, z)
        beta_x = ops.add(
            ops.multiply(1 - c, beta_x),
            ops.multiply(c, beta_z),
        )
        # y <- beta_x + (1 - beta) * z
        self.assign(y, ops.add(beta_x, ops.multiply(ops.subtract(1, beta), z)))

    def _get_schedule_and_c(self):
        k_plus_1 = ops.cast(ops.add(self.iterations, 1), "float32")
        warmup_steps_plus_1 = ops.cast(ops.add(self.warmup_steps, 1), "float32")
        schedule = ops.divide(
            ops.minimum(k_plus_1, warmup_steps_plus_1), warmup_steps_plus_1
        )

        weight_sum = self._weight_sum
        weight = ops.power(schedule, self.schedule_weight_exponent)

        self.assign_add(weight_sum, weight)
        c = ops.divide(weight, weight_sum)

        return schedule, c

    def get_schedule_and_c(self, y):
        return ops.cast(self._schedule, y.dtype), ops.cast(self._c, y.dtype)

    def get_weight_decay(self, y):
        if self._use_weight_decay(y):
            return ops.cast(self.sf_weight_decay, y.dtype)
        else:
            return 0.0

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.sf_weight_decay,
                "decay_type": self.decay_type,
            }
        )
        return config


class SGDScheduleFree(BaseScheduleFree):
    """Schedule Free SGD Optimizer

    This is a naive implementation of the the Schedule Free SGD optimizer from
    [The Road Less Scheduled](https://doi.org/10.48550/arXiv.2405.15682).

    Args:
      learning_rate: How much to scale normalized gradient before adding to `x`.
      momentum: Functions similar to SGD momentum although implementation is different.
      warmup_steps: Ramp up learning rate to final value over this many steps.

    Other args are passed onto Optimizer, so see the Keras docks for
    their definition.

    """

    def __init__(
        self, *, momentum: float = 0.9, name: str = "SGDScheduleFree", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if isinstance(momentum, (int, float)) and not 0 <= momentum <= 1:
            raise ValueError("`momentum` must be between [0, 1].")
        self.momentum = momentum

    def get_beta(self):
        return self.momentum

    def update_step(self, gradient, y, learning_rate):
        """Update variable given gradient"""
        z = self._zs[self._get_variable_index(y)]
        schedule, c = self.get_schedule_and_c(y)
        learning_rate = ops.cast(learning_rate, y.dtype)
        gamma = ops.multiply(schedule, learning_rate)
        self.sf_step(y=y, z=z, gradient=gradient, c=c, gamma=gamma)

    def get_config(self):
        config = super().get_config()
        config.update({"momentum": self.momentum})
        return config


class AdamScheduleFree(BaseScheduleFree):
    """Schedule Free Adam Optimizer

    This is a naive implementation of the the Schedule Free Adam optimizer from
    [The Road Less Scheduled](https://doi.org/10.48550/arXiv.2405.15682).

    Args:
      learning_rate: How much to scale normalized gradient before adding to `x`.
      alpha: Interpolates between Adam (1) and SGD (0)
      beta_1: Functions similar to Adam's beta_1 although implementation is different.
      beta_2: How long to remember the RMS gradient values in (0, 1), 1 = forever.
      epsilon: Added to denominator to prevent division by zero.
      warmup_steps: Ramp up learning rate to final value over this many steps. We also
        ramp up beta_2 over the same period[1].

    [1] This makes early values of the gradient, where it is less stable,
        less influential on the normalization portion of Adam. This differs
        from the paper, where only the learning rate is warmed up.


    Other args are passed onto Optimizer, so see the Keras docs for
    their definition.

    """

    def __init__(
        self,
        *,
        beta_1: float = 0.9,
        beta_2: float = 0.998,
        epsilon: float = 1e-4,
        name: str = "AdamScheduleFree",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if isinstance(beta_1, (int, float)) and not 0 <= beta_1 <= 1:
            raise ValueError("`beta_1` must be between [0, 1].")
        if isinstance(beta_2, (int, float)) and not 0 <= beta_2 <= 1:
            raise ValueError("`beta_2` must be between [0, 1].")

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables"""
        if self.built:
            return
        super().build(var_list)
        self._vs = []
        for var in var_list:
            self._vs.append(
                self.add_variable_from_reference(reference_variable=var, name="v")
            )
        self._built = True

    def setup_update_steps(self, trainable_variables):
        def initialize():
            for y in trainable_variables:  # pyright: ignore
                z = self._zs[self._get_variable_index(y)]
                self.assign(z, y)

        ops.cond(
            self._iterations == 0,
            initialize,
            lambda: None,
        )
        super().setup_update_steps(trainable_variables=trainable_variables)

    def get_beta(self):
        return self.beta_1

    def update_step(self, gradient, y, learning_rate):
        """Update variable given gradient"""
        z = self._zs[self._get_variable_index(y)]
        schedule, c = self.get_schedule_and_c(y)

        assert 0 < self.beta_2 <= 1

        beta_2 = ops.cast(self.beta_2, y.dtype)
        gamma = ops.multiply(schedule, ops.cast(learning_rate, y.dtype))

        v = self._vs[self._get_variable_index(y)]

        k_plus_1 = ops.cast(ops.add(self.iterations, 1), y.dtype)
        bias = ops.subtract(1, ops.power(beta_2, k_plus_1))

        gamma = ops.multiply(ops.sqrt(bias), gamma)

        # v_t <- beta_2 * v_t + (1 - beta_2) * gradient**2)
        grad_squared = ops.square(gradient)
        self.assign(
            v,
            ops.add(
                ops.multiply(beta_2, v),
                ops.multiply(ops.subtract(1, beta_2), grad_squared),
            ),
        )
        # sigma = sqrt(v_t / sum_t) + epsilon
        sigma = ops.add(ops.sqrt(v), self.epsilon)
        normalized_gradient = ops.divide(gradient, sigma)
        self.sf_step(y=y, z=z, gradient=normalized_gradient, c=c, gamma=gamma)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
            }
        )
        return config
