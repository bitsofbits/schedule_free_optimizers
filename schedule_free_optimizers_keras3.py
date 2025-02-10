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
        self.orig_weights = self.model.optimizer.replace_y_with_x(
            self.model.trainable_variables
        )

    def on_test_end(self, logs=None):
        assert self.model is not None
        self.model.optimizer.set_weights(self.orig_weights)


class BaseScheduleFree(optimizers.Optimizer):
    schedule_weight_exponent = 2.0

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        polynomial_average_exponent: float = 0.0,
        name: str = "BaseScheduleFree",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=0.0, name=name, **kwargs
        )
        assert not self.use_ema, "ScheduleFreeOptimizers do not support EMA"
        self.warmup_steps = warmup_steps
        self.polynomial_average_exponent = polynomial_average_exponent
        self.sf_weight_decay = weight_decay

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

    def setup_update_steps(self, trainable_variables: list):
        """Do any updates that we want to perform once per call of self.update"""

        # Update _schedule, _weight_sum and _c
        def initialize():
            for y in trainable_variables:  # pyright: ignore
                z = self._zs[self._get_variable_index(y)]
                self.assign(z, y)

        ops.cond(
            self._iterations == 0,
            initialize,
            lambda: None,
        )

        k_plus_1 = ops.cast(ops.add(self.iterations, 1), "float32")

        warmup_steps_plus_1 = ops.cast(ops.add(self.warmup_steps, 1), "float32")
        self.assign(
            self._schedule,
            ops.divide(ops.minimum(k_plus_1, warmup_steps_plus_1), warmup_steps_plus_1),
        )

        weight_sum = self._weight_sum
        weight = ops.power(k_plus_1, self.polynomial_average_exponent)
        self.assign_add(weight_sum, weight)

        self.assign(self._c, ops.divide(weight, weight_sum))

    def _backend_apply_gradients(self, grads, trainable_variables):
        # Override to inject `setup_update_steps`which allows us to
        # update global values before starting per weight updates.
        self.setup_update_steps(trainable_variables)
        super()._backend_apply_gradients(grads, trainable_variables)

    def finalize_variable_values(self, var_list):
        self.replace_y_with_x(var_list)

    def replace_y_with_x(self, var_list):
        """Replace y-weights with more accurate x-weights

        Schedule free optimizers use `y` weights during
        optimization, but the actual target weights are
        `x`, so at the end of training, or when computing
        metrics, replace 'y' with 'x'
        """
        old_weights = []
        new_weights = []
        for var in var_list:
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
        lambda_ = self.get_weight_decay(y)
        if lambda_ is not None:
            gradient = gradient + lambda_ * y

        beta = ops.cast(self.get_beta(), y.dtype)
        # This is based on the the PyTorch implementation from the paper
        self.assign_add(y, c * (z - y) + gamma * (beta * (1 - c) - 1) * gradient)
        self.assign_sub(z, gamma * gradient)

    def get_weight_decay(self, y):
        if self._use_weight_decay(y):
            return ops.cast(self.sf_weight_decay, y.dtype)
        else:
            return None

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "warmup_steps": self.warmup_steps,
                "weight_decay": self.sf_weight_decay,
                "polynomial_average_exponent": self.polynomial_average_exponent,
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
        schedule = ops.cast(self._schedule, y.dtype)
        c = ops.cast(self._c, y.dtype)
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

    def get_beta(self):
        return self.beta_1

    def update_step(self, gradient, y, learning_rate):
        """Update variable given gradient"""
        ndx = self._get_variable_index(y)
        z = self._zs[ndx]
        v = self._vs[ndx]
        schedule = ops.cast(self._schedule, y.dtype)
        c = ops.cast(self._c, y.dtype)

        beta_2 = ops.cast(self.beta_2, y.dtype)
        gamma = ops.multiply(schedule, ops.cast(learning_rate, y.dtype))

        k_plus_1 = ops.cast(self.iterations + 1, y.dtype)  # pyright:ignore
        bias_correction2 = 1 - beta_2**k_plus_1  # pyright:ignore

        self.assign_add(
            v,
            (1 - beta_2) * (ops.square(gradient) - v),  # pyright:ignore
        )
        scale = ops.sqrt(bias_correction2 / (v + self.epsilon**2))  # pyright:ignore
        self.sf_step(y=y, z=z, gradient=scale * gradient, c=c, gamma=gamma)

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
