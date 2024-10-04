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
        self.orig_weights = self.model.optimizer.finalize_weights(self.model)

    def on_test_end(self, logs=None):
        self.model.optimizer.set_weights(self.orig_weights)

    def on_train_end(self, logs=None):
        self.orig_weights = self.model.optimizer.finalize_weights(self.model)


class BaseScheduleFree(optimizers.Optimizer):
    sf_schedule_weight_exponent = 2.0

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        decay_type: str = 'z_at_y',
        name: str = 'BaseScheduleFree',
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=0.0, name=name, **kwargs
        )
        self.sf_warmup_steps = warmup_steps
        self.sf_weight_decay = weight_decay
        self.sf_decay_type = decay_type

    def finalize_weights(self, model):
        old_weights = []
        new_weights = []
        for var in model.trainable_weights:
            y = var.numpy()
            z = self.sf_z_t[self._get_variable_index(var)].numpy()
            beta = self.get_beta()
            x = (y - (1 - beta) * z) / beta
            new_weights.append((var, x))
            old_weights.append((var, y))
        self.set_weights(new_weights)
        return old_weights

    def set_weights(self, new_weights):
        for var, weights in new_weights:
            self.assign(var, weights)

    def sf_step(self, *, y, z, gradient, c, gamma):
        # y = beta * x + (1 - beta) * z =>
        beta = ops.cast(self.get_beta(), y.dtype)
        one_minus_beta = ops.subtract(1.0, beta)
        # beta_x = y - (1 - beta) * z
        beta_x = ops.subtract(y, ops.multiply(one_minus_beta, z))

        lambda_ = self.get_weight_decay(y)
        scaled_lambda = ops.multiply(gamma, lambda_)
        if self.sf_decay_type == 'z_at_y':
            # Default approach suggested in the paper because "equivalent to L2"
            self.assign_sub(z, ops.multiply(scaled_lambda, y))
        elif self.sf_decay_type == 'z_at_z':
            # Alternative approach mentioned in paper
            self.assign_sub(z, ops.multiply(scaled_lambda, z))
        else:
            raise ValueError(f'unknown value for `decay`: "{self.sf_decay_type}"')

        self.assign_sub(z, ops.multiply(gamma, gradient))
        # beta_x = (1 - c) * beta_x + c * beta * z
        one_minus_c = ops.subtract(1.0, c)
        beta_z = ops.multiply(beta, z)
        beta_x = ops.add(
            ops.multiply(one_minus_c, beta_x),
            ops.multiply(c, beta_z),
        )
        # y <- beta_x + (1 - beta) * z
        self.assign(y, ops.add(beta_x, ops.multiply(one_minus_beta, z)))

    def get_z_schedule_and_c(self, y):
        var_index = self._get_variable_index(y)
        z_t = self.sf_z_t[var_index]
        self.assign(
            z_t, ops.cond(ops.equal(self.iterations, 0), lambda: y, lambda: z_t)
        )

        k_plus_1 = ops.cast(ops.add(self.iterations, 1), y.dtype)
        warmup_steps_plus_1 = ops.cast(ops.add(self.sf_warmup_steps, 1), y.dtype)
        schedule = ops.divide(
            ops.minimum(k_plus_1, warmup_steps_plus_1), warmup_steps_plus_1
        )

        weight = ops.power(schedule, self.sf_schedule_weight_exponent)
        weight_sum = self.sf_weight_sum[var_index]
        self.assign_add(weight_sum, weight)
        c = ops.divide(weight, weight_sum)

        return z_t, schedule, c

    def get_weight_decay(self, y):
        if self._use_weight_decay(y):
            return ops.cast(self.sf_weight_decay, y.dtype)
        else:
            return 0.0

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.sf_weight_sum = []
        self.sf_z_t = []
        for var in var_list:
            z = self.add_variable_from_reference(reference_variable=var, name='z_t')
            self.sf_z_t.append(z)
            self.sf_weight_sum.append(self.add_variable(shape=(), dtype=var.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'warmup_steps': self.sf_warmup_steps,
                'weight_decay': self.sf_weight_decay,
                'decay': self.sf_decay,
            }
        )


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
        self, *, momentum: float = 0.9, name: str = 'SGDScheduleFree', **kwargs
    ):
        super().__init__(name=name, **kwargs)
        if isinstance(momentum, (int, float)) and not 0 <= momentum <= 1:
            raise ValueError('`momentum` must be between [0, 1].')
        self.sf_momentum = momentum

    def get_beta(self):
        return self.sf_momentum

    def update_step(self, gradient, y, learning_rate):
        """Update variable given gradient"""
        z, schedule, c = self.get_z_schedule_and_c(y)
        learning_rate = ops.cast(learning_rate, y.dtype)
        gamma = ops.multiply(schedule, learning_rate)
        self.sf_step(y=y, z=z, gradient=gradient, c=c, gamma=gamma)

    def get_config(self):
        config = super().get_config()
        config.update({'momentum': self.sf_momentum})
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
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        name: str = 'AdamScheduleFree',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if isinstance(beta_1, (int, float)) and not 0 <= beta_1 <= 1:
            raise ValueError('`beta_1` must be between [0, 1].')
        if isinstance(beta_2, (int, float)) and not 0 <= beta_2 <= 1:
            raise ValueError('`beta_2` must be between [0, 1].')

        self.sf_beta_1 = beta_1
        self.sf_beta_2 = beta_2
        self.sf_epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables"""
        if self.built:
            return
        super().build(var_list)
        self.sf_v_t = []
        self.sf_sum_t = []
        add_var = self.add_variable_from_reference
        for var in var_list:
            self.sf_v_t.append(add_var(reference_variable=var, name='v_t'))
            self.sf_sum_t.append(self.add_variable((), dtype=var.dtype))
        self._built = True

    def get_beta(self):
        return self.sf_beta_1

    def update_step(self, gradient, y, learning_rate):
        """Update variable given gradient"""
        z_t, schedule, c = self.get_z_schedule_and_c(y)

        beta_2 = ops.multiply(schedule, ops.cast(self.sf_beta_2, y.dtype))
        gamma = ops.multiply(schedule, ops.cast(learning_rate, y.dtype))

        sum_t = self.sf_sum_t[self._get_variable_index(y)]
        v_t = self.sf_v_t[self._get_variable_index(y)]

        # sum_t <- beta_2 * sum_t + (1 - beta_2)
        new_sum_t = ops.add(ops.multiply(beta_2, sum_t), ops.subtract(1, beta_2))
        self.assign(sum_t, new_sum_t)

        # v_t <- beta_2 * v_t + (1 - beta_2) * gradient**2)
        one_minus_beta_2 = ops.subtract(1.0, beta_2)
        grad_squared = ops.power(gradient, 2)
        self.assign(
            v_t,
            ops.add(
                ops.multiply(beta_2, v_t), ops.multiply(one_minus_beta_2, grad_squared)
            ),
        )
        # sigma = sqrt(v_t / sum_t + sf_epsilon**2)
        eps_squared = ops.cast(ops.power(self.sf_epsilon, 2), y.dtype)
        sigma = ops.sqrt(ops.add(ops.divide(v_t, sum_t), eps_squared))
        normalized_gradient = ops.divide(gradient, sigma)
        self.sf_step(y=y, z=z_t, gradient=normalized_gradient, c=c, gamma=gamma)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'beta_1': self.sf_beta_1,
                'beta_2': self.sf_beta_2,
                'epsilon': self.sf_epsilon,
            }
        )
        return config
