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

from typing import Optional

import tensorflow as tf
from tf_keras import callbacks, optimizers


class ScheduleFreeCallback(callbacks.Callback):
    def on_test_begin(self, logs=None):
        self.orig_weights = self.model.optimizer.finalize_weights(self.model)

    def on_test_end(self, logs=None):
        self.model.optimizer.set_weights(self.orig_weights)

    def on_train_end(self, logs=None):
        self.orig_weights = self.model.optimizer.finalize_weights(self.model)


class BaseScheduleFree(optimizers.Optimizer):
    schedule_weight_exponent = 2.0

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: Optional[int] = None,
        jit_compile: bool = True,
        decay: str = 'z_at_y',
        name: str = 'BaseScheduleFree',
        **kwargs,
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            weight_decay=0.0,
            **kwargs,
        )
        self.sf_weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.decay = decay

    def finalize_weights(self, model):
        old_weights = []
        new_weights = []
        for var in model.trainable_weights:
            beta = self.get_beta()
            var_index = self._index_dict[self._var_key(var)]
            z_t = self.z_t[var_index]
            y = var.numpy()
            beta_x = y - (1 - beta) * z_t
            new_weights.append((var, beta_x / beta))
            old_weights.append((var, y))
        self.set_weights(new_weights)
        return old_weights

    def set_weights(self, new_weights):
        for var, weights in new_weights:
            var.assign(weights)

    def _step_dense(self, *, y, z, gradient, c, gamma, lambda_):
        # y = beta * x + (1 - beta) * z =>
        beta = tf.cast(self.get_beta(), y.dtype)
        beta_x = y - (1 - beta) * z

        if self.decay == 'z_at_y':
            # Default approach suggested in the paper because "equivalent to L2"
            z.assign_sub(gamma * lambda_ * y)
        elif self.decay == 'z_at_z':
            # Alternative approach mentioned in paper
            z.assign_sub(gamma * lambda_ * z)
        else:
            raise ValueError(f'unknown value for `decay`: "{self.decay}"')

        z.assign_sub(gamma * gradient)
        beta_x = (1 - c) * beta_x + c * beta * z
        y.assign(beta_x + (1 - beta) * z)

    def _compute_schedule_and_c(self, y):
        var_index = self._index_dict[self._var_key(y)]
        dtype = y.dtype
        k_plus_1 = tf.cast(self.iterations + 1, dtype)
        warmup_steps_plus_1 = tf.cast(self.warmup_steps + 1, dtype)
        schedule = tf.minimum(k_plus_1, warmup_steps_plus_1) / warmup_steps_plus_1

        weight = schedule**self.schedule_weight_exponent
        weight_sum = self.weight_sum[var_index]
        weight_sum.assign_add(weight)
        c = weight / weight_sum

        return schedule, c

    def get_weight_decay(self, y):
        if self._use_weight_decay(y):
            return tf.cast(self.sf_weight_decay, y.dtype)
        else:
            return 0.0

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self.weight_sum = []
        for var in var_list:
            self.weight_sum.append(self.add_variable((), dtype=var.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'warmup_steps': self.warmup_steps,
                'weight_decay': self.sf_weight_decay,
                'decay': self.decay,
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
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: Optional[int] = None,
        jit_compile: bool = True,
        name: str = 'SGDScheduleFree',
        **kwargs,
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            **kwargs,
        )

        if isinstance(momentum, (int, float)) and not 0 <= momentum <= 1:
            raise ValueError('`momentum` must be between [0, 1].')

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.momentum = momentum

    def build(self, var_list):
        """Initialize optimizer variables"""

        super().build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self.z_t = []
        add_var = self.add_variable_from_reference
        for var in var_list:
            self.z_t.append(
                add_var(model_variable=var, variable_name='z_t', initial_value=var)
            )
        self._built = True

    def get_beta(self):
        return self.momentum

    def update_step(self, gradient, y):
        """Update variable given gradient"""
        schedule, c = self._compute_schedule_and_c(y)

        gamma = schedule * tf.cast(self.learning_rate, y.dtype)
        lambda_ = self.get_weight_decay(y)
        var_index = self._index_dict[self._var_key(y)]
        z_t = self.z_t[var_index]

        if isinstance(gradient, tf.IndexedSlices):
            # # Sparse gradients.
            raise NotImplementedError()
        else:
            self._step_dense(
                y=y,
                z=z_t,
                gradient=gradient,
                c=c,
                gamma=gamma,
                lambda_=lambda_,
            )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'learning_rate': self._serialize_hyperparameter(self._learning_rate),
                'momentum': self.momentum,
            }
        )
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


    Other args are passed onto Optimizer, so see the Keras docks for
    their definition.

    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: Optional[int] = None,
        jit_compile: bool = True,
        name: str = 'AdamScheduleFree',
        **kwargs,
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            **kwargs,
        )

        if isinstance(beta_2, (int, float)) and not 0 <= beta_2 <= 1:
            raise ValueError('`beta_2` must be between [0, 1].')
        if isinstance(beta_2, (int, float)) and not 0 <= beta_2 <= 1:
            raise ValueError('`beta_2` must be between [0, 1].')

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables"""
        super().build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self.z_t = []
        self.v_t = []
        self.sum_t = []
        add_var = self.add_variable_from_reference
        for var in var_list:
            self.z_t.append(
                add_var(model_variable=var, variable_name='z_t', initial_value=var)
            )
            self.v_t.append(add_var(model_variable=var, variable_name='v_t'))
            self.sum_t.append(self.add_variable((), dtype=var.dtype))
        self._built = True

    def get_beta(self):
        return self.beta_1

    def update_step(self, gradient, y):
        """Update variable given gradient"""
        schedule, c = self._compute_schedule_and_c(y)

        beta_2 = schedule * tf.cast(self.beta_2, y.dtype)
        gamma = schedule * tf.cast(self.learning_rate, y.dtype)
        lambda_ = self.get_weight_decay(y)

        var_index = self._index_dict[self._var_key(y)]
        z_t = self.z_t[var_index]
        v_t = self.v_t[var_index]
        sum_t = self.sum_t[var_index]

        sum_t.assign(beta_2 * sum_t + (1 - beta_2))

        if isinstance(gradient, tf.IndexedSlices):
            # # Sparse gradients.
            raise NotImplementedError()
        else:
            # Dense gradients
            v_t.assign(beta_2 * v_t + (1 - beta_2) * gradient**2)
            sigma = tf.math.sqrt(v_t / sum_t + self.epsilon**2)
            normalized_gradient = gradient / sigma
            self._step_dense(
                y=y,
                z=z_t,
                gradient=normalized_gradient,
                c=c,
                gamma=gamma,
                lambda_=lambda_,
            )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'learning_rate': self._serialize_hyperparameter(self._learning_rate),
                'beta_1': self.beta_1,
                'beta_2': self.beta_2,
                'epsilon': self.epsilon,
            }
        )
        return config
