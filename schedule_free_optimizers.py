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
from tensorflow.keras import optimizers


class BaseScheduleFree(optimizers.Optimizer):
    weight_gamma_exponent = 2.0

    @staticmethod
    def _step_dense(*, x, y, z, gradient, c, beta, gamma, lambda_):
        y.assign(beta * x + (1 - beta) * z)
        z.assign(z - gamma * gradient - gamma * lambda_ * y)
        x.assign((1 - c) * x + c * z)

    def _compute_schedule_and_c(self, y):
        var_index = self._index_dict[self._var_key(y)]
        dtype = y.dtype
        kp1 = tf.cast(self.iterations + 1, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        n = tf.minimum(kp1, warmup_steps)
        sched = n / warmup_steps

        weight = sched**self.weight_gamma_exponent
        weight_sum = self.weight_sum[var_index]
        weight_sum.assign_add(weight)
        c = weight / weight_sum

        return sched, c

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.weight_sum = []
        for var in var_list:
            self.weight_sum.append(self.add_variable((), dtype=var.dtype))

    def get_config(self):
        config = super().get_config()
        config.update(
            {'warmup_steps': self.warmup_steps, "weight_decay": self.sf_weight_decay}
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
        ema_overwrite_frequency: int = None,
        jit_compile: bool = True,
        name: str = "SGDScheduleFree",
        **kwargs
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
            **kwargs
        )

        if isinstance(momentum, (int, float)) and not 0 <= momentum <= 1:
            raise ValueError('`momentum` must be between [0, 1].')

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.sf_weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_steps = warmup_steps  # >= 0

    def build(self, var_list):
        """Initialize optimizer variables"""

        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.z_t = []
        self.x_t = []
        add_var = self.add_variable_from_reference
        for var in var_list:
            self.z_t.append(
                add_var(model_variable=var, variable_name="z_t", initial_value=var)
            )
            self.x_t.append(
                add_var(model_variable=var, variable_name="x_t", initial_value=var)
            )
        self._built = True

    def update_step(self, gradient, y):
        """Update variable given gradient"""
        schedule, c = self._compute_schedule_and_c(y)

        gamma = schedule * tf.cast(self.learning_rate, y.dtype)
        beta = tf.cast(self.momentum, y.dtype)
        lambda_ = tf.cast(self.sf_weight_decay, y.dtype)
        var_index = self._index_dict[self._var_key(y)]
        z_t = self.z_t[var_index]
        x_t = self.x_t[var_index]

        if isinstance(gradient, tf.IndexedSlices):
            # # Sparse gradients.
            raise NotImplementedError()
        else:
            self._step_dense(
                x=x_t,
                y=y,
                z=z_t,
                gradient=gradient,
                c=c,
                beta=beta,
                gamma=gamma,
                lambda_=lambda_,
            )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "momentum": self.momentum,
            }
        )
        return config


class AdamScheduleFree(BaseScheduleFree):
    """Schedule Free Adam Optimizer

    This is a naive implementation of the the Schedule Free Adam optimizer from
    [The Road Less Scheduled](https://doi.org/10.48550/arXiv.2405.15682).

    Args:
      learning_rate: How much to scale normalized gradient before adding to `x`.
      beta_1: Functions similar to Adam's beta_1 although implementation is different.
      beta_2: How long to remember the RMS gradient values in (0, 1), 1 = forever.
      epsilon: Added to denominator to prevent division by zero.
      amsgrad: Apply AMSGrad; that is, use maximum v_t rather than current v_t.
      warmup_steps: Ramp up learning rate to final value over this many steps.

    Other args are passed onto Optimizer, so see the Keras docks for
    their definition.

    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
        global_clipnorm: Optional[float] = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: Optional[int] = None,
        jit_compile: bool = True,
        name: str = "AdamScheduleFree",
        **kwargs
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
            **kwargs
        )

        if isinstance(beta_2, (int, float)) and not 0 <= beta_2 <= 1:
            raise ValueError("`beta_2` must be between [0, 1].")
        if isinstance(beta_2, (int, float)) and not 0 <= beta_2 <= 1:
            raise ValueError("`beta_2` must be between [0, 1].")

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.amsgrad = amsgrad
        self.sf_weight_decay = weight_decay

    def build(self, var_list):
        """Initialize optimizer variables"""
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.x_t = []
        self.z_t = []
        self.v_t = []
        if self.amsgrad:
            self.v_hat = []
        add_var = self.add_variable_from_reference
        for var in var_list:
            self.x_t.append(
                add_var(model_variable=var, variable_name="x_t", initial_value=var)
            )
            self.z_t.append(
                add_var(model_variable=var, variable_name="z_t", initial_value=var)
            )
            self.v_t.append(add_var(model_variable=var, variable_name="v_t"))
            if self.amsgrad:
                self.v_hat.append(add_var(model_variable=var, variable_name="v_hat"))
        self._built = True

    def update_step(self, gradient, y):
        """Update variable given gradient"""
        schedule, c = self._compute_schedule_and_c(y)

        beta_1 = tf.cast(self.beta_1, y.dtype)
        beta_2 = tf.cast(self.beta_2, y.dtype)
        gamma = schedule * tf.cast(self.learning_rate, y.dtype)
        lambda_ = tf.cast(self.sf_weight_decay, y.dtype)

        var_index = self._index_dict[self._var_key(y)]
        x_t = self.x_t[var_index]
        z_t = self.z_t[var_index]
        v_t = self.v_t[var_index]

        kp1 = tf.cast(self.iterations + 1, y.dtype)
        bias_correction = tf.math.sqrt(1 - beta_2**kp1)

        if isinstance(gradient, tf.IndexedSlices):
            # # Sparse gradients.
            raise NotImplementedError()
        else:
            # Dense gradients
            v_t.assign(beta_2 * v_t + (1 - beta_2) * gradient**2)
            if self.amsgrad:
                v_hat = self.v_hat[var_index]
                v_t = tf.maximum(v_t, v_hat)
                v_hat.assign(v_t)
            grad_n = gradient * bias_correction / (tf.math.sqrt(v_t) + self.epsilon)
            self._step_dense(
                x=x_t,
                y=y,
                z=z_t,
                gradient=grad_n,
                c=c,
                beta=beta_1,
                gamma=gamma,
                lambda_=lambda_,
            )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(self._learning_rate),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config
