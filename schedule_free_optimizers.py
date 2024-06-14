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
    def _step_dense(*, x, y, z, gradient, c, beta, gamma):
        y.assign((1 - beta) * z + beta * x)
        z.assign_sub(gamma * gradient)
        x.assign((1 - c) * x + c * z)

    def _compute_gamma_and_c(self, gamma, var_index):
        dtype = gamma.dtype
        kp1 = tf.cast(self.iterations + 1, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        n = tf.minimum(kp1, warmup_steps)
        sched = n / warmup_steps

        gamma = gamma * sched

        weight = gamma**self.weight_gamma_exponent
        weight_sum = self.weight_sum[var_index]
        weight_sum.assign_add(weight)
        c = weight / weight_sum

        return gamma, c

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
            {
                'warmup_steps': self.warmup_steps,
            }
        )


class SGDScheduleFree(BaseScheduleFree):
    def __init__(
        self,
        learning_rate=0.1,
        momentum=0.9,
        warmup_steps=0,
        weight_decay=0.004,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="SGDScheduleFree",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
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
        self.momentum = momentum
        self.warmup_steps = warmup_steps  # >= 0

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
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
        """Update step given gradient and the associated model variable."""
        gamma = tf.cast(self.learning_rate, y.dtype)
        beta = tf.cast(self.momentum, y.dtype)
        var_index = self._index_dict[self._var_key(y)]
        z_t = self.z_t[var_index]
        x_t = self.x_t[var_index]

        gamma, c = self._compute_gamma_and_c(gamma, var_index)

        if isinstance(gradient, tf.IndexedSlices):
            # # Sparse gradients.
            raise NotImplementedError()
        else:
            self._step_dense(
                x=x_t, y=y, z=z_t, gradient=gradient, c=c, beta=beta, gamma=gamma
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
    def __init__(
        self,
        learning_rate: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        amsgrad: bool = False,
        warmup_steps: int = 0,
        weight_decay: Optional[float] = None,
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
            weight_decay=weight_decay,
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

    def build(self, var_list):
        """Initialize optimizer variables.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
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
        """Update step given gradient and the associated model variable."""
        beta_1 = tf.cast(self.beta_1, y.dtype)
        beta_2 = tf.cast(self.beta_2, y.dtype)
        var_index = self._index_dict[self._var_key(y)]
        x_t = self.x_t[var_index]
        z_t = self.z_t[var_index]
        v_t = self.v_t[var_index]

        kp1 = tf.cast(self.iterations + 1, y.dtype)
        gamma = tf.cast(self.learning_rate, y.dtype)

        bias_correction = tf.math.sqrt(1 - beta_2**kp1)
        gamma, c = self._compute_gamma_and_c(gamma * bias_correction, var_index)

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
            normed = gradient / (tf.math.sqrt(v_t) + self.epsilon)
            self._step_dense(
                x=x_t, y=y, z=z_t, gradient=normed, c=c, beta=beta_1, gamma=gamma
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