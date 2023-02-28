from typing import Sequence, Callable
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


class MLP(nn.Module):
    """
    Multi-layer perceptron in Flax. We use a gaussian kernel with a standard deviation
    of `init_weight_scale=1e-4` by default.

    Args:
        features: (list of int) The number of features in each layer.
        activation: (callable) The activation function at each level
        use_bias: (bool) Whether to use bias in the layers.
        init_weight_scale: (float) The initial weight scale for the layers.
        kernel_init: (callable) The kernel initializer for the layers.
    """

    features: Sequence[int]
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-4
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.layers = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal"),
            )
            for feat in self.features
        ]

    def __call__(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x