from typing import Sequence, Callable, List
import jax
from flax import linen as nn
import equinox as eqx



class MLP(nn.Module):
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
    
class MLP(eqx.Module):
    r"""Multilayer perceptron.

    Args:
        shape (Iterable[int]): Shape of the MLP. The first element is the input dimension, the last element is the output dimension.
        key (jax.random.PRNGKey): Random key.

    Attributes:
        layers (List): List of layers.
        activation (Callable): Activation function.
        use_bias (bool): Whether to use bias.        
    """
    layers: List
    activation: Callable = jax.nn.relu
    use_bias: bool = True

    def __init__(self, shape, key):
        self.layers = []
        for i in range(len(shape) - 2):
            key, subkey = jax.random.split(key)
            self.layers.append(eqx.nn.Linear(shape[i], shape[i + 1], key=subkey, use_bias=self.use_bias))
            self.layers.append(self.activation)
        key, subkey = jax.random.split(key)
        self.layers.append(eqx.nn.Linear(shape[-2], shape[-1], key=subkey, use_bias=self.use_bias))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x