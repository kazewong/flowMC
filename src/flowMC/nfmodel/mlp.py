from typing import Callable, List, Iterable
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array
    
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

    def __init__(self, shape: Iterable[int], key: jax.random.PRNGKey, scale: float = 1e-4, activation: Callable = jax.nn.relu, use_bias: bool = True):
        self.layers = []
        for i in range(len(shape) - 2):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            layer = eqx.nn.Linear(shape[i], shape[i + 1], key=subkey1, use_bias=use_bias)
            weight = jax.random.normal(subkey2, (shape[i + 1], shape[i]))*jnp.sqrt(scale/shape[i])
            layer = eqx.tree_at(lambda l: l.weight, layer, weight)
            self.layers.append(layer)
            self.layers.append(activation)
        key, subkey = jax.random.split(key)
        self.layers.append(eqx.nn.Linear(shape[-2], shape[-1], key=subkey, use_bias=use_bias))

    def __call__(self, x: Array):
        for layer in self.layers:
            x = layer(x)
        return x
    
    @property
    def n_input(self) -> int:
        return self.layers[0].in_features
    
    @property
    def n_output(self) -> int:
        return self.layers[-1].out_features