from typing import Callable, List, Iterable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx

from flowMC.nfmodel.base import Bijection

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


class MaskedCouplingLayer(Bijection):

    r"""Masked coupling layer.

    Adopted from distrax masked compuling layer. But now it should be compatible with equinox.
    """

    _mask: Array
    conditioner: eqx.Module
    bijector: Bijection

    @property
    def mask(self):
        return jax.lax.stop_gradient(self._mask)

    def __init__(self, conditioner: eqx.Module, bijector: Bijection, mask: Array):
        self.conditioner = conditioner
        self.bijector = bijector
        self._mask = mask

    def forward(self, x: Array) -> Tuple[Array, Array]:
        y, log_det = self.bijector(self.conditioner(x*self.mask))
        y = (1-self.mask)*y + self.mask*x
        log_det = ((1-self.mask)*log_det).sum()
        return y, log_det

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        y, log_det = self.bijector.inverse(self.conditioner(x*self.mask))
        y = (1-self.mask)*y + self.mask*x
        log_det = ((1-self.mask)*log_det).sum()
        return y, log_det

class AffineTransformation(Bijection):
    scale_MLP: MLP
    shift_MLP: MLP
    dt: float = 1

    def __init__(self, scale_MLP: MLP, shift_MLP: MLP, dt: float = 1):
        self.scale_MLP = scale_MLP
        self.shift_MLP = shift_MLP
        self.dt = dt

    def forward(self, x: Array) -> Tuple[Array, Array]:
        shift = self.shift_MLP(x) * self.dt
        scale = self.scale_MLP(x) * self.dt
        log_det = jnp.sum(scale)
        y = x + shift * jnp.exp(scale)
        return y, log_det

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        shift = self.shift_MLP(x) * self.dt
        scale = self.scale_MLP(x) * self.dt
        log_det = -jnp.sum(scale)
        y = x  * jnp.exp(-scale) - shift
        return y, log_det
