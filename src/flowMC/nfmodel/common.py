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

    @property
    def dtype(self) -> jnp.dtype:
        return self.layers[0].weight.dtype


class MaskedCouplingLayer(Bijection):

    r"""Masked coupling layer.

    Adopted from distrax masked compuling layer. But now it should be compatible with equinox.
    """

    _mask: Array
    bijector: Bijection

    @property
    def mask(self):
        return jax.lax.stop_gradient(self._mask)

    def __init__(self, bijector: Bijection, mask: Array):
        self.bijector = bijector
        self._mask = mask

    def forward(self, x: Array) -> Tuple[Array, Array]:
        y, log_det = self.bijector(x, x*self.mask)
        y = (1-self.mask)*y + self.mask*x
        log_det = ((1-self.mask)*log_det).sum()
        return y, log_det

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        y, log_det = self.bijector.inverse(x, x*self.mask)
        y = (1-self.mask)*y + self.mask*x
        log_det = ((1-self.mask)*log_det).sum()
        return y, log_det

class MLPAffine(Bijection):
    scale_MLP: MLP
    shift_MLP: MLP
    dt: float = 1

    def __init__(self, scale_MLP: MLP, shift_MLP: MLP, dt: float = 1):
        self.scale_MLP = scale_MLP
        self.shift_MLP = shift_MLP
        self.dt = dt

    def __call__(self, x: Array, condition_x: Array) -> Tuple[Array, Array]:
        return self.forward(x, condition_x)

    def forward(self, x: Array, condition_x: Array) -> Tuple[Array, Array]:
        # Note that this note output log_det as an array instead of a number.
        # This is because we need to sum over the log_det in the masked coupling layer.
        scale = jnp.tanh(self.scale_MLP(condition_x)) * self.dt
        shift = self.shift_MLP(condition_x) * self.dt
        log_det = scale
        y = (x + shift) * jnp.exp(scale)
        return y, log_det

    def inverse(self, x: Array, condition_x: Array) -> Tuple[Array, Array]:
        scale = jnp.tanh(self.scale_MLP(condition_x)) * self.dt
        shift = self.shift_MLP(condition_x) * self.dt
        log_det = -scale
        y = x  * jnp.exp(-scale) - shift
        return y, log_det

class ScalarAffine(Bijection):
    scale: Array
    shift: Array

    def __init__(self, scale: float, shift: float):
        self.scale = jnp.array(scale)
        self.shift = jnp.array(shift)

    def __call__(self, x: Array, condition_x: Array) -> Tuple[Array, Array]:
        return self.forward(x, condition_x)

    def forward(self, x: Array, condition_x: Array) -> Tuple[Array, Array]:
        y = (x + self.shift) * jnp.exp(self.scale)
        log_det = self.scale
        return y, log_det

    def inverse(self, x: Array, condition_x: Array) -> Tuple[Array, Array]:
        y = x * jnp.exp(-self.scale) - self.shift
        log_det = -self.scale
        return y, log_det
