from typing import Callable, List, Iterable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx

from flowMC.nfmodel.base import Bijection, Distribution

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

    f(x) = (1-m)*b(x;c(m*x;z)) + m*x
    where b is the inner bijector, m is the mask, and c is the conditioner.

    Args:
        bijector (Bijection): inner bijector in the masked coupling layer.
        mask (Array): Mask. 0 for the input variables that are transformed, 1 for the input variables that are not transformed.

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

class Gaussian(Distribution):

    r"""Multivariate Gaussian distribution.
    
    Args:
        mean (Array): Mean.
        cov (Array): Covariance matrix.
        learnable (bool): Whether the mean and covariance matrix are learnable parameters.

    Attributes:
        mean (Array): Mean.
        cov (Array): Covariance matrix.
    """

    _mean: Array
    _cov: Array
    learnable: bool = False

    @property
    def mean(self) -> Array:
        if self.learnable:
            return self._mean
        else:
            return jax.lax.stop_gradient(self._mean)

    @property
    def cov(self) -> Array:
        if self.learnable:
            return self._cov
        else:
            return jax.lax.stop_gradient(self._cov)

    def __init__(self, mean: Array, cov: Array, learnable: bool = False):
        self._mean = mean
        self._cov = cov
        self.learnable = learnable

    def log_prob(self, x: Array) -> Array:
        return jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov)

    def sample(self, key: jax.random.PRNGKey, n_samples: int = 1) -> Array:
        return jax.random.multivariate_normal(key, self.mean, self.cov, (n_samples,))
    
class Composable(Distribution):

    distributions: list[Distribution]
    partitions: dict[str, tuple[int, int]]

    def __init__(self, distributions: list[Distribution], partitions: dict):
        self.distributions = distributions
        self.partitions = partitions

    def log_prob(self, x: Array) -> Array:
        log_prob = 0
        for dist, (_, ranges) in zip(self.distributions, self.partitions.items()):
            log_prob += dist.log_prob(x[ranges[0]:ranges[1]])
        return log_prob
    
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        samples = {}
        for dist, (key,_) in zip(self.distributions, self.partitions.items()):
            rng_key, sub_key = jax.random.split(rng_key)
            samples[key] = dist.sample(sub_key, n_samples=n_samples)