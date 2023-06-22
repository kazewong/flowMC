from typing import List, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from flowMC.nfmodel.base import NFModel
from flowMC.nfmodel.common import MLP, MaskedCouplingLayer, MLPAffine
from jaxtyping import Array

class AffineCoupling(eqx.Module):
    """
    Affine coupling layer. 
    (Defined in the RealNVP paper https://arxiv.org/abs/1605.08803)
    We use tanh as the default activation function.

    Args:
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        mask: (ndarray) Alternating mask for the affine coupling layer.
        dt: (float) Scaling factor for the affine coupling layer.
    """
    _mask: Array
    scale_MLP: eqx.Module
    translate_MLP: eqx.Module
    dt: float = 1

    def __init__(self, n_features: int, n_hidden: int, mask:Array, key: jax.random.PRNGKey, dt: float = 1, scale: float = 1e-4):
        self._mask = mask
        self.dt = dt
        key, scale_subkey, translate_subkey = jax.random.split(key, 3)
        features = [n_features, n_hidden, n_features]
        self.scale_MLP = MLP(features, key=scale_subkey, scale=scale)
        self.translate_MLP = MLP(features, key=translate_subkey, scale=scale)

    @property
    def mask(self):
        return jax.lax.stop_gradient(self._mask)

    @property
    def n_features(self):
        return self.scale_MLP.n_input

    def __call__(self, x: Array):
        return self.forward(x)

    def forward(self, x: Array):
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s) * self.dt
        t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
        log_det = s.sum()
        outputs = (x + t) * jnp.exp(s)
        return outputs, log_det

    def inverse(self, x: Array):
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s) * self.dt
        t = self.mask * self.translate_MLP(x * (1 - self.mask)) * self.dt
        log_det = -s.sum()
        outputs = x * jnp.exp(-s) - t
        return outputs, log_det
    



class RealNVP(NFModel):
    """
    RealNVP mode defined in the paper https://arxiv.org/abs/1605.08803.
    MLP is needed to make sure the scaling between layers are more or less the same.

    Args:
        n_layer: (int) The number of affine coupling layers.
        n_features: (int) The number of features in the input.
        n_hidden: (int) The number of hidden units in the MLP.
        dt: (float) Scaling factor for the affine coupling layer.

    Properties:
        base_mean: (ndarray) Mean of Gaussian base distribution
        base_cov: (ndarray) Covariance of Gaussian base distribution
    """

    affine_coupling: List[MaskedCouplingLayer]
    _base_mean: Array
    _base_cov: Array
    n_features: int

    @property
    def n_layer(self):
        return len(self.affine_coupling)


    @property
    def base_mean(self):
        return jax.lax.stop_gradient(self._base_mean)

    @property
    def base_cov(self):
        return jax.lax.stop_gradient(self._base_cov)


    def __init__(self, n_layer: int, n_features: int, n_hidden: int, key: jax.random.PRNGKey, dt: float = 1, **kwargs):
        self.n_features = n_features
        affine_coupling = []
        for i in range(n_layer):
            key, scale_subkey, shift_subkey = jax.random.split(key, 3)
            mask = np.ones(n_features)
            mask[int(n_features / 2):] = 0
            if i % 2 == 0:
                mask = 1 - mask
            mask = jnp.array(mask)
            scale_MLP = MLP([n_features, n_hidden, n_features], key=scale_subkey)
            shift_MLP = MLP([n_features, n_hidden, n_features], key=shift_subkey)
            affine_coupling.append(
                # AffineCoupling(n_features, n_hidden, mask, scale_subkey, dt=dt)
                MaskedCouplingLayer(MLPAffine(scale_MLP, shift_MLP), mask)
            )
        self.affine_coupling = affine_coupling
        if kwargs.get("base_mean") is not None:
            self._base_mean = kwargs.get("base_mean")
        else:
            self._base_mean = jnp.zeros(n_features)
        if kwargs.get("base_cov") is not None:
            self._base_cov = kwargs.get("base_cov")
        else:
            self._base_cov = jnp.eye(n_features)


    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.forward(x)

    def forward(self, x: Array) -> Tuple[Array, Array]:
        log_det = 0
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[i](x)
            log_det += log_det_i
        return x, log_det

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        x = (x-self.base_mean)/jnp.sqrt(jnp.diag(self.base_cov))
        log_det = 0
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[self.n_layer - 1 - i].inverse(x)
            log_det += log_det_i
        return x, log_det

    inverse_vmap = jax.vmap(inverse, in_axes=(None, 0))

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        gaussian = jax.random.multivariate_normal(
            rng_key, jnp.zeros(self.n_features), jnp.eye(self.n_features), shape=(n_samples,)
        )
        samples = self.inverse_vmap(gaussian)[0]
        samples = samples * jnp.sqrt(jnp.diag(self.base_cov)) + self.base_mean
        return samples # Return only the samples 

    def log_prob(self, x: Array) -> Array:
        x = (x-self.base_mean)/jnp.sqrt(jnp.diag(self.base_cov))
        y, log_det = self.__call__(x)
        log_det = log_det + jax.scipy.stats.multivariate_normal.logpdf(
            y, jnp.zeros(self.n_features), jnp.eye(self.n_features)
        )
        return log_det

