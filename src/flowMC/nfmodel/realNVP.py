import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from flowMC.nfmodel.mlp import MLP
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
    dt: float = 1

    def __init__(self, n_features: int, n_hidden: int, mask:Array, key: jax.random.PRNGKey):
        self.mask = mask
        key, scale_subkey, translate_subkey = jax.random.split(key, 3)
        features = [n_features, n_hidden, n_features]
        self.scale_MLP = MLP(features, key=scale_subkey, activation=jax.nn.tanh)
        self.translate_MLP = MLP(features, key=translate_subkey, activation=jax.nn.tanh)

    def __call__(self, x: Array):
        return self.forward(x)

    def forward(self, x: Array):
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s)
        t = self.mask * self.translate_MLP(x * (1 - self.mask))
        s = self.dt * s
        t = self.dt * t
        log_det = s.reshape(s.shape[0], -1).sum(axis=-1)
        outputs = (x + t) * jnp.exp(s)
        return outputs, log_det

    def inverse(self, x):
        s = self.mask * self.scale_MLP(x * (1 - self.mask))
        s = jnp.tanh(s)
        t = self.mask * self.translate_MLP(x * (1 - self.mask))
        s = self.dt * s
        t = self.dt * t
        log_det = -s.reshape(s.shape[0], -1).sum(axis=-1)
        outputs = x * jnp.exp(-s) - t
        return outputs, log_det


class RealNVP(eqx.Module):
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

    n_layer: int
    n_features: int
    n_hidden: int
    dt: float = 1

    def setup(self):
        affine_coupling = []
        for i in range(self.n_layer):
            mask = np.ones(self.n_features)
            mask[int(self.n_features / 2):] = 0
            if i % 2 == 0:
                mask = 1 - mask
            mask = jnp.array(mask)
            affine_coupling.append(
                AffineCoupling(self.n_features, self.n_hidden, mask, dt=self.dt)
            )
        self.affine_coupling = affine_coupling

        self.base_mean = self.variable(
            "variables", "base_mean", jnp.zeros, ((self.n_features))
        )
        self.base_cov = self.variable(
            "variables", "base_cov", jnp.eye, (self.n_features)
        )

    def __call__(self, x):
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[i](x)
            log_det += log_det_i
        return x, log_det

    def inverse(self, x):
        x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[self.n_layer - 1 - i].inverse(x)
            log_det += log_det_i
        return x, log_det

    def sample(self, rng_key, n_samples):
        gaussian = jax.random.multivariate_normal(
            rng_key, jnp.zeros(self.n_features), jnp.eye(self.n_features), shape=(n_samples,)
        )
        samples = self.inverse(gaussian)[0]
        samples = samples * jnp.sqrt(jnp.diag(self.base_cov.value)) + self.base_mean.value
        return samples # Return only the samples 

    def log_prob(self, x):
        x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
        y, log_det = self.__call__(x)
        log_det = log_det + jax.scipy.stats.multivariate_normal.logpdf(
            y, jnp.zeros(self.n_features), jnp.eye(self.n_features)
        )
        return log_det
