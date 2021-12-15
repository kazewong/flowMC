from typing import Sequence, Callable
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu
    use_bias: bool = True
    init_weight_scale: float = 1e-4
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.layers = [nn.Dense(feat, use_bias=self.use_bias, kernel_init=self.kernel_i(self.init_weight_scale, "fan_in", "normal")) for feat in self.features]

    def __call__(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class AffineCoupling(nn.Module):

    n_features: int
    n_hidden: int
    mask: jnp.array
    dt: float = 1

    def setup(self):
        self.scale_MLP = MLP([self.n_features, self.n_hidden, self.n_features])
        self.translate_MLP = MLP([self.n_features, self.n_hidden, self.n_features])

    def __call__(self, x):
        s = self.mask * self.scale_MLP(x*(1-self.mask))
        s = jnp.tanh(s)
        t = self.mask * self.translate_MLP(x*(1-self.mask))
        s = self.dt * s
        t = self.dt * t
        log_det = s.reshape(s.shape[0], -1).sum(axis=-1)
        outputs = (x + t) * jnp.exp(s)
        return outputs, log_det

    def inverse(self, x):
        s = self.mask * self.scale_MLP(x*(1-self.mask))
        s = jnp.tanh(s)
        t = self.mask * self.translate_MLP(x*(1-self.mask))
        s = self.dt * s
        t = self.dt * t
        log_det = -s.reshape(s.shape[0], -1).sum(axis=-1)
        outputs = x * jnp.exp(-s) - t
        return outputs, log_det



class RealNVP(nn.Module):
    
    n_layer: int
    n_features: int
    n_hidden: int
    dt: float = 1

    def setup(self):
        affine_coupling = []
        for i in range(self.n_layer):
            mask = np.ones(self.n_features)
            mask[int(self.n_features/2):] = 0
            if i % 2 == 0:
                mask = 1 - mask
            mask = jnp.array(mask)
            affine_coupling.append(AffineCoupling(self.n_features, self.n_hidden, mask, dt=self.dt))
        self.affine_coupling = affine_coupling

    def __call__(self, x):
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[i](x)
            log_det += log_det_i
        return x, log_det

    def inverse(self, x):
        log_det = jnp.zeros(x.shape[0])
        for i in range(self.n_layer):
            x, log_det_i = self.affine_coupling[self.n_layer-1-i].inverse(x)
            log_det += log_det_i
        return x, log_det

    def sample(self, rng_key, n_samples, params):
        mean = jnp.zeros((n_samples,self.n_features))
        cov = jnp.repeat(jnp.eye(self.n_features)[None,:],n_samples,axis=0)
        gaussian = jax.random.multivariate_normal(rng_key, mean, cov)
        samples = self.apply({'params': params},gaussian,method=self.inverse)
        return samples