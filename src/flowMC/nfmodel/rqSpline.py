from typing import Sequence, Callable, List
import numpy as np
import jax
import jax.numpy as jnp
import distrax

from flax import linen as nn           # The Linen API


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.tanh
    use_bias: bool = True
    init_weight_scale: float = 1e-2
    kernel_i: Callable = jax.nn.initializers.variance_scaling

    def setup(self):
        self.layers = [nn.Dense(feat, use_bias=self.use_bias, kernel_init=self.kernel_i(
            self.init_weight_scale, "fan_in", "normal")) for feat in self.features]

    def __call__(self, x):
        for l, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class Reshape(nn.Module):
    shape: Sequence[int]
    def __call__(self, x):
        return jnp.reshape(x.T, self.shape)


class Conditioner(nn.Module):
    input_size: int
    hidden_size: Sequence[int]
    num_bijector_params: int

    def setup(self):
        self.conditioner = nn.Sequential([MLP([self.input_size]+list(self.hidden_size)), nn.tanh, nn.Dense(
            self.input_size * self.num_bijector_params, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros), Reshape((self.input_size, self.num_bijector_params))])

    def __call__(self, x):
        return self.conditioner(x)

class Scalar(nn.Module):
    input_size: int

    def setup(self):
        self.shift = self.param('shifts',lambda rng, shape: jnp.zeros(shape), (self.input_size))
        self.scale = self.param('scales',lambda rng, shape: jnp.ones(shape), (self.input_size))

    def __call__(self, x):
        return self.scale, self.shift

def bijector_fn(params: jnp.ndarray):
    return distrax.RationalQuadraticSpline(params, range_min=-3., range_max=3.)

def scalar_affine(params: jnp.ndarray):
    return distrax.ScalarAffine(scale=params[0],shift=params[1])


class RQSpline(nn.Module):
    input_size: int
    num_layers: int
    hidden_size: Sequence[int]
    num_bins: int

    def setup(self):
        conditioner = []
        scalar = []
        for i in range(self.num_layers):
            conditioner.append(Conditioner(self.input_size, self.hidden_size, 3*self.num_bins+1))
            scalar.append(Scalar(self.input_size))

        self.conditioner = conditioner
        self.scalar = scalar

    def make_flow(self):
        mask = (jnp.arange(0, self.input_size) % 2).astype(bool)
        mask_all = (jnp.zeros(self.input_size)).astype(bool)
        layers = []
        for i in range(self.num_layers):
            layers.append(distrax.MaskedCoupling(
                mask=mask_all, bijector=scalar_affine, conditioner=self.scalar[i]))
            layers.append(distrax.MaskedCoupling(
                mask=mask, bijector=bijector_fn, conditioner=self.conditioner[i]))
            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(distrax.Chain(layers))
        base_dist = distrax.Independent(distrax.MultivariateNormalDiag(
            loc=jnp.zeros(self.input_size), scale_diag=jnp.ones(self.input_size)))

        return base_dist, flow

    def __call__(self, x):
        base_dist, flow = self.make_flow()
        return distrax.Transformed(base_dist, flow).log_prob(x)

    def sample(self, rng, num_samples):
        base_dist, flow = self.make_flow()
        return distrax.Transformed(base_dist, flow).sample(seed=rng, sample_shape=(num_samples))

    def log_prob(self,x):
        return self.__call__(x)