from typing import Sequence, Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array
import distrax
import equinox as eqx

from flax import linen as nn
from flowMC.nfmodel.base import NFModel  # The Linen API
from flowMC.nfmodel.mlp import MLP


class Reshape(nn.Module):
    shape: Sequence[int]

    def __call__(self, x):
        return jnp.reshape(x.T, self.shape)


class Conditioner(nn.Module):
    n_features: int
    hidden_size: Sequence[int]
    num_bijector_params: int

    def setup(self):
        self.conditioner = nn.Sequential(
            [
                MLP([self.n_features] + list(self.hidden_size),
                nn.tanh,
                init_weight_scale=1e-2),
                nn.Dense(
                    self.n_features * self.num_bijector_params,
                    kernel_init=jax.nn.initializers.zeros,
                    bias_init=jax.nn.initializers.zeros,
                ),
                Reshape((self.n_features, self.num_bijector_params)),
            ]
        )

    def __call__(self, x):
        return self.conditioner(x)


class Scalar(nn.Module):
    n_features: int

    def setup(self):
        self.shift = self.param(
            "shifts", lambda rng, shape: jnp.zeros(shape), (self.n_features)
        )
        self.scale = self.param(
            "scales", lambda rng, shape: jnp.ones(shape), (self.n_features)
        )

    def __call__(self, x):
        return self.scale, self.shift


def scalar_affine(params: jnp.ndarray):
    return distrax.ScalarAffine(scale=params[0], shift=params[1])


class RQSpline(nn.Module):
    """
    Rational quadratic spline normalizing flow model using distrax.

    Args:
        n_features : (int) Number of features in the data.
        num_layers : (int) Number of layers in the flow.
        num_bins : (int) Number of bins in the spline.
        hidden_size : (Sequence[int]) Size of the hidden layers in the conditioner.
        spline_range : (Sequence[float]) Range of the spline.
    
    Properties:
        base_mean: (ndarray) Mean of Gaussian base distribution
        base_cov: (ndarray) Covariance of Gaussian base distribution
    """

    n_features: int
    num_layers: int
    hidden_size: Sequence[int]
    num_bins: int
    spline_range: Sequence[float] = (-10.0, 10.0)

    def setup(self):
        conditioner = []
        scalar = []
        for i in range(self.num_layers):
            conditioner.append(
                Conditioner(self.n_features, self.hidden_size, 3 * self.num_bins + 1)
            )
            scalar.append(Scalar(self.n_features))

        self.conditioner = conditioner
        self.scalar = scalar

        self.base_mean = self.variable(
            "variables", "base_mean", jnp.zeros, ((self.n_features))
        )
        self.base_cov = self.variable(
            "variables", "base_cov", jnp.eye, (self.n_features)
        )

        self.vmap_call = jax.jit(jax.vmap(self.__call__))

        def bijector_fn(params: jnp.ndarray):
            return distrax.RationalQuadraticSpline(
                params, range_min=self.spline_range[0], range_max=self.spline_range[1]
            )

        self.bijector_fn = bijector_fn

    def make_flow(self):
        mask = (jnp.arange(0, self.n_features) % 2).astype(bool)
        mask_all = (jnp.zeros(self.n_features)).astype(bool)
        layers = []
        for i in range(self.num_layers):
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask_all, bijector=scalar_affine, conditioner=self.scalar[i]
                )
            )
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=self.bijector_fn,
                    conditioner=self.conditioner[i],
                )
            )
            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(distrax.Chain(layers))
        base_dist = distrax.Independent(
            distrax.MultivariateNormalFullCovariance(
                loc=jnp.zeros(self.n_features),
                covariance_matrix=jnp.eye(self.n_features),
            )
        )

        return base_dist, flow

    def __call__(self, x: jnp.array) -> jnp.array:
        x = (x-self.base_mean.value)/jnp.sqrt(jnp.diag(self.base_cov.value))
        base_dist, flow = self.make_flow()
        return distrax.Transformed(base_dist, flow).log_prob(x)

    def sample(self, rng: jax.random.PRNGKey, num_samples: int) -> jnp.array:
        """"
        Sample from the flow.
        """
        base_dist, flow = self.make_flow()
        samples = distrax.Transformed(base_dist, flow).sample(
            seed=rng, sample_shape=(num_samples)
        )
        return samples * jnp.sqrt(jnp.diag(self.base_cov.value)) + self.base_mean.value

    def log_prob(self, x: jnp.array) -> jnp.array:
        return self.vmap_call(x)

class Reshape(eqx.Module):
    shape: Sequence[int]

    def __call__(self, x):
        return jnp.reshape(x.T, self.shape)


class Conditioner(eqx.Module):
    conditioner: list

    def __init__(self, n_features: int, hidden_size: Sequence[int], num_bijector_params: int, key: jax.random.PRNGKey):
        key, mlp_key, linear_key = jax.random.split(key, 3)
        mlp = MLP([n_features] + list(hidden_size), key=mlp_key, scale=1e-2, activation=jax.nn.tanh)
        linear_layer = eqx.nn.Linear(hidden_size[-1], n_features*num_bijector_params, key=linear_key, use_bias=True)
        weight = jnp.zeros((n_features*num_bijector_params, hidden_size[-1]))
        bias = jnp.zeros(n_features*num_bijector_params)
        linear = eqx.tree_at(lambda l: l.weight, linear_layer, weight)
        linear = eqx.tree_at(lambda l: l.bias, linear, bias)

        self.conditioner = [mlp, linear,
                Reshape((n_features, num_bijector_params)),
            ]

    def __call__(self, x):
        for layer in self.conditioner:
            x = layer(x)
        return x


class Scalar(eqx.Module):
    shifts: Array
    scales: Array

    def __init__(self, n_features: int):
        self.shifts = jnp.zeros(n_features)
        self.scales = jnp.ones(n_features)

    def __call__(self, x):
        return self.scales, self.shifts


def scalar_affine(params: jnp.ndarray):
    return distrax.ScalarAffine(scale=params[0], shift=params[1])

class RQSpline(NFModel):
    r""" Rational quadratic spline normalizing flow model using distrax.

    Args:
        n_features : (int) Number of features in the data.
        num_layers : (int) Number of layers in the flow.
        num_bins : (int) Number of bins in the spline.
        hidden_size : (Sequence[int]) Size of the hidden layers in the conditioner.
        spline_range : (Sequence[float]) Range of the spline.
    
    Properties:
        base_mean: (ndarray) Mean of Gaussian base distribution
        base_cov: (ndarray) Covariance of Gaussian base distribution
    """

    _base_mean: Array
    _base_cov: Array
    flow: distrax.Transformed

    @property
    def base_mean(self):
        return jax.lax.stop_gradient(self._base_mean)

    @property
    def base_cov(self):
        return jax.lax.stop_gradient(self._base_cov)

    def __init__(self,
                n_features: int,
                num_layers: int,
                hidden_size: Sequence[int],
                num_bins: int,
                key: jax.random.PRNGKey,
                spline_range: Sequence[float] = (-10.0, 10.0), **kwargs):

        if kwargs.get("base_mean") is not None:
            self._base_mean = kwargs.get("base_mean")
        else:
            self._base_mean = jnp.zeros(n_features)
        if kwargs.get("base_cov") is not None:
            self._base_cov = kwargs.get("base_cov")
        else:
            self._base_cov = jnp.eye(n_features)

        conditioner = []
        scalar = []
        for i in range(num_layers):
            key, conditioner_key= jax.random.split(key)
            conditioner.append(
                Conditioner(n_features, hidden_size, 3 * num_bins + 1, key=conditioner_key)
            )
            scalar.append(Scalar(n_features))

        bijector_fn = lambda x: distrax.RationalQuadraticSpline(
            x, range_min=spline_range[0], range_max=spline_range[1]
        )
        mask = (jnp.arange(0, n_features) % 2).astype(bool)
        mask_all = (jnp.zeros(n_features)).astype(bool)
        layers = []
        for i in range(num_layers):
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask_all, bijector=scalar_affine, conditioner=scalar[i]
                )
            )
            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,
                    bijector=bijector_fn,
                    conditioner=conditioner[i],
                )
            )
            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(distrax.Chain(layers))
        base_dist = distrax.Independent(
            distrax.MultivariateNormalFullCovariance(
                loc=jnp.zeros(n_features),
                covariance_matrix=jnp.eye(n_features),
            )
        )

        self.flow = distrax.Transformed(base_dist, flow)

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        x = (x-self.base_mean)/jnp.sqrt(jnp.diag(self.base_cov))
        return self.flow.log_prob(x)

    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        return self.flow.sample(seed=rng_key, sample_shape=(n_samples,))

    def inverse(self, x: Array) -> Tuple[Array, Array]:
        return super().inverse(x)

    inverse_vmap = jax.vmap(inverse, in_axes=(None, 0))

    def log_prob(self, x: Array) -> Array:
        return self.__call__(x)