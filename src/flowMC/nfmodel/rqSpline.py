from typing import Sequence, Callable, List
import numpy as np
import jax
import jax.numpy as jnp
import distrax

from flax import linen as nn  # The Linen API


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.tanh
    use_bias: bool = True
    init_weight_scale: float = 1e-2
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
                MLP([self.n_features] + list(self.hidden_size)),
                nn.tanh,
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

    Parameters
    ----------
    n_features : int
        Number of features in the data.
    num_layers : int
        Number of layers in the flow.
    num_bins : int
        Number of bins in the spline.
    hidden_size : Sequence[int]
        Size of the hidden layers in the conditioner.
    spline_range : Sequence[float]
        Range of the spline.
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
        base_dist, flow = self.make_flow()
        samples = distrax.Transformed(base_dist, flow).sample(
            seed=rng, sample_shape=(num_samples)
        )
        return samples * jnp.sqrt(jnp.diag(self.base_cov.value)) + self.base_mean.value

    def log_prob(self, x: jnp.array) -> jnp.array:
        return self.vmap_call(x)
