from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from flowMC.resource.nf_model.base import Bijection, Distribution, NFModel
from flowMC.resource.nf_model.common import (
    MLP,
    Gaussian,
    MaskedCouplingLayer,
    ScalarAffine,
)


@partial(jax.vmap, in_axes=(0, None, None))
def _normalize_bin_sizes(
    unnormalized_bin_sizes: Array, total_size: float, min_bin_size: float
) -> Array:
    """Make bin sizes sum to `total_size` and be no less than `min_bin_size`."""
    num_bins = unnormalized_bin_sizes.shape[-1]
    bin_sizes = jax.nn.softmax(unnormalized_bin_sizes, axis=-1)
    return bin_sizes * (total_size - num_bins * min_bin_size) + min_bin_size


@partial(jax.vmap, in_axes=(0, None))
def _normalize_knot_slopes(
    unnormalized_knot_slopes: Array, min_knot_slope: Float
) -> Array:
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    min_knot_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope


@partial(jax.vmap, in_axes=(0, 0, 0, 0))
def _rational_quadratic_spline_fwd(
    x: Array, x_pos: Array, y_pos: Array, knot_slopes: Array
) -> tuple[Array, Array]:
    """Applies a rational-quadratic spline to a scalar.

    Args:
    x: a scalar (0-dimensional array). The scalar `x` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[x_pos[0], x_pos[-1]]`, and it will be transformed linearly if it's
        outside that interval.
    x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
    y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
    knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
    Returns:
    A tuple of two scalars: the output of the transformation and the log of the
    absolute first derivative at `x`.
    """
    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)
    # If y does not fall into any bin, we use the first spline in the following
    # computations to avoid numerical issues.
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    # Dot product of each parameter with the correct bin mask.
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width

    z = (x - x_pos_bin[0]) / bin_width
    # `z` should be in range [0, 1] to avoid NaNs later. This can happen because
    # of small floating point issues or when x is outside of the range of bins.
    # To avoid all problems, we restrict z in [0, 1].
    z = jnp.clip(z, 0.0, 1.0)
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    numerator = bin_height * (bin_slope * sq_z + knot_slopes_bin[0] * z1mz)
    denominator = bin_slope + slopes_term * z1mz
    y = y_pos_bin[0] + numerator / denominator

    # Compute log det Jacobian.
    # The logdet is a sum of 3 logs. It is easy to see that the inputs of the
    # first two logs are guaranteed to be positive because we ensured that z is in
    # [0, 1]. This is also true of the log(denominator) because:
    # denominator
    # == bin_slope + (knot_slopes_bin[1] + knot_slopes_bin[0] - 2 * bin_slope) *
    # z*(1-z)
    # >= bin_slope - 2 * bin_slope * z * (1-z)
    # >= bin_slope - 2 * bin_slope * (1/4)
    # == bin_slope / 2
    logdet = (
        2.0 * jnp.log(bin_slope)
        + jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        - 2.0 * jnp.log(denominator)
    )

    # If x is outside the spline range, we default to a linear transformation.
    y = jnp.where(below_range, (x - x_pos[0]) * knot_slopes[0] + y_pos[0], y)
    y = jnp.where(
        above_range,
        (x - x_pos[-1]) * knot_slopes[-1] + y_pos[-1],
        y,  # type: ignore
    )
    logdet = jnp.where(below_range, jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, jnp.log(knot_slopes[-1]), logdet)
    return y, logdet


def _safe_quadratic_root(a: Array, b: Array, c: Array) -> Array:
    """Implement a numerically stable version of the quadratic formula."""
    # This is not a general solution to the quadratic equation, as it assumes
    # b ** 2 - 4. * a * c is known a priori to be positive (and which of the two
    # roots is to be used, see https://arxiv.org/abs/1906.04032).
    # There are two sources of instability:
    # (a) When b ** 2 - 4. * a * c -> 0, sqrt gives NaNs in gradient.
    # We clip sqrt_diff to have the smallest float number.
    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))
    # If sqrt_diff is non-positive, we set sqrt to 0. as it should be positive.
    safe_sqrt = jnp.where(sqrt_diff > 0.0, safe_sqrt, 0.0)
    # (b) When 4. * a * c -> 0. We use the more stable quadratic solution
    # depending on the sign of b.
    # See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf (eq 7 and 8).
    # Solution when b >= 0
    numerator_1 = 2.0 * c
    denominator_1 = -b - safe_sqrt
    # Solution when b < 0
    numerator_2 = -b + safe_sqrt
    denominator_2 = 2 * a
    # Choose the numerically stable solution.
    numerator = jnp.where(b >= 0, numerator_1, numerator_2)
    denominator = jnp.where(b >= 0, denominator_1, denominator_2)
    return numerator / denominator


@partial(jax.vmap, in_axes=(0, 0, 0, 0))
def _rational_quadratic_spline_inv(
    y: Array, x_pos: Array, y_pos: Array, knot_slopes: Array
) -> tuple[Array, Array]:
    """Applies the inverse of a rational-quadratic spline to a scalar.

    Args:
    y: a scalar (0-dimensional array). The scalar `y` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[y_pos[0], y_pos[-1]]`, and it will be transformed linearly if it's
        outside that interval.
    x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
    y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
    knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
    Returns:
    A tuple of two scalars: the output of the inverse transformation and the log
    of the absolute first derivative of the inverse at `y`.
    """
    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)
    # If y does not fall into any bin, we use the first spline in the following
    # computations to avoid numerical issues.
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    # Dot product of each parameter with the correct bin mask.
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    # These are the parameters for the corresponding bin.
    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width
    w = (y - y_pos_bin[0]) / bin_height
    w = jnp.clip(w, 0.0, 1.0)  # Ensure w is in [0, 1].
    # Compute quadratic coefficients: az^2 + bz + c = 0
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    c = -bin_slope * w
    b = knot_slopes_bin[0] - slopes_term * w
    a = bin_slope - b

    # Solve quadratic to obtain z and then x.
    z = _safe_quadratic_root(a, b, c)
    z = jnp.clip(z, 0.0, 1.0)  # Ensure z is in [0, 1].
    x = bin_width * z + x_pos_bin[0]

    # Compute log det Jacobian.
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet = (
        -2.0 * jnp.log(bin_slope)
        - jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        + 2.0 * jnp.log(denominator)
    )

    # If y is outside the spline range, we default to a linear transformation.
    x = jnp.where(below_range, (y - y_pos[0]) / knot_slopes[0] + x_pos[0], x)
    x = jnp.where(
        above_range,
        (y - y_pos[-1]) / knot_slopes[-1] + x_pos[-1],
        x,  # type: ignore
    )
    logdet = jnp.where(below_range, -jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, -jnp.log(knot_slopes[-1]), logdet)
    return x, logdet


class RQSpline(Bijection):
    _range_min: float
    _range_max: float
    _num_bins: int
    _min_bin_size: float
    _min_knot_slope: float
    conditioner: MLP
    """A rational-quadratic spline bijection.

    This bijection is a piecewise rational-quadratic spline with `num_bins` bins.
    The spline is defined by the bin boundaries on the x and y axes, the slopes
    at the knot points, and the slopes at the boundaries of the spline range.
    The spline is linear outside the spline range.

    Args:
        conditioner (eqx.Module): A conditioner that takes the input and returns
            the parameters of the spline.
        range_min (float): The minimum value of the spline range.
        range_max (float): The maximum value of the spline range.
        num_bins (int): The number of bins in the spline.
        min_bin_size (float): The minimum size of a bin.
        min_knot_slope (float): The minimum slope of the spline at the knot points.

    Attributes:
        range_min (float): The minimum value of the spline range.
        range_max (float): The maximum value of the spline range.
    """

    @property
    def range_min(self):
        return jax.lax.stop_gradient(self._range_min)

    @property
    def range_max(self):
        return jax.lax.stop_gradient(self._range_max)

    @property
    def num_bins(self):
        return jax.lax.stop_gradient(self._num_bins)

    @property
    def min_bin_size(self):
        return jax.lax.stop_gradient(self._min_bin_size)

    @property
    def min_knot_slope(self):
        return jax.lax.stop_gradient(self._min_knot_slope)

    @property
    def dtype(self):
        return self.conditioner.dtype

    def __init__(
        self,
        conditioner: MLP,
        range_min: float,
        range_max: float,
        min_bin_size: float = 1e-4,
        min_knot_slope: float = 1e-4,
    ):
        self._range_min = range_min
        self._range_max = range_max
        self._min_bin_size = min_bin_size
        self._min_knot_slope = min_knot_slope
        self._num_bins = int(conditioner.n_output / conditioner.n_input - 1) // 3

        self.conditioner = conditioner

    def get_params(
        self, x: Float[Array, " n_condition"]
    ) -> tuple[
        Float[Array, " n_param"], Float[Array, " n_param"], Float[Array, " n_param"]
    ]:
        params = self.conditioner(x).reshape(-1, self._num_bins * 3 + 1)
        unnormalized_bin_widths = params[:, : self._num_bins]
        unnormalized_bin_heights = params[:, self._num_bins : 2 * self._num_bins]
        unnormalized_knot_slopes = params[:, 2 * self._num_bins :]
        # Normalize bin sizes and compute bin positions on the x and y axis.
        range_size = self.range_max - self.range_min
        bin_widths = _normalize_bin_sizes(
            unnormalized_bin_widths, range_size, self.min_bin_size
        )
        bin_heights = _normalize_bin_sizes(
            unnormalized_bin_heights, range_size, self.min_bin_size
        )
        x_pos = self.range_min + jnp.cumsum(bin_widths[..., :-1], axis=-1)
        y_pos = self.range_min + jnp.cumsum(bin_heights[..., :-1], axis=-1)
        pad_shape = params.shape[:-1] + (1,)
        pad_below = jnp.full(pad_shape, self.range_min, dtype=self.dtype)
        pad_above = jnp.full(pad_shape, self.range_max, dtype=self.dtype)
        x_pos = jnp.concatenate([pad_below, x_pos, pad_above], axis=-1)
        y_pos = jnp.concatenate([pad_below, y_pos, pad_above], axis=-1)
        # Normalize knot slopes and enforce requested boundary conditions.
        knot_slopes = _normalize_knot_slopes(
            unnormalized_knot_slopes, self.min_knot_slope
        )
        return x_pos, y_pos, knot_slopes

    def __call__(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float]:
        return self.forward(x, condition)

    def forward(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float]:
        x_pos, y_pos, knot_slopes = self.get_params(condition)
        return _rational_quadratic_spline_fwd(x, x_pos, y_pos, knot_slopes)

    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Float[Array, " n_condition"],
    ) -> tuple[Float[Array, " n_dim"], Float]:
        x_pos, y_pos, knot_slopes = self.get_params(condition)
        return _rational_quadratic_spline_inv(x, x_pos, y_pos, knot_slopes)


class MaskedCouplingRQSpline(NFModel):
    r"""Rational quadratic spline normalizing flow model using distrax.

    Args:
        n_features (int):  Number of features in the data.
        num_layers (int): Number of layers in the conditioner.
        hidden_size (Sequence[int]): Hidden size of the conditioner.
        num_bins (int): Number of bins in the spline.
        key (PRNGKeyArray): Random key for initialization.
        spline_range (Sequence[float]): Range of the spline. Defaults to (-10.0, 10.0).

    Properties:
        n_features (int) :  Number of features in the data.
        data_mean (Array) : Mean of the data.
        data_cov (Array) : Covariance of the data.
    """

    base_dist: Distribution
    layers: list[Bijection]

    def __repr__(self):
        return (
            "MaskedCouplingRQSpline with n_features="
            + str(self._n_features)
            + ", n_layers="
            + str(len(self.layers))
        )

    def __init__(
        self,
        n_features: int,
        n_layers: int,
        hidden_size: list[int],
        num_bins: int,
        key: PRNGKeyArray,
        spline_range: tuple[float, float] = (-10.0, 10.0),
        **kwargs,
    ):
        if kwargs.get("base_dist") is not None:
            dist = kwargs.get("base_dist")
            assert isinstance(dist, Distribution)
            self.base_dist = dist
        else:
            self.base_dist = Gaussian(
                jnp.zeros(n_features), jnp.eye(n_features), learnable=False
            )

        if kwargs.get("data_mean") is not None:
            data_mean = kwargs.get("data_mean")
            assert isinstance(data_mean, Array)
            self._data_mean = data_mean
        else:
            self._data_mean = jnp.zeros(n_features)

        if kwargs.get("data_cov") is not None:
            data_cov = kwargs.get("data_cov")
            assert isinstance(data_cov, Array)
            self._data_cov = data_cov
        else:
            self._data_cov = jnp.eye(n_features)

        self._n_features = n_features

        def make_layer(i: int, key: PRNGKeyArray):
            mlp = MLP(
                [n_features] + hidden_size + [n_features * (num_bins * 3 + 1)],
                key,
                scale=1e-2,
                activation=jax.nn.tanh,
            )
            mask = ((jnp.arange(0, n_features) + i) % 2).astype(bool)
            mask_all = (jnp.zeros(n_features)).astype(bool)
            layer1 = MaskedCouplingLayer(ScalarAffine(0.0, 0.0), mask_all)
            layer2 = MaskedCouplingLayer(
                RQSpline(mlp, spline_range[0], spline_range[1]), mask
            )
            return eqx.nn.Sequential([layer1, layer2])  # type: ignore

        keys = jax.random.split(key, n_layers)
        self.layers = eqx.filter_vmap(make_layer)(jnp.arange(n_layers), keys)

    def __call__(
        self, x: Float[Array, " n_dim"]
    ) -> tuple[Float[Array, " n_dim"], Float]:
        return self.forward(x)

    def forward(
        self,
        x: Float[Array, " n_dim"],
        key: Optional[PRNGKeyArray] = None,
        condition: Optional[Float[Array, " n_condition"]] = None,
    ) -> tuple[Float[Array, " n_dim"], Float]:
        log_det = 0.0
        dynamics, statics = eqx.partition(self.layers, eqx.is_array)

        def f(carry, data):
            x, log_det = carry
            layers = eqx.combine(data, statics)
            x, log_det_i = layers[0](x, condition)
            log_det += log_det_i
            x, log_det_i = layers[1](x, condition)
            return (x, log_det + log_det_i), None

        (x, log_det), _ = jax.lax.scan(f, (x, log_det), dynamics)
        return x, log_det

    def inverse(
        self,
        x: Float[Array, " n_dim"],
        condition: Optional[Float[Array, " n_condition"]] = None,
    ) -> tuple[Float[Array, " n_dim"], Float]:
        """From latent space to data space."""
        log_det = 0.0
        dynamics, statics = eqx.partition(self.layers, eqx.is_array)

        def f(carry, data):
            x, log_det = carry
            layers = eqx.combine(data, statics)
            x, log_det_i = layers[0].inverse(x, condition)
            log_det += log_det_i
            x, log_det_i = layers[1].inverse(x, condition)
            return (x, log_det + log_det_i), None

        (x, log_det), _ = jax.lax.scan(f, (x, log_det), dynamics, reverse=True)
        return x, log_det

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> Float[Array, "n_samples n_dim"]:
        samples = self.base_dist.sample(rng_key, n_samples)
        samples = jax.vmap(self.inverse)(samples)[0]
        samples = samples * jnp.sqrt(jnp.diag(self.data_cov)) + self.data_mean
        return samples

    def log_prob(self, x: Float[Array, "n_sample n_dim"]) -> Float[Array, " n_sample"]:
        """From data space to latent space."""
        # TODO: Check if taking away vmap hurts accuracy.
        x = (x - self.data_mean) / jnp.sqrt(jnp.diag(self.data_cov))
        y, log_det = self.__call__(x)
        log_det = log_det + self.base_dist.log_prob(y)
        return log_det

    def print_parameters(self):
        print("RQSpline parameters:")
