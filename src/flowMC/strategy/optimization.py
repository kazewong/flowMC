from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource


class AdamOptimization(Strategy):
    """Optimize a set of chains using Adam optimization. Note that if the posterior can
    go to infinity, this optimization scheme is likely to return NaNs.

    Args:
        n_steps: int = 100
            Number of optimization steps.
        learning_rate: float = 1e-2
            Learning rate for the optimization.
        noise_level: float = 10
            Variance of the noise added to the gradients.
        bounds: Float[Array, " n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]])
            Bounds for the optimization. The optimization will be projected to these bounds.
            If bounds has shape (1, 2), it will be broadcast to all dimensions. For n_dim > 1,
            passing a (1, 2) array applies the same bound to every dimension. To specify different
            bounds per dimension, provide an array of shape (n_dim, 2).
    """

    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    n_steps: int = 100
    learning_rate: float = 1e-2
    noise_level: float = 10
    bounds: Float[Array, " n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]])

    def __repr__(self):
        return "AdamOptimization"

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_steps: int = 100,
        learning_rate: float = 1e-2,
        noise_level: float = 10,
        bounds: Float[Array, " n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]]),
    ):
        self.logpdf = logpdf
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        self.bounds = bounds

        # Validate bounds shape
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"bounds must have shape (n_dim, 2) or (1, 2), got {bounds.shape}"
            )
        # If bounds is (1, 2), it will be broadcast to all dimensions. If not, check compatibility.
        # Try to infer n_dim from logpdf signature or initial_position, but here we can't, so warn in runtime.

        self.solver = optax.chain(
            optax.adam(learning_rate=self.learning_rate),
        )

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, " n_chain n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        dict[str, Resource],
        Float[Array, " n_chain n_dim"],
    ]:
        def loss_fn(params: Float[Array, " n_dim"], data: dict) -> Float:
            return -self.logpdf(params, data)

        rng_key, optimized_positions, _ = self.optimize(
            rng_key, loss_fn, initial_position, data
        )

        return rng_key, resources, optimized_positions

    def optimize(
        self,
        rng_key: PRNGKeyArray,
        objective: Callable,
        initial_position: Float[Array, " n_chain n_dim"],
        data: dict,
    ):
        # Validate bounds shape against n_dim
        n_dim = initial_position.shape[-1]
        if not (self.bounds.shape[0] == 1 or self.bounds.shape[0] == n_dim):
            raise ValueError(
                f"bounds shape {self.bounds.shape} is incompatible with n_dim={n_dim}. "
                "Provide bounds of shape (1, 2) for broadcasting or (n_dim, 2) for per-dimension bounds."
            )

        """Optimization kernel. This can be used independently of the __call__ method.

        Args:
            rng_key: PRNGKeyArray
                Random key for the optimization.
            objective: Callable
                Objective function to optimize.
            initial_position: Float[Array, " n_chain n_dim"]
                Initial positions for the optimization.
            data: dict
                Data to pass to the objective function.

        Returns:
            rng_key: PRNGKeyArray
                Updated random key.
            optimized_positions: Float[Array, " n_chain n_dim"]
                Optimized positions.
            final_log_prob: Float[Array, " n_chain"]
                Final log-probabilities of the optimized positions.
        """
        grad_fn = jax.jit(jax.grad(objective))

        def _kernel(carry, _step):
            key, params, opt_state = carry

            key, subkey = jax.random.split(key)
            grad = grad_fn(params, data) * (
                1 + jax.random.normal(subkey) * self.noise_level
            )
            updates, opt_state = self.solver.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = optax.projections.projection_box(
                params, self.bounds[:, 0], self.bounds[:, 1]
            )
            return (key, params, opt_state), None

        def _single_optimize(
            key: PRNGKeyArray,
            initial_position: Float[Array, " n_dim"],
        ) -> Float[Array, " n_dim"]:
            opt_state = self.solver.init(initial_position)

            (key, params, opt_state), _ = jax.lax.scan(
                _kernel,
                (key, initial_position, opt_state),
                jnp.arange(self.n_steps),
            )

            return params  # type: ignore

        print("Using Adam optimization")
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, initial_position.shape[0])
        optimized_positions = jax.vmap(_single_optimize, in_axes=(0, 0))(
            keys, initial_position
        )

        final_log_prob = jax.vmap(self.logpdf, in_axes=(0, None))(
            optimized_positions, data
        )

        if jnp.isinf(final_log_prob).any() or jnp.isnan(final_log_prob).any():
            print("Warning: Optimization accessed infinite or NaN log-probabilities.")

        return rng_key, optimized_positions, final_log_prob
