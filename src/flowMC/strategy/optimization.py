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
    """

    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    n_steps: int = 100
    learning_rate: float = 1e-2
    noise_level: float = 10
    bounds: Float[Array, "n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]])

    def __repr__(self):
        return "AdamOptimization"

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_steps: int = 100,
        learning_rate: float = 1e-2,
        noise_level: float = 10,
        bounds: Float[Array, "n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]]),
    ):
        self.logpdf = logpdf
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        self.bounds = bounds

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
        Float[Array, "n_chains n_dim"],
    ]:
        def loss_fn(params: Float[Array, " n_dim"]) -> Float:
            return -self.logpdf(params, data)

        rng_key, optimized_positions = self.optimize(
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
        """Optimization kernel. This can be used independently of the __call__ method.

        Args:
            rng_key: PRNGKeyArray
                Random key for the optimization.
            objective: Callable
                Objective function to optimize.
            initial_position: Float[Array, " n_chain n_dim"]
                Initial positions for the optimization.
        """
        grad_fn = jax.jit(jax.grad(objective))

        def _kernel(carry, data):
            key, params, opt_state = carry

            key, subkey = jax.random.split(key)
            grad = grad_fn(params) * (1 + jax.random.normal(subkey) * self.noise_level)
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

        return rng_key, optimized_positions
