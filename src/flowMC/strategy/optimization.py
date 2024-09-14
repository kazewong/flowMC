from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from flowMC.proposal.base import ProposalBase
from flowMC.proposal.NF_proposal import NFProposal
from flowMC.strategy.base import Strategy
from flowMC.utils.debug import flush


class optimization_Adam(Strategy):
    """
    Optimize a set of chains using Adam optimization.
    Note that if the posterior can go to infinity, this optimization scheme is likely to return NaNs.

    Args:
        n_steps: int = 100
            Number of optimization steps.
        learning_rate: float = 1e-2
            Learning rate for the optimization.
        noise_level: float = 10
            Variance of the noise added to the gradients.
    """

    n_steps: int = 100
    learning_rate: float = 1e-2
    noise_level: float = 10
    bounds: Float[Array, "n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]])

    @property
    def __name__(self):
        return "AdamOptimization"

    def __init__(
        self,
        bounds: Float[Array, "n_dim 2"] = jnp.array([[-jnp.inf, jnp.inf]]),
        **kwargs,
    ):
        class_keys = list(self.__class__.__annotations__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

        self.solver = optax.chain(
            optax.adam(learning_rate=self.learning_rate),
        )

        self.bounds = bounds

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        local_sampler: ProposalBase,
        global_sampler: NFProposal,
        initial_position: Float[Array, " n_chain n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray, Float[Array, " n_chain n_dim"], ProposalBase, NFProposal, PyTree
    ]:
        def loss_fn(params: Float[Array, " n_dim"]) -> Float:
            return -local_sampler.logpdf(params, data)

        grad_fn = jax.jit(jax.grad(loss_fn))

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

        flush("Using Adam optimization")
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, initial_position.shape[0])
        optimized_positions = jax.vmap(_single_optimize, in_axes=(0, 0))(
            keys, initial_position
        )

        summary = {}
        summary["initial_positions"] = initial_position
        summary["initial_log_prob"] = local_sampler.logpdf_vmap(initial_position, data)
        summary["final_positions"] = optimized_positions
        summary["final_log_prob"] = local_sampler.logpdf_vmap(optimized_positions, data)

        if (
            jnp.isinf(summary["final_log_prob"]).any()
            or jnp.isnan(summary["final_log_prob"]).any()
        ):
            print("Warning: Optimization accessed infinite or NaN log-probabilities.")

        return rng_key, optimized_positions, local_sampler, global_sampler, summary

    def optimize(
        self,
        rng_key: PRNGKeyArray,
        objective: Callable,
        initial_position: Float[Array, " n_chain n_dim"],
    ):
        """
        Standalone optimization function that takes an objective function and returns the optimized positions.

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

        summary = {}
        summary["initial_positions"] = initial_position
        summary["initial_log_prob"] = jax.jit(jax.vmap(objective))(initial_position)
        summary["final_positions"] = optimized_positions
        summary["final_log_prob"] = jax.jit(jax.vmap(objective))(optimized_positions)

        if (
            jnp.isinf(summary["final_log_prob"]).any()
            or jnp.isnan(summary["final_log_prob"]).any()
        ):
            print("Warning: Optimization accessed infinite or NaN log-probabilities.")

        return rng_key, optimized_positions, summary


class Evosax_CMA_ES(Strategy):
    def __init__(
        self,
        **kwargs,
    ):
        class_keys = list(self.__class__.__annotations__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        local_sampler: ProposalBase,
        global_sampler: NFProposal,
        initial_position: Array,
        data: dict,
    ) -> tuple[PRNGKeyArray, Array, ProposalBase, NFProposal, PyTree]:
        raise NotImplementedError
