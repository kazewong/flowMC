import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from tqdm import tqdm
from typing import Callable

from flowMC.proposal.base import ProposalBase
from flowMC.proposal.NF_proposal import NFProposal
from flowMC.strategy.base import Strategy


class Adam(Strategy):

    n_steps: int = 100
    learning_rate: float = 1e-2

    def __init__(
        self,
        **kwargs,
    ):
        class_keys = list(self.__class__.__annotations__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

        self.solver = optax.adam(learning_rate=self.learning_rate)

    def _single_optimize(
        self,
        loss_fn: Callable,
        initial_position: Float[Array, " n_dim"],
    ) -> Float[Array, " n_dim"]:

        opt_state = self.solver.init(initial_position)
        grad_fn = jax.jit(jax.grad(loss_fn))

        (params, opt_state, _), _ = jax.lax.scan(
            self._kernel,
            (initial_position, opt_state, grad_fn),
            jnp.arange(self.n_steps),
        )

        return params  # type: ignore

    def _kernel(self, carry, data):
        params, opt_state, grad_fn = carry

        grad = grad_fn(params)
        updates, opt_state = self.solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state, grad_fn), None

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

        optimized_positions = jax.vmap(self._single_optimize, in_axes=(None, 0))(
            loss_fn, initial_position
        )

        return rng_key, optimized_positions, local_sampler, global_sampler, data


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