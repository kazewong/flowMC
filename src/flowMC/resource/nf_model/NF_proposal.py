from math import ceil

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from typing import Callable

from flowMC.resource.nf_model.base import NFModel
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF


class NFProposal(ProposalBase):
    n_flow_sample: int

    def __repr__(self):
        return "NF proposal with " + self.model.__repr__()

    def __init__(self, model: NFModel, n_flow_sample: int = 1000):
        super().__init__()
        self.model = model
        self.n_flow_sample = n_flow_sample

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, "n_step n_dim"], Float[Array, "n_step 1"], Int[Array, "n_step 1"]
    ]:
        n_steps = data["n_steps"]
        n_dims = position.shape[-1]

        rng_key, subkey = random.split(rng_key)

        # nf_current is size (1, n_dim)
        log_prob_nf_current = self.model.log_prob(position)

        # All these are size (n_steps, n_dim)
        proposal_position = self.sample_flow(subkey, n_steps, n_dims)
        log_prob_proposed = jax.vmap(logpdf, in_axes=(0, None))(proposal_position, data)
        log_prob_nf_proposed = jax.vmap(self.model.log_prob)(proposal_position)

        def body(carry, data):
            (
                rng_key,
                position_current,
                log_prob_current,
                log_prob_nf_current,
            ) = carry
            (position_proposed, log_prob_proposal, log_prob_nf_proposal) = data

            rng_key, subkey = random.split(rng_key)
            ratio = (log_prob_proposal - log_prob_current) - (
                log_prob_nf_proposal - log_prob_nf_current
            )
            uniform_random = jnp.log(jax.random.uniform(subkey))
            do_accept = uniform_random < ratio
            position_current = jnp.where(do_accept, position_proposed, position_current)
            log_prob_current = jnp.where(do_accept, log_prob_proposal, log_prob_current)
            log_prob_nf_current = jnp.where(
                do_accept, log_prob_nf_proposal, log_prob_nf_current
            )

            return (rng_key, position_current, log_prob_current, log_prob_nf_current), (
                position_current,
                log_prob_current,
                do_accept,
            )

        _, (positions, log_prob, do_accept) = jax.lax.scan(
            body,
            (
                rng_key,
                position,
                log_prob,
                log_prob_nf_current,
            ),
            (proposal_position, log_prob_proposed, log_prob_nf_proposed),
        )

        return positions, log_prob, do_accept

    def sample_flow(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        n_dims: int,
    ):
        total_size = n_steps * n_dims
        if total_size > self.n_flow_sample:
            rng_key = rng_key
            n_batch = ceil(total_size / self.n_flow_sample)
            n_sample = total_size // n_batch
            proposal_position = jnp.zeros((n_batch, n_sample, n_dims))
            for i in range(n_batch):
                rng_key, subkey = random.split(rng_key)
                proposal_position = proposal_position.at[i].set(
                    self.model.sample(subkey, n_sample)
                )

            proposal_position = proposal_position.reshape(-1, n_dims)[:total_size]

        else:
            proposal_position = self.model.sample(rng_key, n_steps)

        proposal_position = proposal_position.reshape(n_steps, n_dims)

        return proposal_position

    def print_parameters(self):
        # TODO: Implement this
        raise NotImplementedError

    def save_resource(self, path):
        # TODO: Implement this
        raise NotImplementedError

    def load_resource(self, path):
        # TODO: Implement this
        raise NotImplementedError
