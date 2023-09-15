from typing import Tuple
import jax
import jax.numpy as jnp
from jax import random, vmap
from tqdm import tqdm
from flowMC.nfmodel.base import NFModel
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable
from flowMC.sampler.Proposal_Base import ProposalBase
from jaxtyping import Array, Float, Int, PRNGKeyArray
import equinox as eqx
from math import ceil


@jax.tree_util.register_pytree_node_class
class NFProposal(ProposalBase):

    model: NFModel

    def __init__(
        self, logpdf: Callable, jit: bool, model: NFModel, n_sample_max: int = 10000
    ):
        super().__init__(logpdf, jit, {})
        self.model = model
        self.n_sample_max = n_sample_max
        self.update_vmap = jax.vmap(self.update, in_axes=(None, (0)))
        if self.jit == True:
            self.update_vmap = jax.jit(self.update_vmap)

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, "ndim"],
        proposal_position: Float[Array, "ndim"],
        log_prob_initial: Float[Array, "1"],
        log_prob_proposal: Float[Array, "1"],
        log_prob_nf_initial: Float[Array, "1"],
        log_prob_nf_proposal: Float[Array, "1"],
    ) -> tuple[Float[Array, "ndim"], Float[Array, "1"], Int[Array, "1"]]:
        rng_key, subkey = random.split(rng_key)

        ratio = (log_prob_proposal - log_prob_initial) - (
            log_prob_nf_proposal - log_prob_nf_initial
        )
        uniform_random = jnp.log(jax.random.uniform(subkey))
        do_accept = uniform_random < ratio
        position = jnp.where(do_accept, proposal_position, initial_position)
        log_prob = jnp.where(do_accept, log_prob_proposal, log_prob_initial)
        log_prob_nf = jnp.where(do_accept, log_prob_nf_proposal, log_prob_nf_initial)
        return position, log_prob, log_prob_nf, do_accept

    def update(
        self, i, state
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep ndim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        (
            key,
            positions,
            proposal,
            log_prob,
            log_prob_proposal,
            log_prob_nf,
            log_prob_nf_proposal,
            acceptance,
        ) = state
        key, subkey = jax.random.split(key)
        new_position, new_log_prob, new_log_prob_nf, do_accept = self.kernel(
            subkey,
            positions[i - 1],
            proposal[i],
            log_prob[i - 1],
            log_prob_proposal[i],
            log_prob_nf[i - 1],
            log_prob_nf_proposal[i],
        )
        positions = positions.at[i].set(new_position)
        log_prob = log_prob.at[i].set(new_log_prob)
        log_prob_nf = log_prob_nf.at[i].set(new_log_prob_nf)
        acceptance = acceptance.at[i].set(do_accept)
        return (
            key,
            positions,
            proposal,
            log_prob,
            log_prob_proposal,
            log_prob_nf,
            log_prob_nf_proposal,
            acceptance,
        )

    def sample(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        initial_position: Float[Array, "n_chains ndim"],
        data: PyTree,
        verbose: bool = False,
    ) -> tuple[
        Float[Array, "n_chains n_steps ndim"],
        Float[Array, "n_chains n_steps 1"],
        Int[Array, "n_chains n_steps 1"],
    ]:
        n_chains = initial_position.shape[0]
        n_dim = initial_position.shape[-1]
        rng_key, *subkeys = random.split(rng_key, 3)
        total_size = initial_position.shape[0] * n_steps
        log_prob_initial = self.logpdf_vmap(initial_position, data)[:, None]
        log_prob_nf_initial = self.model.log_prob(initial_position)[:, None]
        if total_size > self.n_sample_max:
            sampling_key = subkeys[0]
            n_batch = ceil(total_size / self.n_sample_max)
            n_sample = total_size // n_batch
            proposal_position = jnp.zeros((n_batch, n_sample, initial_position.shape[-1]))
            log_prob_proposal = jnp.zeros((n_batch, n_sample))
            log_prob_nf_proposal = jnp.zeros((n_batch, n_sample))
            for i in range(n_batch):
                sampling_key, subkey = random.split(sampling_key)
                proposal_position = proposal_position.at[i].set(
                    self.model.sample(subkey, n_sample)
                )
                log_prob_proposal = log_prob_proposal.at[i].set(
                    self.logpdf_vmap(proposal_position[i], data)
                )
                log_prob_nf_proposal = log_prob_nf_proposal.at[i].set(
                    self.model.log_prob(proposal_position[i])
                )

            proposal_position = proposal_position.reshape(-1, n_dim)[:total_size]
            log_prob_proposal = log_prob_proposal.reshape(-1)[:total_size]
            log_prob_nf_proposal = log_prob_nf_proposal.reshape(-1)[:total_size]

        else:
            proposal_position = self.model.sample(subkeys[0], total_size)
            log_prob_proposal = self.logpdf_vmap(proposal_position, data)
            log_prob_nf_proposal = self.model.log_prob(proposal_position)

        proposal_position = proposal_position.reshape(n_chains, n_steps, n_dim)
        log_prob_proposal = log_prob_proposal.reshape(n_chains, n_steps)
        log_prob_nf_proposal = log_prob_nf_proposal.reshape(n_chains, n_steps)

        state = (
            jax.random.split(subkeys[1], n_chains),
            jnp.zeros((n_chains, n_steps, n_dim)) + initial_position[:, None],
            proposal_position,
            jnp.zeros(
                (
                    n_chains,
                    n_steps,
                )
            )
            + log_prob_initial,
            log_prob_proposal,
            jnp.zeros(
                (
                    n_chains,
                    n_steps,
                )
            )
            + log_prob_nf_initial,
            log_prob_nf_proposal,
            jnp.zeros(
                (
                    n_chains,
                    n_steps,
                )
            ),
        )
        if verbose:
            iterator_loop = tqdm(
                range(1, n_steps),
                desc="Sampling Globally",
                miniters=int(n_steps / 10),
            )
        else:
            iterator_loop = range(1, n_steps)
        for i in iterator_loop:
            state = self.update_vmap(i, state)
        return rng_key, state[1], state[3], state[5], state[7]

    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data["model"] = self.model
        aux_data["n_sample_max"] = self.n_sample_max
        return (children, aux_data)
