from typing import Callable
import jax
import jax.numpy as jnp
from tqdm import tqdm
from flowMC.proposal.base import ProposalBase
from jaxtyping import PyTree, Array, Float, Int, PRNGKeyArray


class GaussianRandomWalk(ProposalBase):
    """
    Gaussian random walk sampler class builiding the rw_sampler method

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    step_size: Float

    def __init__(
        self,
        logpdf: Callable[[Float[Array, "n_dim"], PyTree], Float],
        jit: bool,
        step_size: Float,
    ):
        super().__init__(logpdf, jit, step_size=step_size)
        self.step_size = step_size
        self.logpdf = logpdf

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, "n_dim"],
        log_prob: Float[Array, "1"],
        data: PyTree,
    ) -> tuple[Float[Array, "n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """
        Random walk gaussian kernel.
        This is a kernel that only evolve a single chain.

        Args:
            rng_key (PRNGKeyArray): Jax PRNGKey
            position (Float[Array, "n_dim"]): current position of the chain
            log_prob (Float[Array, "1"]): current log-probability of the chain
            data (PyTree): data to be passed to the logpdf function

        Returns:
            position (Float[Array, "n_dim"]): new position of the chain
            log_prob (Float[Array, "1"]): new log-probability of the chain
            do_accept (Int[Array, "1"]): whether the new position is accepted
        """

        key1, key2 = jax.random.split(rng_key)
        move_proposal: Float[Array, "n_dim"] = (
            jax.random.normal(key1, shape=position.shape) * self.step_size
        )

        proposal = position + move_proposal
        proposal_log_prob: Float[Array, "n_dim"] = self.logpdf(proposal, data)

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept = log_uniform < proposal_log_prob - log_prob

        position = jnp.where(do_accept, proposal, position)
        log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
        return position, log_prob, do_accept

    def update(self, i, state) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep n_dim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        key, positions, log_p, acceptance, data = state
        _, key = jax.random.split(key)
        new_position, new_log_p, do_accept = self.kernel(
            key, positions[i - 1], log_p[i - 1], data
        )
        positions = positions.at[i].set(new_position)
        log_p = log_p.at[i].set(new_log_p)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, log_p, acceptance, data)

    def sample(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        initial_position: Float[Array, "n_chains n_dim"],
        data: PyTree,
        verbose: bool = False,
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "n_chains n_steps n_dim"],
        Float[Array, "n_chains n_steps 1"],
        Int[Array, "n_chains n_steps 1"],
    ]:
        logp = self.logpdf_vmap(initial_position, data)
        n_chains = rng_key.shape[0]
        acceptance = jnp.zeros((n_chains, n_steps))
        all_positions = (
            jnp.zeros((n_chains, n_steps) + initial_position.shape[-1:])
        ) + initial_position[:, None]
        all_logp = jnp.zeros((n_chains, n_steps)) + logp[:, None]
        state = (rng_key, all_positions, all_logp, acceptance, data)
        if verbose:
            iterator_loop = tqdm(
                range(1, n_steps),
                desc="Sampling Locally",
                miniters=int(n_steps / 10),
            )
        else:
            iterator_loop = range(1, n_steps)

        for i in iterator_loop:
            state = self.update_vmap(i, state)
        return state[:-1]
