from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from flowMC.resource.local_kernel.base import ProposalBase


class GaussianRandomWalk(ProposalBase):
    """
    Gaussian random walk sampler class.

    """

    step_size: Float

    def __repr__(self):
        return "Gaussian Random Walk with step size " + str(self.step_size)

    def __init__(
        self,
        step_size: Float,
    ):
        super().__init__()
        self.step_size = step_size

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
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
        proposal_log_prob: Float[Array, "n_dim"] = log_pdf(proposal, data)

        log_uniform = jnp.log(jax.random.uniform(key2))
        do_accept = log_uniform < proposal_log_prob - log_prob

        position = jnp.where(do_accept, proposal, position)
        log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
        return position, log_prob, do_accept

    def print_parameters(self):
        print("Gaussian Random Walk parameters:")
        print("Step size: ", self.step_size)

    
    def save_resource(self, path):
        pass
    
    def load_resource(self, path):
        raise NotImplementedError