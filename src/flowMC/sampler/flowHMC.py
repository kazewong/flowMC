import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import NFModel
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable
from flowMC.sampler.HMC import HMC
from flowMC.sampler.NF_proposal import NFProposal
from jaxtyping import Array, Float, Int, PRNGKeyArray
from math import ceil
from jax import random


@jax.tree_util.register_pytree_node_class
class flowHMC(HMC, NFProposal):
    model: NFModel

    def __init__(
        self,
        logpdf: Callable,
        jit: bool,
        model: NFModel,
        n_sample_max: int = 10000,
        params: dict = {},
    ):
        super().__init__(logpdf, jit, params)
        self.model = model
        self.n_sample_max = n_sample_max
        self.update_vmap = jax.vmap(self.update, in_axes=(None, (0)))
        if self.jit is True:
            self.update_vmap = jax.jit(self.update_vmap)

    def covariance_estimate(self, points: Float[Array, "n_point n_dim"], k: int = 3) -> Float[Array, "n_point n_dim n_dim"]:
        distance = jax.lax.dot(points, points.T)
        neighbor_indcies = jax.lax.approx_min_k(distance, k=k)[1]
        neighbor_points = points[neighbor_indcies].swapaxes(1, 2)
        covariance = jax.vmap(jnp.cov)(neighbor_points)
        return covariance

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, "ndim"],
        log_prob: Float[Array, "1"],
        flow_position: Float[Array, "ndim"],
        flow_covariance: Float[Array, "ndim ndim"],
        data: PyTree,
    ) -> tuple[Float[Array, "ndim"], Float[Array, "1"], Int[Array, "1"]]:
        
        key1, key2, key3 = jax.random.split(rng_key, 3)

        momentum = (
            jax.random.normal(key1, shape=position.shape)
            * self.params["inverse_metric"] ** -0.5
        )

        # TODO: Double check whether I can compute the hamiltonian before the map
        initial_Ham = log_prob + self.kinetic(momentum, self.params)

        # Sample momentum

        # Push through map

        # Make HMC step

        # Compute acceptance probability

        pass

    def update(
        self, i, state
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep ndim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        pass

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
        pass

    def sample_flow(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        initial_position: Float[Array, "n_chains ndim"],
        data: PyTree,
    ):
        n_chains = initial_position.shape[0]
        n_dim = initial_position.shape[-1]
        total_size = initial_position.shape[0] * n_steps
        if total_size > self.n_sample_max:
            rng_key = rng_key
            n_batch = ceil(total_size / self.n_sample_max)
            n_sample = total_size // n_batch
            proposal_position = jnp.zeros(
                (n_batch, n_sample, initial_position.shape[-1])
            )
            proposal_covariance = jnp.zeros(
                (n_batch, n_sample, initial_position.shape[-1], initial_position.shape[-1])
            )
            log_prob_proposal = jnp.zeros((n_batch, n_sample))
            log_prob_nf_proposal = jnp.zeros((n_batch, n_sample))
            for i in range(n_batch):
                rng_key, subkey = random.split(rng_key)
                proposal_position = proposal_position.at[i].set(
                    self.model.sample(subkey, n_sample)
                )
                proposal_covariance = proposal_covariance.at[i].set(
                    self.covariance_estimate(proposal_position[i])
                )
                log_prob_proposal = log_prob_proposal.at[i].set(
                    self.logpdf_vmap(proposal_position[i], data)
                )
                log_prob_nf_proposal = log_prob_nf_proposal.at[i].set(
                    self.model.log_prob(proposal_position[i])
                )

            proposal_position = proposal_position.reshape(-1, n_dim)[:total_size]
            proposal_covariance = proposal_covariance.reshape(-1, n_dim, n_dim)[:total_size]
            log_prob_proposal = log_prob_proposal.reshape(-1)[:total_size]
            log_prob_nf_proposal = log_prob_nf_proposal.reshape(-1)[:total_size]

        else:
            proposal_position = self.model.sample(rng_key, total_size)
            proposal_covariance = self.covariance_estimate(proposal_position)
            log_prob_proposal = self.logpdf_vmap(proposal_position, data)
            log_prob_nf_proposal = self.model.log_prob(proposal_position)

        proposal_position = proposal_position.reshape(n_chains, n_steps, n_dim)
        proposal_covariance = proposal_covariance.reshape(n_chains, n_steps, n_dim, n_dim)
        log_prob_proposal = log_prob_proposal.reshape(n_chains, n_steps)
        log_prob_nf_proposal = log_prob_nf_proposal.reshape(n_chains, n_steps)

        return proposal_position, proposal_covariance, log_prob_proposal, log_prob_nf_proposal

    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data["model"] = self.model
        aux_data["n_sample_max"] = self.n_sample_max
        return (children, aux_data)
