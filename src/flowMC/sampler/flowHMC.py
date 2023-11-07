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


# Note that the inverse metric needs float64 precision
jax.config.update("jax_enable_x64", True)


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
        self.kinetic = lambda p, M: 0.5 * (p**2 * M).sum()
        self.grad_kinetic = jax.grad(self.kinetic)
        self.model = model
        self.n_sample_max = n_sample_max
        self.update_vmap = jax.vmap(self.update, in_axes=(None, (0)))
        if self.jit is True:
            self.update_vmap = jax.jit(self.update_vmap)

    def covariance_estimate(
        self, points: Float[Array, "n_point n_dim"], k: int = 3
    ) -> Float[Array, "n_point n_dim n_dim"]:
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
        flow_metric: Float[Array, "ndim ndim"],
        data: PyTree,
    ) -> tuple[Float[Array, "ndim"], Float[Array, "1"], Int[Array, "1"]]:

        key1, key2 = jax.random.split(rng_key)

        momentum = (
            jax.random.normal(key1, shape=position.shape)
            * flow_metric ** -0.5
        )

        # TODO: Double check whether I can compute the hamiltonian before the map
        initial_Ham = log_prob + self.kinetic(momentum, flow_metric)

        # First HMC part

        middle_position, middle_momentum = self.leapfrog_step(
            position, momentum, data, flow_metric
        )

        # Push through map

        flow_start_prob = self.model.log_prob(middle_position)
        flow_end_prob = self.model.log_prob(flow_position)

        # Second HMC part

        final_position, final_momentum = self.leapfrog_step(
            flow_position, middle_momentum, data, flow_metric
        )
        final_PE = self.potential(final_position, data)
        final_Ham = final_PE + self.kinetic(final_momentum, flow_metric)

        # Compute acceptance probability

        log_acc = (final_Ham - initial_Ham) - (flow_end_prob - flow_start_prob)
        uniform_random = jnp.log(jax.random.uniform(key2))
        do_accept = log_acc > uniform_random

        # Update position
        position = jnp.where(do_accept, final_position, position)
        log_prob = jnp.where(do_accept, final_PE, log_prob)

        return position, log_prob, do_accept

    def update(
        self, i, state
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep ndim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        key, positions, PE, acceptance, flow_position, flow_metric, data = state
        key, subkey = random.split(key)
        new_position, new_log_prob, do_accept = self.kernel(
            subkey,
            positions[i-1],
            PE[i-1],
            flow_position[i-1],
            flow_metric[i-1],
            data)
        positions = positions.at[i].set(new_position)
        PE = PE.at[i].set(new_log_prob)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, PE, acceptance, flow_position, flow_metric, data)


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
        rng_key, *subkeys = random.split(rng_key, 3)

        n_chains = initial_position.shape[0]
        n_dim = initial_position.shape[-1]
        log_prob_initial = self.logpdf_vmap(initial_position, data)[:, None]
        log_prob_nf_initial = self.model.log_prob(initial_position)[:, None]

        proposal_position, proposal_metric, log_prob_proposal, log_prob_nf_proposal = self.sample_flow(
            subkeys[0], initial_position, data, n_steps
        )

            


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
            proposal_metric = jnp.zeros(
                (
                    n_batch,
                    n_sample,
                    initial_position.shape[-1],
                    initial_position.shape[-1],
                )
            )
            log_prob_proposal = jnp.zeros((n_batch, n_sample))
            log_prob_nf_proposal = jnp.zeros((n_batch, n_sample))
            for i in range(n_batch):
                rng_key, subkey = random.split(rng_key)
                proposal_position = proposal_position.at[i].set(
                    self.model.sample(subkey, n_sample)
                )
                proposal_metric = proposal_metric.at[i].set(
                    jax.vmap(jnp.linalg.inv)(
                        self.covariance_estimate(proposal_position[i])
                    )
                )
                log_prob_proposal = log_prob_proposal.at[i].set(
                    self.logpdf_vmap(proposal_position[i], data)
                )
                log_prob_nf_proposal = log_prob_nf_proposal.at[i].set(
                    self.model.log_prob(proposal_position[i])
                )

            proposal_position = proposal_position.reshape(-1, n_dim)[:total_size]
            proposal_metric = proposal_metric.reshape(-1, n_dim, n_dim)[
                :total_size
            ]
            log_prob_proposal = log_prob_proposal.reshape(-1)[:total_size]
            log_prob_nf_proposal = log_prob_nf_proposal.reshape(-1)[:total_size]

        else:
            proposal_position = self.model.sample(rng_key, total_size)
            proposal_metric = jax.vmap(jnp.linalg.inv)(self.covariance_estimate(proposal_position))
            log_prob_proposal = self.logpdf_vmap(proposal_position, data)
            log_prob_nf_proposal = self.model.log_prob(proposal_position)

        proposal_position = proposal_position.reshape(n_chains, n_steps, n_dim)
        proposal_metric = proposal_metric.reshape(
            n_chains, n_steps, n_dim, n_dim
        )
        log_prob_proposal = log_prob_proposal.reshape(n_chains, n_steps)
        log_prob_nf_proposal = log_prob_nf_proposal.reshape(n_chains, n_steps)

        return (
            proposal_position,
            proposal_metric,
            log_prob_proposal,
            log_prob_nf_proposal,
        )

    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data["model"] = self.model
        aux_data["n_sample_max"] = self.n_sample_max
        return (children, aux_data)
