import jax
import jax.numpy as jnp
from flowMC.nfmodel.base import NFModel
from jaxtyping import Array, PRNGKeyArray, PyTree
from typing import Callable
from flowMC.proposal.HMC import HMC
from flowMC.proposal.NF_proposal import NFProposal
from jaxtyping import Array, Float, Int, PRNGKeyArray
from math import ceil
from jax import random
from tqdm import tqdm

###################################
# This is not in production yet
###################################


# Note that the inverse metric needs float64 precision
jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
class flowHMC(HMC, NFProposal):
    model: NFModel

    condition_matrix: Float[Array, "n_dim n_dim"]

    def __init__(
        self,
        logpdf: Callable,
        jit: bool,
        model: NFModel,
        n_sample_max: int = 10000,
        condition_matrix: Float[Array, "n_dim n_dim"] | Float = 1,
    ):
        super().__init__(logpdf, jit, condition_matrix=condition_matrix, model=model, n_sample_max=n_sample_max)
        self.kinetic = lambda p, M: 0.5 * (p @ M @ p)
        self.grad_kinetic = jax.grad(self.kinetic)
        self.model = model
        self.n_sample_max = n_sample_max
        self.production_covariance = condition_matrix
        self.update_vmap = jax.vmap(self.update, in_axes=(None, (0, 0, 0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, 0, 0, None))
        if self.jit is True:
            self.update_vmap = jax.jit(self.update_vmap)

    def covariance_estimate(
        self, points: Float[Array, "n_point n_dim"], k: int = 100
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
        flow_cov: Float[Array, "ndim ndim"],
        data: PyTree,
    ) -> tuple[Float[Array, "ndim"], Float[Array, "1"], Int[Array, "1"]]:

        key1, key2 = jax.random.split(rng_key)

        noise = jnp.abs(flow_cov).min()*0.1*jnp.eye(flow_cov.shape[0]) # Add small jitter to avoid numerical issues

        momentum = jnp.dot(
            jax.random.normal(key1, shape=position.shape),
            jnp.linalg.cholesky(flow_cov + noise).T)
        mass_matrix = jnp.linalg.inv(flow_cov + noise)

        # TODO: Double check whether I can compute the hamiltonian before the map
        initial_Ham = - log_prob + self.kinetic(momentum, mass_matrix)

        # First HMC part

        middle_position, middle_momentum = self.leapfrog_step(
            position, momentum, data, mass_matrix
        )

        # Push through map

        flow_start_prob = self.model.log_prob(middle_position[None])
        flow_end_prob = self.model.log_prob(flow_position[None])

        # Second HMC part

        final_position, final_momentum = self.leapfrog_step(
            flow_position, middle_momentum, data, mass_matrix
        )
        final_PE = self.potential(final_position, data)
        final_Ham = final_PE + self.kinetic(final_momentum, mass_matrix)

        # Compute acceptance probability

        log_acc = -(final_Ham - initial_Ham) - (flow_end_prob - flow_start_prob)

        uniform_random = jnp.log(jax.random.uniform(key2))
        do_accept = log_acc > uniform_random


        # Update position
        position = jnp.where(do_accept, final_position, position)
        log_prob = jnp.where(do_accept, - final_PE, log_prob)

        return position, log_prob[0], do_accept[0]

    def update(
        self, i, state
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep ndim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        key, positions, potential, acceptance, flow_position, flow_cov, data = state
        key, subkey = random.split(key)
        new_position, new_log_prob, do_accept = self.kernel(
            subkey,
            positions[i - 1],
            potential[i - 1],
            flow_position[i - 1],
            flow_cov[i - 1],
            data,
        )
        positions = positions.at[i].set(new_position)
        potential = potential.at[i].set(new_log_prob)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, potential, acceptance, flow_position, flow_cov, data)

    def sample(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        initial_position: Float[Array, "n_chains ndim"],
        data: PyTree,
        verbose: bool = False,
        mode: str = "training",
    ) -> tuple[
        Float[Array, "n_chains n_steps ndim"],
        Float[Array, "n_chains n_steps 1"],
        Int[Array, "n_chains n_steps 1"],
    ]:

        n_chains = initial_position.shape[0]
        n_dim = initial_position.shape[-1]
        log_prob_initial = self.logpdf_vmap(initial_position, data)

        rng_key, *subkey = random.split(rng_key, n_chains + 1)

        subkey = jnp.array(subkey)

        rng_key, nf_key = random.split(rng_key)

        proposal_position, proposal_cov = self.sample_flow(
            nf_key, initial_position, n_steps
        )

        # if mode == "production":
        #     if self.production_covariance is None:
        #         self.production_covariance = jnp.cov(proposal_position)
        proposal_cov = jnp.repeat(
            jnp.repeat(
                self.production_covariance[None, None],
                proposal_position.shape[0],
                axis=0,
            ),
            proposal_position.shape[1],
            axis=1,
        )

        state = (
            subkey,
            jnp.zeros((n_chains, n_steps, n_dim)) + initial_position[:, None],
            jnp.zeros((n_chains, n_steps)) + log_prob_initial[:, None],
            jnp.zeros((n_chains, n_steps)),
            proposal_position,
            proposal_cov,
            data,
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

        return (rng_key, state[1], state[2], state[3])

    def sample_flow(
        self,
        rng_key: PRNGKeyArray,
        initial_position: Float[Array, "n_chains ndim"],
        n_steps: int,
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
                (
                    n_batch,
                    n_sample,
                    initial_position.shape[-1],
                    initial_position.shape[-1],
                )
            )
            for i in range(n_batch):
                rng_key, subkey = random.split(rng_key)
                proposal_position = proposal_position.at[i].set(
                    self.model.sample(subkey, n_sample)
                )
                proposal_covariance = proposal_covariance.at[i].set(
                    self.covariance_estimate(proposal_position[i])
                )

            proposal_position = proposal_position.reshape(-1, n_dim)[:total_size]
            proposal_covariance = proposal_covariance.reshape(-1, n_dim, n_dim)[
                :total_size
            ]

        else:
            proposal_position = self.model.sample(rng_key, total_size)
            proposal_covariance = self.covariance_estimate(proposal_position)

        proposal_position = proposal_position.reshape(n_chains, n_steps, n_dim)
        proposal_covariance = proposal_covariance.reshape(
            n_chains, n_steps, n_dim, n_dim
        )

        return (
            proposal_position,
            proposal_covariance,
        )

    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data["model"] = self.model
        aux_data["n_sample_max"] = self.n_sample_max
        return (children, aux_data)
