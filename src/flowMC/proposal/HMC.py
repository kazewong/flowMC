from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from flowMC.proposal.base import ProposalBase
from flowMC.utils.debug import flush


class HMC(ProposalBase):
    """
    Hamiltonian Monte Carlo sampler class builiding the hmc_sampler method
    from target logpdf.

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    condition_matrix: Float[Array, " n_dim n_dim"]
    step_size: Float
    n_leapfrog: Int

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], PyTree], Float],
        jit: bool,
        condition_matrix: Float[Array, " n_dim n_dim"] | Float = 1,
        step_size: Float = 0.1,
        n_leapfrog: Int = 10,
    ):
        super().__init__(
            logpdf,
            jit,
            condition_matrix=condition_matrix,
            step_size=step_size,
            n_leapfrog=n_leapfrog,
        )

        self.potential: Callable[[Float[Array, " n_dim"], PyTree], Float] = (
            lambda x, data: -logpdf(x, data)
        )
        self.grad_potential: Callable[
            [Float[Array, " n_dim"], PyTree], Float[Array, " n_dim"]
        ] = jax.grad(self.potential)

        self.condition_matrix = condition_matrix
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

        coefs = jnp.ones((self.n_leapfrog + 2, 2))
        coefs = coefs.at[0].set(jnp.array([0, 0.5]))
        coefs = coefs.at[-1].set(jnp.array([1, 0.5]))
        self.leapfrog_coefs = coefs

        self.kinetic: Callable[
            [Float[Array, " n_dim"], Float[Array, " n_dim n_dim"]], Float
        ] = lambda p, metric: 0.5 * (p**2 * metric).sum()
        self.grad_kinetic = jax.grad(self.kinetic)

    def get_initial_hamiltonian(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        data: PyTree,
    ):
        """
        Compute the value of the Hamiltonian from positions with initial momentum draw
        at random from the standard normal distribution.
        """

        momentum = (
            jax.random.normal(rng_key, shape=position.shape)
            * self.condition_matrix**-0.5
        )
        return self.potential(position, data) + self.kinetic(
            momentum, self.condition_matrix
        )

    def leapfrog_kernel(self, carry, extras):
        position, momentum, data, metric, index = carry

        grad_kinetic_val = self.grad_kinetic(momentum, metric)
        flush(
            "proposal.HMC.leapfrog_kernel.grad_kinetic_val={grad_kinetic_val}",
            grad_kinetic_val=grad_kinetic_val,
        )

        position = (
            position + self.step_size * self.leapfrog_coefs[index][0] * grad_kinetic_val
        )

        grad_potential_val = self.grad_potential(position, data)
        flush(
            "proposal.HMC.leapfrog_kernel.grad_potential={grad_potential_val}",
            grad_potential_val=grad_potential_val,
        )

        momentum = (
            momentum
            - self.step_size * self.leapfrog_coefs[index][1] * grad_potential_val
        )

        index = index + 1
        return (position, momentum, data, metric, index), extras

    def leapfrog_step(
        self,
        position: Float[Array, " n_dim"],
        momentum: Float[Array, " n_dim"],
        data: PyTree,
        metric: Float[Array, " n_dim n_dim"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        (position, momentum, data, metric, index), _ = jax.lax.scan(
            self.leapfrog_kernel,
            (position, momentum, data, metric, 0),
            jnp.arange(self.n_leapfrog + 2),
        )
        return position, momentum

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """
        Note that since the potential function is the negative log likelihood,
        hamiltonian is going down, but the likelihood value should go up.

        Args:
            rng_key (n_chains, 2): random key
            position (n_chains,  n_dim): current position
            PE (n_chains, ): Potential energy of the current position
        """
        key1, key2 = jax.random.split(rng_key)

        momentum: Float[Array, " n_dim"] = (
            jax.random.normal(key1, shape=position.shape) * self.condition_matrix**-0.5
        )
        momentum = jnp.dot(
            jax.random.normal(key1, shape=position.shape),
            jnp.linalg.cholesky(jnp.linalg.inv(self.condition_matrix)).T,
        )
        H = -log_prob + self.kinetic(momentum, self.condition_matrix)
        proposed_position, proposed_momentum = self.leapfrog_step(
            position, momentum, data, self.condition_matrix
        )
        proposed_PE = self.potential(proposed_position, data)
        proposed_ham = proposed_PE + self.kinetic(
            proposed_momentum, self.condition_matrix
        )
        log_acc = H - proposed_ham
        log_uniform = jnp.log(jax.random.uniform(key2))

        do_accept = log_uniform < log_acc

        position = jnp.where(do_accept, proposed_position, position)
        log_prob = jnp.where(do_accept, -proposed_PE, log_prob)  # type: ignore

        return position, log_prob, do_accept

    def update(
        self, i, state
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep  n_dim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        key, positions, PE, acceptance, data = state
        _, key = jax.random.split(key)
        new_position, new_PE, do_accept = self.kernel(
            key, positions[i - 1], PE[i - 1], data
        )
        positions = positions.at[i].set(new_position)
        PE = PE.at[i].set(new_PE)
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, PE, acceptance, data)

    def sample(
        self,
        rng_key: PRNGKeyArray,
        n_steps: int,
        initial_position: Float[Array, "n_chains  n_dim"],
        data: PyTree,
        verbose: bool = False,
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "n_chains n_steps  n_dim"],
        Float[Array, "n_chains n_steps 1"],
        Int[Array, "n_chains n_steps 1"],
    ]:
        keys = jax.vmap(jax.random.split)(rng_key)
        rng_key = keys[:, 0]
        logp = self.logpdf_vmap(initial_position, data)
        n_chains = rng_key.shape[0]
        acceptance = jnp.zeros((n_chains, n_steps))
        all_positions = (
            jnp.zeros((n_chains, n_steps) + initial_position.shape[-1:])
            + initial_position[:, None]
        )
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

        state = (state[0], state[1], state[2], state[3])
        return state
