from typing import Callable
import jax
import jax.numpy as jnp
from tqdm import tqdm
from flowMC.sampler.Proposal_Base import ProposalBase
from jaxtyping import PyTree, Array, Float, Int, PRNGKeyArray


class HMC(ProposalBase):
    """
    Hamiltonian Monte Carlo sampler class builiding the hmc_sampler method
    from target logpdf.

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        super().__init__(logpdf, jit, params)

        self.potential = lambda x, data: -logpdf(x, data)
        self.grad_potential = jax.grad(self.potential)

        self.params = params
        if "condition_matrix" in params:
            self.inverse_metric = params["condition_matrix"]
        else:
            self.inverse_metric = 1

        if "step_size" in params:
            self.step_size = params["step_size"]

        if "n_leapfrog" in params:
            self.n_leapfrog = params["n_leapfrog"]
        else:
            raise NotImplementedError

        self.kinetic = lambda p, params: 0.5 * (p**2 * params["inverse_metric"]).sum()
        self.grad_kinetic = jax.grad(self.kinetic)
        self.logpdf = self.potential

    def get_initial_hamiltonian(
        self,
        rng_key: jax.random.PRNGKey,
        position: jnp.array,
        data: jnp.array,
        params: dict,
    ):
        """
        Compute the value of the Hamiltonian from positions with initial momentum draw
        at random from the standard normal distribution.
        """

        momentum = (
            jax.random.normal(rng_key, shape=position.shape)
            * params["inverse_metric"] ** -0.5
        )
        return self.potential(position, data) + self.kinetic(momentum, params)

    def leapfrog_kernel(self, carry, extras):
        position, momentum, params, data = carry
        position = position + params["step_size"] * self.grad_kinetic(momentum, params)
        momentum = momentum - params["step_size"] * self.grad_potential(position, data)
        return (position, momentum, params, data), extras

    def leapfrog_step(self, position, momentum, data, params: dict):
        momentum = momentum - 0.5 * params["step_size"] * self.grad_potential(
            position, data
        )
        (position, momentum, params, data), _ = jax.lax.scan(
            self.leapfrog_kernel,
            (position, momentum, params, data),
            jnp.arange(self.n_leapfrog - 1),
        )
        position = position + params["step_size"] * self.grad_kinetic(momentum, params)
        momentum = momentum - 0.5 * params["step_size"] * self.grad_potential(
            position, data
        )
        return position, momentum

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, "ndim"],
        log_prob: Float[Array, "1"],
        data: PyTree,
    ) -> tuple[Float[Array, "ndim"], Float[Array, "1"], Int[Array, "1"]]:
        """
        Note that since the potential function is the negative log likelihood,
        hamiltonian is going down, but the likelihood value should go up.

        Args:
            rng_key (n_chains, 2): random key
            position (n_chains, n_dim): current position
            PE (n_chains, ): Potential energy of the current position
        """
        key1, key2 = jax.random.split(rng_key)

        momentum = (
            jax.random.normal(key1, shape=position.shape)
            * self.params["inverse_metric"] ** -0.5
        )
        H = log_prob + self.kinetic(momentum, self.params)
        proposed_position, proposed_momentum = self.leapfrog_step(
            position, momentum, data, self.params
        )
        proposed_PE = self.potential(proposed_position, data)
        proposed_ham = proposed_PE + self.kinetic(proposed_momentum, self.params)
        log_acc = H - proposed_ham
        log_uniform = jnp.log(jax.random.uniform(key2))

        do_accept = log_uniform < log_acc

        position = jnp.where(do_accept, proposed_position, position)
        log_prob = jnp.where(do_accept, proposed_PE, log_prob)

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
        initial_position: Float[Array, "n_chains ndim"],
        data: PyTree,
        verbose: bool = False,
    ) -> tuple[
        Float[Array, "n_chains n_steps ndim"],
        Float[Array, "n_chains n_steps 1"],
        Int[Array, "n_chains n_steps 1"],
    ]:
        keys = jax.vmap(jax.random.split)(rng_key)
        rng_key = keys[:, 0]
        rng_init = keys[:, 1]
        logp = self.logpdf_vmap(initial_position, data)
        n_chains = rng_key.shape[0]
        acceptance = jnp.zeros((n_chains, n_steps))
        all_positions = (
            jnp.zeros(
                (
                    n_chains,
                    n_steps,
                )
                + initial_position.shape[-1:]
            )
            + initial_position[:, None]
        )
        all_logp = (
            jnp.zeros(
                (
                    n_chains,
                    n_steps,
                )
            )
            + logp[:, None]
        )
        state = (rng_key, all_positions, all_logp, acceptance, data, self.params)

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

        state = (state[0], state[1], -state[2], state[3])
        return state
