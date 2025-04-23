from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree

from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.logPDF import LogPDF


class HMC(ProposalBase):
    """Hamiltonian Monte Carlo sampler class builiding the hmc_sampler method from
    target logpdf.

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """

    condition_matrix: Float[Array, " n_dim n_dim"]
    step_size: Float
    leapfrog_coefs: Float[Array, " n_leapfrog n_dim"]

    @property
    def n_leapfrog(self) -> Int:
        return self.leapfrog_coefs.shape[0] - 2

    def __repr__(self):
        return (
            "HMC with step size "
            + str(self.step_size)
            + " and "
            + str(self.n_leapfrog)
            + " leapfrog steps"
        )

    def __init__(
        self,
        condition_matrix: Float[Array, " n_dim n_dim"] | Float = 1,
        step_size: Float = 0.1,
        n_leapfrog: Int = 10,
    ):
        self.condition_matrix = condition_matrix
        self.step_size = step_size

        coefs = jnp.ones((n_leapfrog + 2, 2))
        coefs = coefs.at[0].set(jnp.array([0, 0.5]))
        coefs = coefs.at[-1].set(jnp.array([1, 0.5]))
        self.leapfrog_coefs = coefs

    def get_initial_hamiltonian(
        self,
        potential: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        kinetic: Callable[
            [Float[Array, " n_dim"], Float[Array, " n_dim"]], Float[Array, "1"]
        ],
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        data: PyTree,
    ):
        """Compute the value of the Hamiltonian from positions with initial momentum
        draw at random from the standard normal distribution."""

        momentum = (
            jax.random.normal(rng_key, shape=position.shape)
            * self.condition_matrix**-0.5
        )
        return potential(position, data) + kinetic(momentum, self.condition_matrix)

    def leapfrog_kernel(self, kinetic, potential, carry, extras):
        position, momentum, data, metric, index = carry
        position = position + self.step_size * self.leapfrog_coefs[index][0] * jax.grad(
            kinetic
        )(momentum, metric)
        momentum = momentum - self.step_size * self.leapfrog_coefs[index][1] * jax.grad(
            potential
        )(position, data)
        index = index + 1
        return (position, momentum, data, metric, index), extras

    def leapfrog_step(
        self,
        leapfrog_kernel: Callable,
        position: Float[Array, " n_dim"],
        momentum: Float[Array, " n_dim"],
        data: PyTree,
        metric: Float[Array, " n_dim n_dim"],
    ) -> tuple[Float[Array, " n_dim"], Float[Array, " n_dim"]]:
        print("Compiling leapfrog step")
        (position, momentum, data, metric, index), _ = jax.lax.scan(
            leapfrog_kernel,
            (position, momentum, data, metric, 0),
            jnp.arange(self.n_leapfrog + 2),
        )
        return position, momentum

    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, " n_dim"],
        log_prob: Float[Array, "1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """Note that since the potential function is the negative log likelihood,
        hamiltonian is going down, but the likelihood value should go up.

        Args:
            rng_key (n_chains, 2): random key
            position (n_chains,  n_dim): current position
            PE (n_chains, ): Potential energy of the current position
        """

        def potential(x: Float[Array, " n_dim"], data: PyTree) -> Float[Array, "1"]:
            return -logpdf(x, data)

        def kinetic(
            p: Float[Array, " n_dim"], metric: Float[Array, " n_dim"]
        ) -> Float[Array, "1"]:
            return 0.5 * (p**2 * metric).sum()

        leapfrog_kernel = jax.tree_util.Partial(
            self.leapfrog_kernel, kinetic, potential
        )
        leapfrog_step = jax.tree_util.Partial(self.leapfrog_step, leapfrog_kernel)

        key1, key2 = jax.random.split(rng_key)

        momentum: Float[Array, " n_dim"] = (
            jax.random.normal(key1, shape=position.shape) * self.condition_matrix**-0.5
        )
        momentum = jnp.dot(
            jax.random.normal(key1, shape=position.shape),
            jnp.linalg.cholesky(jnp.linalg.inv(self.condition_matrix)).T,
        )
        H = -log_prob + kinetic(momentum, self.condition_matrix)
        proposed_position, proposed_momentum = leapfrog_step(
            position, momentum, data, self.condition_matrix
        )
        proposed_PE = potential(proposed_position, data)
        proposed_ham = proposed_PE + kinetic(proposed_momentum, self.condition_matrix)
        log_acc = H - proposed_ham
        log_uniform = jnp.log(jax.random.uniform(key2))

        do_accept = log_uniform < log_acc

        position = jnp.where(do_accept, proposed_position, position)  # type: ignore
        log_prob = jnp.where(do_accept, -proposed_PE, log_prob)  # type: ignore

        return position, log_prob, do_accept

    def print_parameters(self):
        print("HMC parameters:")
        print(f"step_size: {self.step_size}")
        print(f"n_leapfrog: {self.n_leapfrog}")
        print(f"condition_matrix: {self.condition_matrix}")

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path):
        raise NotImplementedError
