from abc import abstractmethod
from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, Array, Float, Int, PRNGKeyArray


@jax.tree_util.register_pytree_node_class
class ProposalBase:
    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        """
        Initialize the sampler class
        """
        self.logpdf = logpdf
        self.jit = jit
        self.params = params
        self.logpdf_vmap = jax.vmap(logpdf, in_axes=(0, None))
        self.kernel_vmap = jax.vmap(self.kernel, in_axes=(0, 0, 0, None))
        self.update_vmap = jax.vmap(
            self.update,
            in_axes=(None, (0, 0, 0, 0, None)),
            out_axes=(0, 0, 0, 0, None),
        )
        if self.jit is True:
            self.logpdf_vmap = jax.jit(self.logpdf_vmap)
            self.kernel = jax.jit(self.kernel)
            self.kernel_vmap = jax.jit(self.kernel_vmap)
            self.update = jax.jit(self.update)
            self.update_vmap = jax.jit(self.update_vmap)

    def precompilation(self, n_chains, n_dims, n_step, data):
        if self.jit is True:
            print("jit is requested, precompiling kernels and update...")
        else:
            print("jit is not requested, compiling only vmap functions...")

        key = jax.random.split(jax.random.PRNGKey(0), n_chains)

        self.logpdf_vmap(jnp.ones((n_chains, n_dims)), data)
        self.kernel_vmap(
            key,
            jnp.ones((n_chains, n_dims)),
            jnp.ones((n_chains, 1)),
            data,
        )
        self.update_vmap(
            1,
            (
                key,
                jnp.ones((n_chains, n_step, n_dims)),
                jnp.ones((n_chains, n_step, 1)),
                jnp.zeros((n_chains, n_step, 1)),
                data,
            ),
        )

    @abstractmethod
    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, "nstep ndim"],
        log_prob: Float[Array, "nstep 1"],
        data: PyTree,
        params: dict,
    ) -> tuple[
        Float[Array, "nstep ndim"], Float[Array, "nstep 1"], Int[Array, "n_step 1"]
    ]:
        """
        Kernel for one step in the proposal cycle.
        """

    @abstractmethod
    def update(self, i, state) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep ndim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        """
        Make the update function for multiple steps
        """

    @abstractmethod
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
        """
        Make the sampler for multiple chains given initial positions
        """

    def tree_flatten(self):
        children = ()

        aux_data = {"logpdf": self.logpdf, "jit": self.jit, "params": self.params}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
