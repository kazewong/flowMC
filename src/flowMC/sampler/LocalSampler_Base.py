from abc import abstractmethod
from typing import Callable
import jax
import jax.numpy as jnp
class LocalSamplerBase:

    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        """
        Initialize the sampler class
        """
        self.logpdf = logpdf
        self.jit = jit
        self.params = params

    def precompilation(self, n_chains, n_dims, n_step, data):

        if self.jit == True:
            print("jit is requested, precompiling kernels and update...")
        else:
            print("jit is not requested, compiling only vmap functions...")

        self.kernel = self.make_kernel()
        self.kernel_vmap = jax.vmap(self.kernel, in_axes = (0, 0, 0, None, None), out_axes=(0, 0, 0))
        self.update = self.make_update()
        self.update_vmap = jax.vmap(self.update, in_axes = (None, (0, 0, 0, 0, None, None)), out_axes=(0, 0, 0, 0, None, None))
        self.sampler = self.make_sampler()

        if self.jit == True:
            self.logpdf_vmap = jax.jit(self.logpdf_vmap)
            self.kernel = jax.jit(self.kernel)
            self.kernel_vmap = jax.jit(self.kernel_vmap)
            self.update = jax.jit(self.update)
            self.update_vmap = jax.jit(self.update_vmap)
            self.kernel(jax.random.PRNGKey(0), jnp.ones(n_dims), jnp.ones(1), data, self.params)
            # self.update(1, (jax.random.PRNGKey(0), jnp.ones(n_dims), jnp.ones(1), jnp.zeros((n_step, 1)), data, self.params))
        
        key = jax.random.split(jax.random.PRNGKey(0), n_chains)

        self.logpdf_vmap(jnp.ones((n_chains, n_dims)), data)
        self.kernel_vmap(key, jnp.ones((n_chains, n_dims)), jnp.ones((n_chains, 1)), data, self.params)
        self.update_vmap(1, (key, jnp.ones((n_chains, n_step, n_dims)), jnp.ones((n_chains, n_step, 1)),jnp.zeros((n_chains, n_step, 1)), data, self.params))
        

    @abstractmethod
    def make_kernel(self, return_aux = False) -> Callable:
        """
        Make the kernel of the sampler for one update
        """

    @abstractmethod
    def make_update(self) -> Callable:
        """
        Make the update function for multiple steps
        """

    @abstractmethod
    def make_sampler(self) -> Callable:
        """
        Make the sampler for multiple chains given initial positions
        """