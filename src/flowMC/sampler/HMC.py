from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm

class HMC:
    
    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        self.logpdf = logpdf
        self.jit = jit

        self.potential = lambda x: -self.logpdf(x)
        self.grad_potential = jax.grad(self.potential)
        
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

        self.kinetic = lambda p: 0.5*(p**2 * self.inverse_metric).sum()
        self.grad_kinetic = jax.grad(self.kinetic)


    def make_hmc_kernel(self, return_aux = False) -> Callable:
        """

        Making HMC kernel for a single step

        """

        def leapfrog_kernal(carry, data):
            position, momentum = carry
            position = position + self.step_size * self.grad_kinetic(momentum)
            momentum = momentum - self.step_size * self.grad_potential(position)
            return position, momentum


        def leapfrog_step(position, momentum):
            momentum = momentum - 0.5 * self.step_size * self.grad_potential(position)
            (position, momentum), _ = jax.lax.scan(leapfrog_kernal, (position, momentum), jnp.arange(self.n_leapfrog-1))
            position = position + self.step_size * self.grad_kinetic(momentum)
            momentum = momentum - 0.5*self.step_size * self.grad_potential(position)
            return position, momentum


        def hmc_kernel(rng_key, position, H):
            """
            Args:
            rng_key (n_chains, 2): random key
            position (n_chains, n_dim): current position
            H (n_chains, ): Hamiltonian of the current position
            """
            momentum = jax.random.normal(rng_key, shape=position.shape) * self.inverse_metric
            position, momentum = leapfrog_step(position, momentum)

        
        if return_aux == False:
            return hmc_kernel
        else:
            return hmc_kernel, leapfrog_kernal, leapfrog_step
            

    def make_hmc_update(self) -> Callable:
        """
        
        Making HMC update function for multiple steps

        """

        hmc_kernel = self.make_hmc_kernel(self.logpdf)
        def hmc_update():
            pass

    def make_hmc_sampler(self) -> Callable:
        hmc_update, lp = self.make_hmc_update(self.logpdf)

        if self.jit:
            hmc_update = jax.jit(hmc_update)
            lp = jax.jit(lp)

        hmc_update = jax.vmap(hmc_update, in_axes = (None, (0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, None))
        lp = jax.vmap(lp)

        def hmc_sampler(rng_key, n_steps, initial_position, sampler_params = {}):
            pass

    def make_leapforg_kernel(self):
        def leapfrog_kernal(carry, data):
            position, momentum = carry
            position = position + self.step_size * self.grad_kinetic(momentum)
            momentum = momentum - self.step_size * self.grad_potential(position)
            return position, momentum
        return leapfrog_kernal





from tqdm import tqdm
from functools import partialmethod
