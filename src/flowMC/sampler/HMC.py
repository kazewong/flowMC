from typing import Callable
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from tqdm import tqdm

def make_hmc_kernel(logpdf: Callable) -> Callable:
    """

    Making HMC kernel for a single step

    """

    def body(carry, this_key):
        pass

    def hmc_kernel(rng_key, position, log_prob, params):
        pass

def make_hmc_update(logpdf: Callable) -> Callable:
    """
    
    Making HMC update function for multiple steps

    """

    hmc_kernel = make_hmc_kernel(logpdf)
    def hmc_update():
        pass

def make_hmc_sampler(logpdf: Callable, jit: bool=False) -> Callable:
    hmc_update, lp = make_hmc_update(logpdf)

    if jit:
        hmc_update = jax.jit(hmc_update)
        lp = jax.jit(lp)

    hmc_update = jax.vmap(hmc_update, in_axes = (None, (0, 0, 0, 0, None)), out_axes=(0, 0, 0, 0, None))
    lp = jax.vmap(lp)

    def hmc_sampler(rng_key, n_steps, initial_position, sampler_params = {}):
        pass

from tqdm import tqdm
from functools import partialmethod
