import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import make_mala_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

def log_prob(x):
    return -0.5 * jnp.sum(x ** 2)

n_dim = 5
n_chains = 10

rng_key_set = initialize_rng_keys(n_chains, seed=42)
initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
model = RQSpline(n_dim, 3, [64, 64], 8)
local_sampler_caller = lambda x: make_mala_sampler(x, jit=True)
sampler_params = {'dt': 1e-1}

nf_sampler = Sampler(n_dim, rng_key_set, local_sampler_caller, sampler_params, log_prob,
                    model,
                    n_chains=n_chains)

nf_sampler.sample(initial_position)

