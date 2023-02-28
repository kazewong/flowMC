import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

def log_posterior(x):
    return -0.5 * jnp.sum(x ** 2)

n_dim = 5
n_chains = 10

rng_key_set = initialize_rng_keys(n_chains, seed=42)
initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
model = RQSpline(n_dim, 3, [64, 64], 8)
step_size = 1e-1
local_sampler = MALA(log_posterior, True, {"step_size": step_size})

nf_sampler = Sampler(n_dim,
                    rng_key_set,
                    local_sampler,
                    log_posterior,
                    model,
                    n_local_steps = 50,
                    n_global_steps = 50,
                    n_epochs = 30,
                    learning_rate = 1e-2,
                    batch_size = 1000,
                    n_chains = n_chains)

nf_sampler.sample(initial_position)
chains,log_prob,local_accs, global_accs = nf_sampler.get_sampler_state().values()
