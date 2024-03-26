import jax
import jax.numpy as jnp
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.nfmodel.utils import *


def log_posterior(x, data: dict):
    return -0.5 * jnp.sum((x - data['data']) ** 2)


data = {'data':jnp.arange(5)}

n_dim = 5
n_chains = 10

rng_key = jax.random.PRNGKey(42)
rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1
rng_key, subkey = jax.random.split(rng_key)
model = MaskedCouplingRQSpline(n_dim, 3, [64, 64], 8, subkey)
step_size = 1e-1
local_sampler = MALA(log_posterior, True, {"step_size": step_size})

nf_sampler = Sampler(
    n_dim,
    rng_key,
    data,
    local_sampler,
    model,
    n_local_steps=50,
    n_global_steps=50,
    n_epochs=30,
    learning_rate=1e-2,
    batch_size=1000,
    n_chains=n_chains,
)

# nf_sampler.sample(initial_position, data)
# chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()
