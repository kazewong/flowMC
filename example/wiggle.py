import numpy as np

import jax
import jax.numpy as jnp  # JAX NumPy
from jax.scipy.special import logsumexp

from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.MALA import mala_sampler
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys

from flowMC.nfmodel.utils import *

import matplotlib.pyplot as plt


def wiggle(x):
    """
    Wiggle-distribution 
    """
    mean = jnp.array([0, 6])
    centered_x = x[0] - mean - jnp.sin(5 * x[1] / 5)
    log_prob = - 0.5 * centered_x @ jnp.eye(2) @ centered_x.T
    log_prob -= 0.5 * (jnp.linalg.norm(x) - 5) ** 2 / 8
    return log_prob

d_wiggle = jax.grad(wiggle)

### Demo config

n_dim = 5
n_chains = 10
n_loop = 5
n_local_steps = 100
n_global_steps = 100
learning_rate = 0.1
momentum = 0.9
num_epochs = 5
batch_size = 50
stepsize = 0.01

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1


model = RealNVP(10, n_dim, 64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(n_dim, rng_key_set, model, run_mcmc,
                    wiggle,
                    d_likelihood=d_wiggle,
                    n_loop=n_loop,
                    n_local_steps=n_local_steps,
                    n_global_steps=n_global_steps,
                    n_chains=n_chains,
                    n_epochs=num_epochs,
                    n_nf_samples=100,
                    learning_rate=learning_rate,
                    momentum=momentum,
                    batch_size=batch_size,
                    stepsize=stepsize,
                    use_global=True,)
print("Sampling")

nf_sampler.sample(initial_position)

chains, log_prob, local_accs, global_accs, loss_vals = nf_sampler.get_sampler_state()