"""
Working Neal Funnel.
We use a mapping to and from a normal distribution to efficiently sample from the Neal funnel.

When we go to high dimension, the product term seems to dominate, and most of the mass are at the funnel.
I wonder whether this is well appercimated by the community.
"""

import logging
from flowMC.nfmodel.realNVP import RealNVP
from flowMC.sampler.MALA import mala_sampler
import jax
import jax.numpy as jnp  # JAX NumPy
import numpy as np
from jax.scipy.stats import norm
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys


def neal_funnel(x):
    y_pdf = norm.logpdf(x[0], loc=0, scale=3)
    x_pdf = norm.logpdf(x[1:], loc=0, scale=jnp.exp(x[0] / 2))
    return y_pdf + jnp.sum(x_pdf)


d_neal_funnel = jax.grad(neal_funnel)

n_dim = 5
n_loop = 5
n_local_steps = 20
n_global_steps = 100
n_chains = 100
stepsize = 0.01
learning_rate = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 1000
logging = True

print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(
    rng_key_set[0], shape=(n_chains, n_dim)
)  # (n_dim, n_chains)


model = RealNVP(10, n_dim, 64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None), out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    model,
    run_mcmc,
    neal_funnel,
    d_likelihood=d_neal_funnel,
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
)

print("Sampling")

chains, nf_samples = nf_sampler.sample(initial_position)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.plot(chains[70, :, 0], chains[70, :, 1])
plt.show()
plt.close()

# Plot all chains
corner.corner(
    chains.reshape(-1, n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
)
