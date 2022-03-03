
"""
Working Neal Funnel.
We use a mapping to and from a normal distribution to efficiently sample from the Neal funnel.

When we go to high dimension, the product term seems to dominate, and most of the mass are at the funnel.
I wonder whether this is well appercimated by the community.
"""

from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
import numpy as np  
from jax.scipy.stats import norm
from nfsampler.utils import Sampler, initialize_rng_keys

def neal_funnel(x):
    y_pdf = norm.logpdf(x[0],loc=0,scale=3)
    x_pdf = norm.logpdf(x[1:],loc=0,scale=jnp.exp(x[0]/2))
    return y_pdf + jnp.sum(x_pdf)

d_neal_funnel = jax.grad(neal_funnel)

config = {}
config['n_dim'] = 5
config['n_loop'] = 5
config['n_samples'] = 20
config['nf_samples'] = 100
config['n_chains'] = 100
config['learning_rate'] = 0.01
config['momentum'] = 0.9
config['num_epochs'] = 100
config['batch_size'] = 1000
config['stepsize'] = 0.01



print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(config['n_chains'],seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_set[0],shape=(config['n_chains'],config['n_dim'])) #(n_dim, n_chains)


model = RealNVP(10,config['n_dim'],64, 1)
run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 0, None),
                    out_axes=0)

print("Initializing sampler class")

nf_sampler = Sampler(rng_key_set, config, model, run_mcmc, neal_funnel, d_neal_funnel)

print("Sampling")

chains, nf_samples = nf_sampler.sample(initial_position)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.plot(chains[70,:,0],chains[70,:,1])
plt.show()
plt.close()

# Plot all chains
corner.corner(chains.reshape(-1,config['n_dim']), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
