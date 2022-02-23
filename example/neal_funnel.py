
"""
Working Neal Funnel.
We use a mapping to and from a normal distribution to efficiently sample from the Neal funnel.

When we go to high dimension, the product term seems to dominate, and most of the mass are at the funnel.
I wonder whether this is well appercimated by the community.
"""

from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.Gaussian_random_walk import rw_metropolis_sampler
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  

from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers
from functools import partial
from jax.scipy.stats import norm

from nfsampler.nfmodel.utils import *
from nfsampler.utils import *

def neal_funnel(x):
    # y_dist = partial(norm.logpdf, loc=0, scale=3)
    # x_dist = partial(norm.logpdf, loc=0, scale=jnp.exp(x[0]/2))
    x = x.T
    y_pdf = norm.logpdf(x[0],loc=0,scale=3)
    x_pdf = norm.logpdf(x[1:],loc=0,scale=jnp.exp(x[0]/2))
    return y_pdf + jnp.sum(x_pdf,axis=0)

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
rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = initialize_rng_keys(config['n_chains'],seed=42)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_init,shape=(config['n_dim'], config['n_chains'])) #(n_dim, n_chains)

model = RealNVP(10,config['n_dim'],64, 1)
params = model.init(init_rng_keys_nf, jnp.ones((config['batch_size'],config['n_dim'])))['params']

run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 1, None),
                    out_axes=0)

tx = optax.adam(config['learning_rate'], config['momentum'])
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

print("Sampling")

sampling_step = sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position, run_mcmc, neal_funnel, config, d_likelihood=d_neal_funnel)

chains, nf_samples = sample(rng_keys_nf, rng_keys_mcmc, sampling_loop, initial_position, model, state, run_mcmc, neal_funnel, config, d_likelihood=d_neal_funnel)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.plot(chains[70,:,0],chains[70,:,1])
plt.show()
plt.close()

# Plot all chains
corner.corner(chains.reshape(-1,config['n_dim']), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
