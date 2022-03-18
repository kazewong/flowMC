from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
from nfsampler.utils import Sampler, initialize_rng_keys
from jax.scipy.special import logsumexp
import numpy as np  


from nfsampler.nfmodel.utils import *

def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distribution and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[:1] + jnp.array([-3., 3.])) / 0.8) ** 2
    term3 = -0.5 * ((x[1:2] + jnp.array([-3., 3.])) / 0.6) ** 2
    return -(term1 - logsumexp(term2) - logsumexp(term3))

d_dual_moon = jax.grad(dual_moon_pe)

n_dim = 5
n_samples = 20
nf_samples = 100
n_chains = 100
learning_rate = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 1000


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

nf_sampler = Sampler(rng_key_set, config, model, run_mcmc, dual_moon_pe, d_dual_moon)

print("Sampling")

chains, nf_samples = nf_sampler.sample(initial_position)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.plot(chains[70,:,0],chains[70,:,1])
plt.show()
plt.close()

# Plot all chains
corner.corner(chains.reshape(-1,n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
