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

config = {}
config['n_dim'] = 5
config['n_loop'] = 3
config['n_local_steps'] = 20
config['n_global_steps'] = 3
config['n_chains'] = 4
config['learning_rate'] = 0.01
config['momentum'] = 0.9
config['num_epochs'] = 1
config['batch_size'] = 10  # error if larger than combination of params above
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

chains, nf_samples, local_accs, global_accs = nf_sampler.sample(initial_position)

print('chains shape: ', chains.shape, 'local_accs shape: ', local_accs.shape, 'global_accs shape: ', global_accs.shape)

chains = np.array(chains)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.figure()
plt.plot(chains[:,10,0],chains[:,10,1])

plt.close()

# Plot all chains
figure = corner.corner(chains.reshape(-1,config['n_dim']), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
figure.set_size_inches(7, 7)
figure.suptitle('Visualize samples')

plt.show(block=False)
