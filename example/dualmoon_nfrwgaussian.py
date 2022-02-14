from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.Gaussian_random_walk import rw_metropolis_sampler
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
import numpy as np  

from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers

from nfsampler.nfmodel.utils import *

def dual_moon_pe(x):
    """
    Term 2 and 3 separate the distriubiotn and smear it along the first and second dimension
    """
    term1 = 0.5 * ((jnp.linalg.norm(x, axis=-1) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[..., :1] + jnp.array([-3., 3.])) / 0.8) ** 2
    term3 = -0.5 * ((x[..., 1:2] + jnp.array([-3., 3.])) / 0.6) ** 2
    return -(term1 - logsumexp(term2, axis=-1) - logsumexp(term3, axis=-1))

d_dual_moon = jax.grad(dual_moon_pe)

n_dim = 5
n_samples = 20
nf_samples = 100
n_chains = 100
learning_rate = 0.01
momentum = 0.9
num_epochs = 100
batch_size = 1000

print("Preparing RNG keys")
rng_key = jax.random.PRNGKey(42)
rng_key_init, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

print("Initializing MCMC model and normalizing flow model.")

initial_position = jax.random.normal(rng_key_init,shape=(n_dim, n_chains)) #(n_dim, n_chains)

#model = MaskedAutoregressiveFlow(n_dim,64,4)
model = RealNVP(10,n_dim,64, 1)
params = model.init(init_rng_keys_nf, jnp.ones((batch_size,n_dim)))['params']

run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 1, None),
                    out_axes=0)
# run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
#                     out_axes=0)


tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

print("Sampling")

def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position):
    #rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, dual_moon_pe, initial_position)
    rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, dual_moon_pe, d_dual_moon, initial_position, 0.01)
    flat_chain = positions.reshape(-1,n_dim)
    rng_keys_nf, state = train_flow(rng_key_nf, model, state, flat_chain, num_epochs, batch_size)
    rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, nf_samples, model, state.params , dual_moon_pe, positions[:,-1])

    positions = jnp.concatenate((positions,nf_chain),axis=1)
    return rng_keys_nf, rng_keys_mcmc, state, positions

last_step = initial_position
chains = []
for i in range(5):
	rng_keys_nf, rng_keys_mcmc, state, positions = sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, last_step)
	last_step = positions[:,-1].T
	chains.append(positions)
chains = np.concatenate(chains,axis=1)
nf_samples = sample_nf(model, state.params, rng_keys_nf, 10000)

import corner
import matplotlib.pyplot as plt

# Plot one chain to show the jump
plt.plot(chains[70,:,0],chains[70,:,1])
plt.show()
plt.close()

# Plot all chains
corner.corner(chains.reshape(-1,n_dim), labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"])
