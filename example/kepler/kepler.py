from re import S
import jax
import jax.numpy as jnp                # JAX NumPy
from jax.scipy.special import logsumexp
import numpy as np  
import optax 
from flax.training import train_state  # Useful dataclass to keep train state
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from nfsampler.nfmodel.realNVP import RealNVP
from nfsampler.sampler.MALA import mala_sampler
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
from nfsampler.nfmodel.utils import *

from utils import rv_model, log_likelihood, log_prior, sample_prior
from nfsampler.utils import Sampler, initialize_rng_keys


jax.config.update("jax_enable_x64", True)

## Generate probelm
true_params = jnp.array([
    12.0, # v0
    np.log(0.5), # log_s2
    np.log(14.5), # log_P
    np.log(2.3), # log_k
    np.sin(1.5), # phi
    np.cos(1.5),
    0.4, # ecc
    np.sin(-0.7), # w
    np.cos(-0.7)
])

random = np.random.default_rng(12345)
t = np.sort(random.uniform(0, 100, 50))
rv_err = 0.3
rv_obs = rv_model(true_params, t) + random.normal(0, rv_err, len(t))

plt.plot(t, rv_obs, ".k")
x = np.linspace(0, 100, 500)
plt.plot(x, rv_model(true_params, x), "C0")
plt.show(block=False)

prior_kwargs = {
    'ecc_alpha': 2, 'ecc_beta': 2,
    'log_k_mean': 1, 'log_k_var': 1,
    'v0_mean': 10, 'v0_var': 2,
    'log_period_mean': 2, 'log_period_var': 1,
    'log_s2_mean': 0, 'log_s2_var': 1,
}

#log_likelihood = jax.vmap(log_likelihood,in_axes=(0,None,None,None))


## Setting up sampling -- takes one input at the time.
def log_posterior(x):
    return  log_likelihood(x.T, t, rv_err, rv_obs) + log_prior(x,**prior_kwargs)

d_log_posterior = jax.grad(log_posterior)


config = {}
config['n_dim'] = 9
config['n_loop'] = 10
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

nf_sampler = Sampler(rng_key_set, config, model, run_mcmc, log_posterior, d_log_posterior)

print("Sampling")

chains, nf_samples = nf_sampler.sample(initial_position)
