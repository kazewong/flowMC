import jax
import jax.numpy as jnp
import numpy as np
from nfsampler.nfmodel.utils import sample_nf,train_flow
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler


def initialize_rng_keys(n_chains,seed=42):
    rng_key = jax.random.PRNGKey(42)
    rng_key_init, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

    rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
    rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)
    
    return rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf


def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position, run_mcmc, likelihood, params, d_likelihood=None):
    stepsize = params['stepsize']
    n_dim = params['n_dim']
    n_samples = params['n_samples']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    nf_samples = params['nf_samples']
    if d_likelihood is None:
        rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, initial_position, stepsize)
    else:
        rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, likelihood, d_likelihood, initial_position, stepsize)
    flat_chain = positions.reshape(-1,n_dim)
    rng_keys_nf, state = train_flow(rng_keys_nf, model, state, flat_chain, num_epochs, batch_size)
    rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, nf_samples, model, state.params , likelihood, positions[:,-1])

    positions = jnp.concatenate((positions,nf_chain),axis=1)
    return rng_keys_nf, rng_keys_mcmc, state, positions


def sample(rng_keys_nf, rng_keys_mcmc, sampling_loop, initial_position, nf_model, state, run_mcmc, likelihood, params, d_likelihood=None):
    n_loop = params['n_loop']
    last_step = initial_position
    chains = []
    for i in range(n_loop):
        rng_keys_nf, rng_keys_mcmc, state, positions = sampling_loop(rng_keys_nf, rng_keys_mcmc, nf_model, state, last_step, run_mcmc, likelihood, params, d_likelihood)
        last_step = positions[:,-1].T
        chains.append(positions)
    chains = np.concatenate(chains,axis=1)
    nf_samples = sample_nf(nf_model, state.params, rng_keys_nf, 10000)
    return chains, nf_samples