import jax
import jax.numpy as jnp
import numpy as np
from nfsampler.nfmodel.utils import sample_nf,train_flow
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers


def initialize_rng_keys(n_chains,seed=42):
    """
    Initialize the random number generator keys for the sampler.

    Args:
        n_chains (int): Number of chains for the local sampler.
        seed (int): Seed for the random number generator.


    Returns:
        rng_keys_init (Device Array): RNG keys for sampling initial position from prior.
        rng_keys_mcmc (Device Array): RNG keys for the local sampler.
        rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler.
        init_rng_keys_nf (Device Array): RNG keys for initializing wieght of the normalizing flow model.
    """
    rng_key = jax.random.PRNGKey(42)
    rng_key_init, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

    rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
    rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)
    
    return rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf


def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position, local_sampler, likelihood, params, d_likelihood=None):
    """
    Main worker kernel that do one sample cycle of the local and global sampler.

    Args:
        rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler.
        rng_keys_mcmc (Device Array): RNG keys for the local sampler.
        model (flax module): Normalizing flow model.
        state (flax state): Train state of the normalizing flow model.
        initial_position (Device Array): Initial position for the local sampler.
        local_sampler (function): Local sampler function.
        likelihood (function): Likelihood function.
        params (dict): Configuration parameters.
        d_likelihood (Device Array): Derivative of the likelihood function.
    
    Returns:
        rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler after the loop.
        rng_keys_mcmc (Device Array): RNG keys for the local sampler after the loop.
        state (flax state): Train state of the normalizing flow model after the loop.
        positions (Device Array): Walker positions after the loop.
    """
    stepsize = params['stepsize']
    n_dim = params['n_dim']
    n_samples = params['n_samples']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    nf_samples = params['nf_samples']
    if d_likelihood is None:
        rng_keys_mcmc, positions, log_prob, acceptance = local_sampler(rng_keys_mcmc, n_samples, likelihood, initial_position, stepsize)
    else:
        rng_keys_mcmc, positions, log_prob, acceptance = local_sampler(rng_keys_mcmc, n_samples, likelihood, d_likelihood, initial_position, stepsize)
    # if logging == True:

    flat_chain = positions.reshape(-1,n_dim)
    rng_keys_nf, state = train_flow(rng_keys_nf, model, state, flat_chain, num_epochs, batch_size)
    likelihood_vec = jax.vmap(likelihood)
    rng_keys_nf, nf_chain, log_prob, log_prob_nf = nf_metropolis_sampler(rng_keys_nf, nf_samples, model, state.params , likelihood_vec, positions[:,-1])

    positions = jnp.concatenate((positions,nf_chain),axis=1)
    return rng_keys_nf, rng_keys_mcmc, state, positions


def sample(rng_keys_nf, rng_keys_mcmc, sampling_loop, initial_position, nf_model, state, local_sampler, likelihood, params, d_likelihood=None):
    """
    Wrapper function that calls the sampling loop and produce the final chain.

    Args:
        rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler.
        rng_keys_mcmc (Device Array): RNG keys for the local sampler.
        sampling_loop (function): Sampling loop function.
        initial_position (Device Array): Initial position for the local sampler.
        nf_model (flax module): Normalizing flow model.
        state (flax state): Train state of the normalizing flow model.
        local_sampler (function): Local sampler function.
        likelihood (function): Likelihood function.
        params (dict): Configuration parameters.
        d_likelihood (Device Array): Derivative of the likelihood function.

    Returns:
        chain (Device Array): Sampled chain. (n_walkers, n_samples, n_dim)
        nf_sample (Device Array): Sampled normalizing flow chain.
    """
    n_loop = params['n_loop']
    last_step = initial_position
    chains = []
    for i in range(n_loop):
        rng_keys_nf, rng_keys_mcmc, state, positions = sampling_loop(rng_keys_nf, rng_keys_mcmc, nf_model, state, last_step, local_sampler, likelihood, params, d_likelihood)
        last_step = positions[:,-1]
        chains.append(positions)
    chains = np.concatenate(chains,axis=1)
    nf_samples = sample_nf(nf_model, state.params, rng_keys_nf, 10000)
    return chains, nf_samples


class Sampler:
    """
    Sampler class that host configuration parameters, NF model, and local sampler

    Args:
        rng_key_set (Device Array): RNG keys set generated using initialize_rng_keys.
        config (dict): Configuration parameters.
        nf_model (flax module): Normalizing flow model.
        local_sampler (function): Local sampler function.
        likelihood (function): Likelihood function.
        d_likelihood (Device Array): Derivative of the likelihood function.
    """
    def __init__(self, rng_key_set, config, nf_model, local_sampler, likelihood, d_likelihood=None):
        rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = rng_key_set
        self.config = config
        self.nf_model = nf_model
        params = nf_model.init(init_rng_keys_nf, jnp.ones((config['batch_size'],config['n_dim'])))['params']

        tx = optax.adam(config['learning_rate'], config['momentum'])
        self.state = train_state.TrainState.create(apply_fn=nf_model.apply, params=params, tx=tx)
        self.local_sampler = local_sampler
        self.likelihood = likelihood
        self.d_likelihood = d_likelihood
        self.rng_keys_nf = rng_keys_nf
        self.rng_keys_mcmc = rng_keys_mcmc


    def sample(self, initial_position):
        chains, nf_samples = sample(self.rng_keys_nf, self.rng_keys_mcmc, sampling_loop, initial_position, self.nf_model, self.state, self.local_sampler, self.likelihood, self.config, self.d_likelihood)
        return chains, nf_samples