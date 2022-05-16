import jax
import jax.numpy as jnp
import numpy as np
from nfsampler.nfmodel.utils import sample_nf,train_flow
from nfsampler.sampler.NF_proposal import nf_metropolis_sampler
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers

def initialize_rng_keys(n_chains, seed=42):
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
    rng_key = jax.random.PRNGKey(seed)
    rng_key_init, rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,3)

    rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
    rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)
    
    return rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf


def sampling_loop(rng_keys_nf, rng_keys_mcmc, model, state, initial_position, local_sampler, likelihood, config, d_likelihood=None):

    """
    Sampling loop for both the global sampler and the local sampler.

    Args:
        rng_keys_nf (Device Array): RNG keys for the normalizing flow global sampler.
        rng_keys_mcmc (Device Array): RNG keys for the local sampler.
        d_likelihood ?
        TODO: likelihood vs posterior?
        TODO: nf_samples - sometime int, sometimes samples 

    """

    stepsize = config['stepsize']
    n_dim = config['n_dim']
    n_local_steps = config['n_local_steps']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    n_global_steps = config['n_global_steps']


    if d_likelihood is None:
        rng_keys_mcmc, positions, log_prob, local_acceptance = local_sampler(
            rng_keys_mcmc, n_local_steps, likelihood, initial_position, stepsize
            )
    else:
        rng_keys_mcmc, positions, log_prob, local_acceptance = local_sampler(
            rng_keys_mcmc, n_local_steps, likelihood, d_likelihood, initial_position, stepsize
            )

    flat_chain = positions.reshape(-1,n_dim)
    rng_keys_nf, state, loss_values = train_flow(rng_keys_nf, model, state, flat_chain,
                                    num_epochs, batch_size)
    likelihood_vec = jax.vmap(likelihood)
    rng_keys_nf, nf_chain, log_prob, log_prob_nf, global_acceptance = nf_metropolis_sampler(
        rng_keys_nf, n_global_steps, model, state.params , likelihood_vec,
        positions[:,-1]
        )

    positions = jnp.concatenate((positions,nf_chain),axis=1)

    return rng_keys_nf, rng_keys_mcmc, state, positions, local_acceptance, \
           global_acceptance, loss_values


def sample(rng_keys_nf, rng_keys_mcmc, sampling_loop, initial_position,
           nf_model, state, run_mcmc, likelihood, config, d_likelihood=None,
           n_nf_samples=10000):
    """

    Returns:
        chains (n_chains, n_steps, dim): Sampled positions.
        nf_samples (n_nf_samples, dim): Samples from learned NF.
        local_accs (n_chains, n_local_steps * n_loop): Table of acceptance.
        global_accs (n_chains, n_global_steps * n_loop): Table of acceptance.
    """
    n_loop = config['n_loop']
    last_step = initial_position
    chains = []
    local_accs = []
    global_accs = []
    loss_vals = []

    for i in range(n_loop):
        rng_keys_nf, rng_keys_mcmc, state, positions, local_acceptance, global_acceptance, loss_values = sampling_loop(
            rng_keys_nf, rng_keys_mcmc, nf_model, state, last_step, run_mcmc, likelihood, config, d_likelihood, 
            )
        last_step = positions[:,-1]
        chains.append(positions)
        local_accs.append(local_acceptance)
        global_accs.append(global_acceptance)
        loss_vals.append(loss_values)


    chains = jnp.concatenate(chains, axis=1)
    local_accs = jnp.stack(local_accs, axis=1).reshape(chains.shape[0], -1)
    global_accs = jnp.stack(global_accs, axis=1).reshape(chains.shape[0], -1)
    loss_vals = jnp.concatenate(jnp.array(loss_vals))

    nf_samples = sample_nf(nf_model, state.params, rng_keys_nf, n_nf_samples)
    return chains, nf_samples, local_accs, global_accs, loss_vals


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
    def __init__(self, rng_key_set, config, nf_model, local_sampler,
                 likelihood, d_likelihood=None):
        rng_key_init ,rng_keys_mcmc, rng_keys_nf, init_rng_keys_nf = rng_key_set
        self.config = config
        self.nf_model = nf_model
        params = nf_model.init(init_rng_keys_nf, jnp.ones((config['batch_size'],config['n_dim'])))['params']

        tx = optax.adam(config['learning_rate'], config['momentum'])
        self.state = train_state.TrainState.create(apply_fn=nf_model.apply,
                                                   params=params, tx=tx)
        self.local_sampler = local_sampler
        self.likelihood = likelihood
        self.d_likelihood = d_likelihood
        self.rng_keys_nf = rng_keys_nf
        self.rng_keys_mcmc = rng_keys_mcmc


    def sample(self, initial_position):
        chains, nf_samples, local_accs, global_accs, loss_vals = sample(self.rng_keys_nf, self.rng_keys_mcmc,
                                    sampling_loop, initial_position,
                                    self.nf_model, self.state, 
                                    self.local_sampler, self.likelihood,
                                    self.config, self.d_likelihood, 
                                    )
        return chains, nf_samples, local_accs, global_accs, loss_vals
