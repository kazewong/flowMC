
"""
Working Neal Funnel.
We use a mapping to and from a normal distribution to efficiently sample from the Neal funnel.

When we go to high dimension, the product term seems to dominate, and most of the mass are at the funnel.
I wonder whether this is well appercimated by the community.
"""

import argparse
from functools import partial
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from functools import partial
from jax import jit



@partial(jax.jit, static_argnums=(1, 4, 5))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob, transform_map = None, inverse_map = None):
    """Moves the chains by one step using the Random Walk Metropolis algorithm.
    Attributes
    ----------
    rng_key: jax.random.PRNGKey
      Key for the pseudo random number generator.
    logpdf: function
      Returns the log-probability of the model given a position.
    position: jnp.ndarray, shape (n_dims,)
      The starting position.
    log_prob: float
      The log probability at the starting position.
    Returns
    -------
    Tuple
        The next positions of the chains along with their log probability.
    """
    key1, key2 = jax.random.split(rng_key)
    move_proposal = jax.random.normal(key1, shape=position.shape)
    # if transform_map != None:
    #     move_proposal = transform_map(move_proposal)
    
    if transform_map != None:
        proposal = inverse_map(position) + move_proposal 
        proposal = transform_map(proposal)
        proposal_log_prob = logpdf(proposal)
    else:
        proposal = position + move_proposal
        proposal_log_prob = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


@partial(jax.jit, static_argnums=(1, 2, 4, 5))
def rw_metropolis_sampler(rng_key, n_samples, logpdf, initial_position, transform_map = None, inverse_map = None):

    def mh_update_sol2(i, state):
        key, positions, log_prob = state
        _, key = jax.random.split(key)
        new_position, new_log_prob = rw_metropolis_kernel(key, logpdf, positions[i-1], log_prob, transform_map, inverse_map)
        positions=positions.at[i].set(new_position)
        return (key, positions, new_log_prob)


    logp = logpdf(initial_position)
    all_positions = jnp.zeros((n_samples,)+initial_position.shape)
    initial_state = (rng_key,all_positions, logp)
    rng_key, all_positions, log_prob = jax.lax.fori_loop(1, n_samples, 
                                                 mh_update_sol2, 
                                                 initial_state)
    
    
    return all_positions


def neal_funnel(x):
    y_dist = partial(norm.logpdf, loc=0, scale=3)
    x_dist = partial(norm.logpdf, loc=0, scale=jnp.exp(x[0]/2))
    y_pdf = y_dist(x[0])
    x_pdf = x_dist(x[1:])
    return y_pdf + jnp.sum(x_pdf,axis=0)

def proposal_map(x):
    y = x[0]*3
    z = x[1:]*jnp.exp(y/2)
    return jnp.append(y,z)


def inverse_map(x):
    y = x[0]/3
    z = x[1:]/jnp.exp(x[0]/2)
    return jnp.append(y,z)

samples = 10000
chains = 100
precompiled = False

n_dim = 2
n_samples = samples
n_chains = chains
rng_key = jax.random.PRNGKey(42)

rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
initial_position = jnp.array(np.random.normal(size=(n_dim,n_chains)))  # (n_dim, n_chains)

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1, None, None),
                    out_axes=0)
raw_positions = run_mcmc(rng_keys, n_samples, neal_funnel, initial_position ,None , None)
transformed_positions = run_mcmc(rng_keys, n_samples, neal_funnel, initial_position, proposal_map, inverse_map)

assert raw_positions.shape == (n_chains, n_samples, n_dim)
raw_positions.block_until_ready()
assert transformed_positions.shape == (n_chains, n_samples, n_dim)
transformed_positions.block_until_ready()
