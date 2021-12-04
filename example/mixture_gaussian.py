"""
Augmented https://github.com/rlouf/blog-benchmark-rwmetropolis 
using stack overflow answer https://stackoverflow.com/questions/68303250/how-to-get-intermediate-results-in-jax-fori-loop-mechanism

The gaussian mixture model they used is wrongly constructed.
Instead of mixing n-D multivariate gaussians, they mixed n 1-D gaussians.
This is evident when they use norm.logpdf instead of multivariate_normal.logpdf. 
"""

import argparse
from functools import partial
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, multivariate_normal
from jax.scipy.special import logsumexp
from functools import partial


@partial(jax.jit, static_argnums=(1,))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob):
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
    move_proposal = jax.random.normal(key1, shape=position.shape) * 0.1
    proposal = position + move_proposal
    proposal_log_prob = logpdf(proposal)

    log_uniform = jnp.log(jax.random.uniform(key2))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = jnp.where(do_accept, proposal, position)
    log_prob = jnp.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


@partial(jax.jit, static_argnums=(1, 2))
def rw_metropolis_sampler(rng_key, n_samples, logpdf, initial_position):

    def mh_update_sol2(i, state):
        key, positions, log_prob = state
        _, key = jax.random.split(key)
        new_position, new_log_prob = rw_metropolis_kernel(key, logpdf, positions[i-1], log_prob)
        positions=positions.at[i].set(new_position)
        return (key, positions, new_log_prob)


    logp = logpdf(initial_position)
    all_positions = jnp.zeros((n_samples,)+initial_position.shape)
    initial_state = (rng_key,all_positions, logp)
    rng_key, all_positions, log_prob = jax.lax.fori_loop(1, n_samples, 
                                                 mh_update_sol2, 
                                                 initial_state)
    
    
    return rng_key,all_positions, log_prob


def mixture_logpdf(x):
    """Log probability distribution function of a gaussian mixture model.
    Attribute
    ---------
    x: jnp.ndarray (4,)
        Position at which to evaluate the probability density function.
    Returns
    -------
    float
        The value of the log probability density function at x.
    """
    cov = jnp.repeat(jnp.eye(2)[None,:],1,axis=0)
    mean = jnp.array([[0.0, 0.0]])
    dist_1 = partial(multivariate_normal.logpdf, mean=mean+5.0, cov=cov)
    dist_2 = partial(multivariate_normal.logpdf, mean=mean-5.0, cov=cov)
    # dist_3 = partial(norm.logpdf, loc=3.2, scale=5)
    # dist_4 = partial(norm.logpdf, loc=2.5, scale=2.8)
    log_probs = jnp.array([dist_1(x), dist_2(x)])#, dist_3(x), dist_4(x)])
    weights = jnp.array([0.3, 0.7])#, 0.1, 0.4])
    return logsumexp(jnp.log(weights) + log_probs)


samples = 30000
chains = 100
precompiled = False

n_dim = 2
n_samples = samples
n_chains = chains
rng_key = jax.random.PRNGKey(42)

rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
initial_position = jnp.zeros((n_dim, n_chains))  # (n_dim, n_chains)

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
                    out_axes=0)
positions = run_mcmc(rng_keys, n_samples, mixture_logpdf, initial_position)
assert positions.shape == (n_chains, n_samples, n_dim)
positions.block_until_ready()

flat_chain = np.concatenate(positions[:,1000:],axis=0)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(flat_chain[:,0],flat_chain[:,1],s=0.1)
plt.show()