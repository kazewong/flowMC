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
from flowMC.sampler.MALA import mala_kernel,mala_sampler


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

d_pdf = jax.grad(mixture_logpdf)

samples = 30000
chains = 100
precompiled = False

n_dim = 2
n_samples = samples
n_chains = chains
rng_key = jax.random.PRNGKey(42)

rng_keys = jax.random.split(rng_key, n_chains)  # (nchains,)
initial_position = jnp.zeros((n_dim, n_chains))  # (n_dim, n_chains)

run_mcmc = jax.vmap(mala_sampler, in_axes=(0, None, None, None, 1, None),
                    out_axes=0)
rng_key, positions, log_prob = run_mcmc(rng_keys, n_samples, mixture_logpdf, d_pdf, initial_position, 0.1)
assert positions.shape == (n_chains, n_samples, n_dim)
positions.block_until_ready()

flat_chain = np.concatenate(positions,axis=0)#[:,1000:],axis=0)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(flat_chain[:,0],flat_chain[:,1],s=0.1)
plt.show()