import jax
import jax.numpy as jnp
from flowMC.sampler.MALA import mala_kernel
from flowMC.sampler.Gaussian_random_walk import rw_metropolis_kernel


def test_mala_kernel():
    rng_key = jax.random.PRNGKey(0)
    logpdf = lambda x: x**2
    d_logpdf = lambda x: 2 * x
    position = jnp.array([1.0, 2.0])
    log_prob = logpdf(position)
    dt = 0.1
    position, log_prob, do_accept = mala_kernel(
        rng_key, logpdf, d_logpdf, position, log_prob, dt
    )


def test_rw_metropolis_kernel():
    rng_key = jax.random.PRNGKey(0)
    logpdf = lambda x: x**2
    d_logpdf = lambda x: 2 * x
    position = jnp.array([1.0, 2.0])
    log_prob = logpdf(position)
    dt = 0.1
    position, log_prob, do_accept = rw_metropolis_kernel(
        rng_key, logpdf, d_logpdf, position, log_prob, dt
    )
