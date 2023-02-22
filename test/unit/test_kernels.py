from flowMC.sampler.HMC import HMC
from flowMC.sampler.MALA import MALA, mala_sampler_autotune
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.utils.PRNG_keys import initialize_rng_keys
import jax
import jax.numpy as jnp

def log_posterior(x):
    return -0.5 * jnp.sum(x ** 2)

def test_HMC_deterministic():
    n_dim = 2
    n_chains = 1
    HMC_obj = HMC(log_posterior, True, {"step_size": 1,"n_leapfrog": 3, "inverse_metric": jnp.ones(n_dim)})
    HMC_kernel = HMC_obj.make_kernel()

    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

    initial_PE = jax.vmap(HMC_obj.potential)(initial_position)

    HMC_kernel = HMC_obj.make_kernel()

    result1 = (HMC_kernel(rng_key_set[0], initial_position[0], initial_PE[0], HMC_obj.params))
    result2 = (HMC_kernel(rng_key_set[0], initial_position[0], initial_PE[0], HMC_obj.params))

    assert result1==result2

def test_MALA_deterministic():
    n_dim = 2
    n_chains = 1
    MALA_obj = MALA(log_posterior, True, {"step_size": 1, "inverse_metric": jnp.ones(n_dim)})
    MALA_kernel = MALA_obj.make_kernel()

    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

    initial_PE = jax.vmap(MALA_obj.potential)(initial_position)

    MALA_kernel = MALA_obj.make_kernel()

    result1 = (MALA_kernel(rng_key_set[0], initial_position[0], initial_PE[0], MALA_obj.params))
    result2 = (MALA_kernel(rng_key_set[0], initial_position[0], initial_PE[0], MALA_obj.params))

    assert result1==result2

def test_Gaussian_random_walk_deterministic():
    assert 1


