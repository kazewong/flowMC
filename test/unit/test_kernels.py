from flowMC.sampler.HMC import HMC
from flowMC.sampler.MALA import MALA, mala_sampler_autotune
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.utils.PRNG_keys import initialize_rng_keys
import jax
import jax.numpy as jnp

def log_posterior(x):
    return -0.5 * jnp.sum(x ** 2)

def test_HMC():
    n_dim = 2
    n_chains = 1
    HMC_obj = HMC(log_posterior, True, {"step_size": 1,"n_leapfrog": 5, "inverse_metric": jnp.ones(n_dim)})

    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
    initial_PE = jax.vmap(HMC_obj.potential)(initial_position)

    HMC_kernel, leapfrog_kernel, leapfrog_step = HMC_obj.make_kernel(return_aux=True)

    # Test whether the HMC kernel is deterministic

    result1 = (HMC_kernel(rng_key_set[0], initial_position[0], initial_PE[0], HMC_obj.params))
    result2 = (HMC_kernel(rng_key_set[0], initial_position[0], initial_PE[0], HMC_obj.params))

    assert jnp.allclose(result1[0],result2[0])
    assert result1[1]==result2[1]
    assert result1[2]==result2[2]

    # Test whether the leapfrog kernel is reversible

    key1, key2 = jax.random.split(rng_key_set[0])
    initial_momentum = jax.random.normal(key1, shape=initial_position.shape) * jnp.ones(n_dim)**-0.5

    new_position, new_momentum = leapfrog_step(initial_position, initial_momentum, HMC_obj.params)
    rev_position, rev_momentum = leapfrog_step(new_position, -new_momentum, HMC_obj.params)

    assert jnp.allclose(rev_position, initial_position)
    assert jnp.allclose(initial_PE, HMC_obj.potential(rev_position))

    # Test acceptance rate goes to one when step size is small

    HMC_obj = HMC(log_posterior, True, {"step_size": 0.00001,"n_leapfrog": 5, "inverse_metric": jnp.ones(n_dim)})
    HMC_kernel = HMC_obj.make_kernel()

    n_chains = 100
    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
    initial_PE = jax.vmap(HMC_obj.potential)(initial_position)

    result = jax.vmap(HMC_kernel, in_axes = (0, 0, 0, None), out_axes=(0, 0, 0))(rng_key_set[1], initial_position, initial_PE, HMC_obj.params)

    assert result[2].all()

def test_MALA_deterministic():
    n_dim = 2
    n_chains = 1
    MALA_obj = MALA(log_posterior, True, {"step_size": 1})
    MALA_kernel = MALA_obj.make_kernel()

    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

    initial_logp = log_posterior(initial_position)

    MALA_kernel = MALA_obj.make_kernel()

    result1 = (MALA_kernel(rng_key_set[0], initial_position[0], initial_logp, MALA_obj.params))
    result2 = (MALA_kernel(rng_key_set[0], initial_position[0], initial_logp, MALA_obj.params))

    assert jnp.allclose(result1[0],result2[0])
    assert result1[1]==result2[1]
    assert result1[2]==result2[2]

def test_Gaussian_random_walk_deterministic():
    n_dim = 2
    n_chains = 1
    GRW_obj = GaussianRandomWalk(log_posterior, True, {"step_size": 1})
    GRW_kernel = GRW_obj.make_kernel()

    rng_key_set = initialize_rng_keys(n_chains, seed=42)

    initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

    initial_logp = log_posterior(initial_position)

    GRW_kernel = GRW_obj.make_kernel()

    result1 = (GRW_kernel(rng_key_set[0], initial_position[0], initial_logp, GRW_obj.params))
    result2 = (GRW_kernel(rng_key_set[0], initial_position[0], initial_logp, GRW_obj.params))

    assert jnp.allclose(result1[0],result2[0])
    assert result1[1]==result2[1]
    assert result1[2]==result2[2]
