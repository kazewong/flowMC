from flowMC.sampler.HMC import HMC
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.utils.PRNG_keys import initialize_rng_keys
import jax
import jax.numpy as jnp

def log_posterior(x, data=None):
    return -0.5 * jnp.sum(x ** 2)

class TestHMC:

    def test_HMC_deterministic(self):
        n_dim = 2
        n_chains = 1
        HMC_obj = HMC(log_posterior, True, {"step_size": 1,"n_leapfrog": 5, "inverse_metric": jnp.ones(n_dim)})

        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_PE = jax.vmap(HMC_obj.potential)(initial_position, None)

        HMC_kernel, leapfrog_kernel, leapfrog_step = HMC_obj.make_kernel(return_aux=True)

        # Test whether the HMC kernel is deterministic

        result1 = (HMC_kernel(rng_key_set[0], initial_position[0], initial_PE[0], None, HMC_obj.params))
        result2 = (HMC_kernel(rng_key_set[0], initial_position[0], initial_PE[0], None, HMC_obj.params))

        assert jnp.allclose(result1[0],result2[0])
        assert result1[1]==result2[1]
        assert result1[2]==result2[2]

    def test_leapfrog_reversible(self):
        # Test whether the leapfrog kernel is reversible
        n_dim = 2
        n_chains = 1
        HMC_obj = HMC(log_posterior, True, {"step_size": 1,"n_leapfrog": 5, "inverse_metric": jnp.ones(n_dim)})

        rng_key_set = initialize_rng_keys(n_chains, seed=42)
        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_PE = jax.vmap(HMC_obj.potential, in_axes=(0,None))(initial_position, None)

        HMC_kernel, leapfrog_kernel, leapfrog_step = HMC_obj.make_kernel(return_aux=True)
        key1, key2 = jax.random.split(rng_key_set[0])

        initial_momentum = jax.random.normal(key1, shape=initial_position.shape) * jnp.ones(n_dim)**-0.5
        new_position, new_momentum = leapfrog_step(initial_position, initial_momentum, None, HMC_obj.params)
        rev_position, rev_momentum = leapfrog_step(new_position, -new_momentum, None, HMC_obj.params)

        assert jnp.allclose(rev_position, initial_position)
        assert jnp.allclose(initial_PE, HMC_obj.potential(rev_position, None))

    def test_HMC_acceptance_rate(self):
        # Test acceptance rate goes to one when step size is small

        n_dim = 2
        HMC_obj = HMC(log_posterior, True, {"step_size": 0.00001,"n_leapfrog": 5, "inverse_metric": jnp.ones(n_dim)})
        HMC_kernel = HMC_obj.make_kernel()

        n_chains = 100
        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_PE = jax.vmap(HMC_obj.potential)(initial_position, None)

        result = jax.vmap(HMC_kernel, in_axes = (0, 0, 0, None, None), out_axes=(0, 0, 0))(rng_key_set[1], initial_position, initial_PE, None, HMC_obj.params)

        assert result[2].all()

    def test_HMC_close_gaussian(self):
        n_dim = 2
        n_chains = 1
        HMC_obj = HMC(log_posterior, True, {"step_size": 1,"n_leapfrog": 5, "inverse_metric": jnp.ones(n_dim)})

        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        HMC_obj.precompilation(n_chains, n_dim, 10000, None)
        HMC_sampler = HMC_obj.make_sampler()

        result = HMC_sampler(rng_key_set[1], 10000, initial_position, None)

        assert jnp.isclose(jnp.mean(result[1]),0,atol=1e-2)
        assert jnp.isclose(jnp.var(result[1]),1,atol=1e-2)

class TestMALA:

    def test_MALA_deterministic(self):
        n_dim = 2
        n_chains = 1
        MALA_obj = MALA(log_posterior, True, {"step_size": 1})
        MALA_kernel = MALA_obj.make_kernel()

        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_logp = log_posterior(initial_position, None)

        result1 = (MALA_kernel(rng_key_set[0], initial_position[0], initial_logp, MALA_obj.params))
        result2 = (MALA_kernel(rng_key_set[0], initial_position[0], initial_logp, MALA_obj.params))

        assert jnp.allclose(result1[0],result2[0])
        assert result1[1]==result2[1]
        assert result1[2]==result2[2]

    def test_MALA_acceptance_rate(self):
        # Test acceptance rate goes to one when the step size is small

        MALA_obj = MALA(log_posterior, True, {"step_size": 0.00001})
        MALA_kernel = MALA_obj.make_kernel()

        n_chains = 100
        n_dim = 2
        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        result = jax.vmap(MALA_kernel, in_axes = (0, 0, 0, None, None), out_axes=(0, 0, 0))(rng_key_set[1], initial_position, initial_logp, None, MALA_obj.params)

        assert result[2].all()

    def test_MALA_close_gaussian(self):
        n_dim = 2
        n_chains = 1
        MALA_obj = MALA(log_posterior, True, {"step_size": 1})

        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        MALA_obj.precompilation(n_chains, n_dim, 30000, None)

        MALA_sampler = MALA_obj.make_sampler()

        result = MALA_sampler(rng_key_set[1], 30000, initial_position, None)

        assert jnp.isclose(jnp.mean(result[1]),0,atol=1e-2)
        assert jnp.isclose(jnp.var(result[1]),1,atol=1e-2)

    
class TestGRW():

    def test_Gaussian_random_walk_deterministic(self):
        n_dim = 2
        n_chains = 1
        GRW_obj = GaussianRandomWalk(log_posterior, True, {"step_size": 1})
        GRW_kernel = GRW_obj.make_kernel()

        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_logp = log_posterior(initial_position)

        result1 = (GRW_kernel(rng_key_set[0], initial_position[0], initial_logp, GRW_obj.params))
        result2 = (GRW_kernel(rng_key_set[0], initial_position[0], initial_logp, GRW_obj.params))

        assert jnp.allclose(result1[0],result2[0])
        assert result1[1]==result2[1]
        assert result1[2]==result2[2]

    def test_Gaussian_random_walk_acceptance_rate(self):
        # Test acceptance rate goes to one when the step size is small

        n_dim = 2
        GRW_obj = GaussianRandomWalk(log_posterior, True, {"step_size": 0.00001})
        GRW_kernel = GRW_obj.make_kernel()

        n_chains = 100
        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position)

        result = jax.vmap(GRW_kernel, in_axes = (0, 0, 0, None, None), out_axes=(0, 0, 0))(rng_key_set[1], initial_position, initial_logp, None, GRW_obj.params)

        assert result[2].all()

    def test_Gaussian_random_walk_close_gaussian(self):
        n_dim = 2
        n_chains = 1
        GRW_obj = GaussianRandomWalk(log_posterior, True, {"step_size": 1})

        rng_key_set = initialize_rng_keys(n_chains, seed=42)

        initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1
        GRW_obj.precompilation(n_chains, n_dim, 30000, None)
        GRW_sampler = GRW_obj.make_sampler()

        result = GRW_sampler(rng_key_set[1], 30000, initial_position, None)

        assert jnp.isclose(jnp.mean(result[1]),0,atol=1e-2)
        assert jnp.isclose(jnp.var(result[1]),1,atol=1e-2)