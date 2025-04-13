import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.local_kernel.HMC import HMC
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.states import State
from flowMC.strategy.take_steps import TakeSerialSteps
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF


def log_posterior(x, data=None):
    return -0.5 * jnp.sum(x**2)


n_dims = 2
logpdf = LogPDF(log_posterior, n_dims=2)


class TestHMC:

    def test_repr(self):
        HMC_obj = HMC(step_size=1, n_leapfrog=5)
        assert repr(HMC_obj) == "HMC with step size 1 and 5 leapfrog steps"

    def test_print_params(self, capsys):
        HMC_obj = HMC(step_size=1, n_leapfrog=5)
        HMC_obj.print_parameters()
        captured = capsys.readouterr()
        assert (
            captured.out
            == "HMC parameters:\nstep_size: 1\nn_leapfrog: 5\ncondition_matrix: 1\n"
        )

    def test_HMC_deterministic(self):
        n_chains = 1
        HMC_obj = HMC(
            step_size=1,
            n_leapfrog=5,
            condition_matrix=jnp.eye(n_dims),
        )

        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_PE = jax.vmap(lambda x, data: -log_posterior(x, data))(
            initial_position, None
        )

        # Test whether the HMC kernel is deterministic

        rng_key, subkey = jax.random.split(rng_key)
        result1 = HMC_obj.kernel(subkey, initial_position[0], initial_PE, logpdf, None)
        result2 = HMC_obj.kernel(subkey, initial_position[0], initial_PE, logpdf, None)

        assert jnp.allclose(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]

    def test_leapfrog_reversible(self):
        # Test whether the leapfrog kernel is reversible
        n_chains = 1
        HMC_obj = HMC(
            step_size=1,
            n_leapfrog=5,
            condition_matrix=jnp.eye(n_dims),
        )

        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_PE = jax.vmap(lambda x, data: -log_posterior(x, data))(
            initial_position, None
        )
        rng_key, subkey = jax.random.split(rng_key)
        key1, key2 = jax.random.split(subkey)

        def potential(x: Float[Array, " n_dims"], data: PyTree) -> Float[Array, "1"]:
            return -log_posterior(x, data)

        def kinetic(
            p: Float[Array, " n_dims"], metric: Float[Array, " n_dims"]
        ) -> Float[Array, "1"]:
            return 0.5 * (p**2 * metric).sum()

        leapfrog_kernel = jax.tree_util.Partial(
            HMC_obj.leapfrog_kernel, kinetic, potential
        )

        initial_momentum = (
            jax.random.normal(key1, shape=initial_position.shape)
            * jnp.ones(n_dims) ** -0.5
        )
        new_position, new_momentum = HMC_obj.leapfrog_step(
            leapfrog_kernel,
            initial_position,
            initial_momentum,
            None,
            jnp.eye(n_dims),
        )
        rev_position, rev_momentum = HMC_obj.leapfrog_step(
            leapfrog_kernel, new_position, -new_momentum, None, jnp.eye(n_dims)
        )

        assert jnp.allclose(rev_position, initial_position)
        assert jnp.allclose(initial_PE, -log_posterior(rev_position, None))

    def test_HMC_acceptance_rate(self):
        # Test acceptance rate goes to one when step size is small

        HMC_obj = HMC(
            step_size=0.0000001,
            n_leapfrog=5,
            condition_matrix=jnp.eye(n_dims),
        )

        n_chains = 100
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(HMC_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )

        assert result[2].all()

    def test_HMC_close_gaussian(self):
        n_chains = 1
        n_local_steps = 30000
        HMC_obj = HMC(
            step_size=0.1,
            n_leapfrog=5,
            condition_matrix=jnp.eye(n_dims),
        )
        positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
        log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
        acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)
        sampler_state = State(
            {
                "positions": "positions",
                "log_prob": "log_prob",
                "acceptance": "acceptance",
            },
            name="sampler_state",
        )
        resource = {
            "positions": positions,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "HMC": HMC_obj,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }

        # Defining strategy

        strategy = TakeSerialSteps(
            "logpdf",
            kernel_name="HMC",
            state_name="sampler_state",
            buffer_names=["positions", "log_prob", "acceptance"],
            n_steps=n_local_steps,
        )

        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        rng_key, subkey = jax.random.split(rng_key)
        result = strategy(subkey, resource, initial_position, {})[1]["positions"]

        assert isinstance(result, Buffer)

        assert jnp.isclose(jnp.mean(result.data), 0, atol=3e-2)
        assert jnp.isclose(jnp.var(result.data), 1, atol=3e-2)


class TestMALA:

    def test_repr(self):
        MALA_obj = MALA(step_size=1)
        assert repr(MALA_obj) == "MALA with step size 1"

    def test_print_params(self, capsys):
        MALA_obj = MALA(step_size=1)
        MALA_obj.print_parameters()
        captured = capsys.readouterr()
        assert captured.out == "MALA parameters:\nstep_size: 1\n"

    def test_MALA_deterministic(self):
        n_chains = 1
        MALA_obj = MALA(step_size=1)

        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = log_posterior(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        result1 = MALA_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )
        result2 = MALA_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )

        assert jnp.allclose(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]

    def test_MALA_acceptance_rate(self):
        # Test acceptance rate goes to one when the step size is small

        MALA_obj = MALA(step_size=0.00001)

        n_chains = 100
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(MALA_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )

        assert result[2].all()

    def test_MALA_close_gaussian(self):
        n_dims = 2
        n_chains = 1
        n_local_steps = 50000
        MALA_Sampler = MALA(step_size=1)
        positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
        log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
        acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)
        sampler_state = State(
            {
                "positions": "positions",
                "log_prob": "log_prob",
                "acceptance": "acceptance",
            },
            name="sampler_state",
        )

        resource = {
            "positions": positions,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "MALA": MALA_Sampler,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }

        # Defining strategy

        strategy = TakeSerialSteps(
            "logpdf",
            kernel_name="MALA",
            state_name="sampler_state",
            buffer_names=["positions", "log_prob", "acceptance"],
            n_steps=n_local_steps,
        )

        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        rng_key, subkey = jax.random.split(rng_key)
        result = strategy(subkey, resource, initial_position, {})[1]["positions"]

        assert isinstance(result, Buffer)

        assert jnp.isclose(jnp.mean(result.data), 0, atol=3e-2)
        assert jnp.isclose(jnp.var(result.data), 1, atol=3e-2)


class TestGRW:

    def test_repr(self):
        GRW_obj = GaussianRandomWalk(step_size=1)
        assert repr(GRW_obj) == "Gaussian Random Walk with step size 1"

    def test_print_params(self, capsys):
        GRW_obj = GaussianRandomWalk(step_size=1)
        GRW_obj.print_parameters()
        captured = capsys.readouterr()
        assert captured.out == "Gaussian Random Walk parameters:\nstep_size: 1\n"

    def test_Gaussian_random_walk_deterministic(self):
        n_chains = 1
        GRW_obj = GaussianRandomWalk(step_size=1)
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = log_posterior(initial_position)

        rng_key, subkey = jax.random.split(rng_key)
        result1 = GRW_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )
        result2 = GRW_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )

        assert jnp.allclose(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]

    def test_Gaussian_random_walk_acceptance_rate(self):
        # Test acceptance rate goes to one when the step size is small

        n_dim = 2
        GRW_obj = GaussianRandomWalk(step_size=0.00001)

        n_chains = 100
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(GRW_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )
        assert result[2].all()

    def test_Gaussian_random_walk_close_gaussian(self):
        n_chains = 1
        n_local_steps = 50000
        GRW_Sampler = GaussianRandomWalk(step_size=1)

        positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
        log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
        acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)

        sampler_state = State(
            {
                "positions": "positions",
                "log_prob": "log_prob",
                "acceptance": "acceptance",
            },
            name="sampler_state",
        )

        resource = {
            "positions": positions,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "GRW": GRW_Sampler,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }

        # Defining strategy

        strategy = TakeSerialSteps(
            "logpdf",
            kernel_name="GRW",
            state_name="sampler_state",
            buffer_names=["positions", "log_prob", "acceptance"],
            n_steps=n_local_steps,
        )

        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        rng_key, subkey = jax.random.split(rng_key)
        result = strategy(subkey, resource, initial_position, {})[1]["positions"]

        assert isinstance(result, Buffer)

        assert jnp.isclose(jnp.mean(result.data), 0, atol=3e-2)
        assert jnp.isclose(jnp.var(result.data), 1, atol=3e-2)
