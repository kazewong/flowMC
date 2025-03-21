import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.local_kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.local_kernel.HMC import HMC
from flowMC.resource.buffers import Buffer

# from flowMC.strategy.optimization import optimization_Adam
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.optimization import AdamOptimization
from flowMC.strategy.train_model import TrainModel


def log_posterior(x, data={}):
    return -0.5 * jnp.sum(x**2)


class TestOptimizationStrategies:
    n_dim = 2
    n_chains = 20
    n_steps = 100

    strategy = AdamOptimization(
        log_posterior,
        n_steps,
        learning_rate=5e-2,
        noise_level=0.0,
        bounds=jnp.array([[-jnp.inf, jnp.inf]]),
    )

    def test_Adam_optimization(self):
        key = jax.random.PRNGKey(42)

        key, subkey = jax.random.split(key)
        initial_position = (
            jax.random.normal(subkey, shape=(self.n_chains, self.n_dim)) * 1 + 10
        )

        _, _, optimized_position = self.strategy(key, {}, initial_position, {})

        assert optimized_position.shape == (self.n_chains, self.n_dim)
        assert jnp.all(
            jnp.mean(optimized_position, axis=1) < jnp.mean(initial_position, axis=1)
        )

    def test_standalone_optimize(self):
        key = jax.random.PRNGKey(42)

        key, subkey = jax.random.split(key)
        initial_position = (
            jax.random.normal(subkey, shape=(self.n_chains, self.n_dim)) * 1 + 10
        )

        def loss_fn(params: Float[Array, " n_dim"]) -> Float:
            return -log_posterior(params, {})

        rng_key, optimized_position = self.strategy.optimize(
            key, loss_fn, initial_position, {}
        )

        assert optimized_position.shape == (self.n_chains, self.n_dim)
        assert jnp.all(
            jnp.mean(optimized_position, axis=1) < jnp.mean(initial_position, axis=1)
        )


class TestStrategies:
    def test_take_local_step(self):
        n_chains = 5
        n_steps = 25
        n_dims = 2
        n_batch = 5

        test_position = Buffer("test_position", (n_chains, n_steps, n_dims), 1)
        test_log_prob = Buffer("test_log_prob", (n_chains, n_steps), 1)
        test_acceptance = Buffer("test_acceptance", (n_chains, n_steps), 1)

        mala_kernel = MALA(1.0)
        grw_kernel = GaussianRandomWalk(1.0)
        hmc_kernel = HMC(jnp.eye(n_dims), 0.1, 10)

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "MALA": mala_kernel,
            "GRW": grw_kernel,
            "HMC": hmc_kernel,
        }

        strategy = TakeSerialSteps(
            log_posterior,
            "MALA",
            ["test_position", "test_log_prob", "test_acceptance"],
            n_batch,
        )
        key = jax.random.PRNGKey(42)
        positions = test_position.data[:, 0]

        for i in range(n_batch):
            key, subkey1, subkey2 = jax.random.split(key, 3)
            _, resources, positions = strategy(
                rng_key=subkey1,
                resources=resources,
                initial_position=positions,
                data={},
            )

        key, subkey1, subkey2 = jax.random.split(key, 3)
        strategy.set_current_position(0)
        _, resources, positions = strategy(
            rng_key=subkey1,
            resources=resources,
            initial_position=positions,
            data={},
        )

        key, subkey1, subkey2 = jax.random.split(key, 3)
        strategy.kernel_name = "GRW"
        strategy.set_current_position(0)
        _, resources, positions = strategy(
            rng_key=subkey1,
            resources=resources,
            initial_position=positions,
            data={},
        )

        strategy.kernel_name = "HMC"
        _, resources, positions = strategy(
            rng_key=subkey1,
            resources=resources,
            initial_position=positions,
            data={},
        )


class TestNFStrategies:
    n_chains = 5
    n_steps = 25
    n_dims = 2
    n_batch = 5

    n_features = n_dims
    hidden_layes = [16, 16]
    n_layers = 3
    n_bins = 8

    def test_training(self):
        # TODO: Need to check for accuracy still
        rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0), 2)
        model = MaskedCouplingRQSpline(
            self.n_features,
            self.n_layers,
            self.hidden_layes,
            self.n_bins,
            jax.random.PRNGKey(10),
        )

        test_data = Buffer("test_data", (self.n_chains, self.n_steps, self.n_dims), 1)
        test_data.update_buffer(
            jax.random.normal(
                rng_subkey, shape=(self.n_chains, self.n_steps, self.n_dims)
            ),
        )
        optimizer = Optimizer(model)

        resources = {
            "test_data": test_data,
            "optimizer": optimizer,
            "model": model,
        }

        strategy = TrainModel(
            "model",
            "test_data",
            "optimizer",
            n_epochs=10,
            batch_size=self.n_chains * self.n_steps,
            n_max_examples=10000,
            thinning=1,
            verbose=True,
        )

        key = jax.random.PRNGKey(42)

        print(resources["model"].data_mean, resources["model"].data_cov)
        key, resources, positions = strategy(
            key,
            resources,
            jax.random.normal(key, shape=(self.n_chains, self.n_dims)),
            {},
        )
        assert isinstance(resources["model"], MaskedCouplingRQSpline)
        print(resources["model"].data_mean, resources["model"].data_cov)

    def test_take_NF_step(self):
        test_position = Buffer(
            "test_position", (self.n_chains, self.n_steps, self.n_dims), 1
        )
        test_log_prob = Buffer("test_log_prob", (self.n_chains, self.n_steps), 1)
        test_acceptance = Buffer("test_acceptance", (self.n_chains, self.n_steps), 1)

        model = MaskedCouplingRQSpline(
            self.n_features,
            self.n_layers,
            self.hidden_layes,
            self.n_bins,
            jax.random.PRNGKey(10),
        )

        proposal = NFProposal(model)

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "NFProposal": proposal,
        }

        def test_target(x, data={}):
            return model.log_prob(x)

        strategy = TakeGroupSteps(
            test_target,
            "NFProposal",
            ["test_position", "test_log_prob", "test_acceptance"],
            self.n_steps,
        )
        key = jax.random.PRNGKey(42)
        positions = test_position.data[:, 0]
        print(test_position.data[:, :, 0])
        strategy(
            rng_key=key,
            resources=resources,
            initial_position=positions,
            data={},
        )
        print(test_position.data[:, :, 0])

    def test_training_effect(self):
        Buffer("test_position", (self.n_chains, self.n_steps, self.n_dims), 1)
        Buffer("test_log_prob", (self.n_chains, self.n_steps), 1)
        Buffer("test_acceptance", (self.n_chains, self.n_steps), 1)

        model = MaskedCouplingRQSpline(
            self.n_features,
            self.n_layers,
            self.hidden_layes,
            self.n_bins,
            jax.random.PRNGKey(10),
        )

        NFProposal(model)
