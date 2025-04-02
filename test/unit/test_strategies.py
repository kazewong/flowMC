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
from flowMC.resource.logPDF import LogPDF, TemperedPDF

# from flowMC.strategy.optimization import optimization_Adam
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.optimization import AdamOptimization
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.parallel_tempering import ParallelTempering


def log_posterior(x, data={}):
    return -0.5 * jnp.sum((x - data["data"]) ** 2)


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

        _, _, optimized_position = self.strategy(
            key, {}, initial_position, {"data": jnp.arange(self.n_dim)}
        )

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
            return -log_posterior(params, {"data": jnp.arange(self.n_dim)})

        rng_key, optimized_position = self.strategy.optimize(
            key, loss_fn, initial_position, {"data": jnp.arange(self.n_dim)}
        )

        assert optimized_position.shape == (self.n_chains, self.n_dim)
        assert jnp.all(
            jnp.mean(optimized_position, axis=1) < jnp.mean(initial_position, axis=1)
        )


class TestLocalStep:
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

        logpdf = LogPDF(log_posterior, n_dims=n_dims)

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "logpdf": logpdf,
            "MALA": mala_kernel,
            "GRW": grw_kernel,
            "HMC": hmc_kernel,
        }

        strategy = TakeSerialSteps(
            "logpdf",
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
                data={"data": jnp.arange(n_dims)},
            )

        key, subkey1, subkey2 = jax.random.split(key, 3)
        strategy.set_current_position(0)
        _, resources, positions = strategy(
            rng_key=subkey1,
            resources=resources,
            initial_position=positions,
            data={"data": jnp.arange(n_dims)},
        )

        key, subkey1, subkey2 = jax.random.split(key, 3)
        strategy.kernel_name = "GRW"
        strategy.set_current_position(0)
        _, resources, positions = strategy(
            rng_key=subkey1,
            resources=resources,
            initial_position=positions,
            data={"data": jnp.arange(n_dims)},
        )

        strategy.kernel_name = "HMC"
        _, resources, positions = strategy(
            rng_key=subkey1,
            resources=resources,
            initial_position=positions,
            data={"data": jnp.arange(n_dims)},
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

    def initialize(self):
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

        return rng_key, resources

    def test_training(self):
        # TODO: Need to check for accuracy still
        rng_key, resources = self.initialize()

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
            {"data": jnp.arange(self.n_dims)},
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

        def test_target(x, data={}):
            return model.log_prob(x)

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "NFProposal": proposal,
            "logpdf": LogPDF(test_target, n_dims=self.n_dims),
        }

        strategy = TakeGroupSteps(
            "logpdf",
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
            data={"data": jnp.arange(self.n_dims)},
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


class TestTemperingStrategies:
    n_temps = 5
    n_dims = 3
    n_chains = 7
    n_steps = 4

    def initialize(self):
        mala = MALA(1.0)
        logpdf = TemperedPDF(
            log_posterior,
            lambda x, data: jnp.array(0.0),
            n_dims=self.n_dims,
            n_temps=self.n_temps,
        )

        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)

        initial_position = jax.random.normal(subkey, shape=(self.n_chains, self.n_dims))
        key, subkey = jax.random.split(key)
        tempered_initial_position = jax.random.normal(
            subkey, shape=(self.n_chains, self.n_temps - 1, self.n_dims)
        )
        tempered_positions = Buffer(
            "tempered_positions", (self.n_chains, self.n_temps - 1, self.n_dims), 2
        )
        tempered_positions.update_buffer(tempered_initial_position)
        temperatures = Buffer("temperatures", (self.n_temps,), 0)
        temperatures.update_buffer(jnp.arange(self.n_temps) + 1.0)

        resources = {
            "logpdf": logpdf,
            "MALA": mala,
            "tempered_positions": tempered_positions,
            "temperatures": temperatures,
        }

        parallel_tempering_strat = ParallelTempering(
            n_steps=self.n_steps,
            tempered_logpdf_name="logpdf",
            kernel_name="MALA",
            tempered_buffer_names=["tempered_positions", "temperatures"],
        )

        return key, resources, parallel_tempering_strat, initial_position

    def test_individual_step_body(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        mala = resources["MALA"]
        logpdf = resources["logpdf"]
        key, subkey = jax.random.split(key)
        position = initial_position[0]
        data = {"data": jnp.arange(self.n_dims)}

        log_prob = logpdf(position, data)
        carry, extras = parallel_tempering_strat._individual_step_body(
            mala, (key, position, log_prob, logpdf, jnp.array(1.0), data), None
        )

        # TODO: Add assertions

        assert carry[1].shape == (self.n_dims,)

    def test_individual_step(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        mala = resources["MALA"]
        logpdf = resources["logpdf"]
        initial_position = jnp.concatenate(
            [initial_position[:, None, :], resources["tempered_positions"].data],
            axis=1,
        )

        positions, log_probs, do_accept = parallel_tempering_strat._individal_step(
            mala,
            key,
            initial_position[0, 0],
            logpdf,
            jnp.array(1),
            {"data": jnp.arange(self.n_dims)},
        )

        # TODO: Add assertions
        assert positions.shape == (self.n_dims,)

    def test_ensemble_step(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        mala = resources["MALA"]
        logpdf = resources["logpdf"]
        initial_position = jnp.concatenate(
            [initial_position[:, None, :], resources["tempered_positions"].data],
            axis=1,
        )
        key, subkey = jax.random.split(key)
        positions, log_probs, do_accept = parallel_tempering_strat._ensemble_step(
            mala,
            subkey,
            initial_position[0],
            logpdf,
            jnp.arange(self.n_temps) + 1.0,
            {
                "data": jnp.arange(self.n_dims),
            },
        )

        # TODO: Add assertions

        keys = jax.random.split(key, self.n_chains)
        positions, log_probs, do_accept = jax.vmap(
            parallel_tempering_strat._ensemble_step,
            in_axes=(None, 0, 0, None, None, None),
        )(
            mala,
            keys,
            initial_position,
            logpdf,
            jnp.arange(self.n_temps) + 1.0,
            {
                "data": jnp.arange(self.n_dims),
            },
        )

        # TODO: Add assertions

    def test_exchange_step(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        logpdf = resources["logpdf"]
        temperatures = jnp.arange(self.n_temps) * 0.3 + 1
        data = {"data": jnp.arange(self.n_dims)}
        log_probs = jax.vmap(logpdf.tempered_log_pdf, in_axes=(None, 0, None))(
            temperatures,
            initial_position,
            data,
        )
        key = jax.random.split(key, self.n_chains)
        initial_position = jnp.concatenate(
            [initial_position[:, None, :], resources["tempered_positions"].data],
            axis=1,
        )
        positions, log_probs, do_accept = jax.jit(
            jax.vmap(parallel_tempering_strat._exchange, in_axes=(0, 0, 0, None, None))
        )(
            key,
            initial_position,
            logpdf,
            temperatures,
            {"data": jnp.arange(self.n_dims)},
        )

    def test_adapt_temperatures(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        temperatures = jnp.arange(self.n_temps) * 0.3 + 1
        parallel_tempering_strat._adapt_temperature(
            temperatures,
            jnp.ones((self.n_chains, self.n_temps)),
        )
        assert temperatures.shape == (self.n_temps,)

    def test_parallel_tempering(self):
        key, resources, parallel_tempering_strat, initial_position = self.initialize()
        key, subkey = jax.random.split(key)
        rng_key, resources, positions = parallel_tempering_strat(
            key,
            resources,
            initial_position,
            {
                "data": jnp.arange(self.n_dims),
            },
        )
        print(positions)
