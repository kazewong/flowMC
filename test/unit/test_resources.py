import jax
import jax.numpy as jnp
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF, Variable, TemperedPDF
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.strategy.take_steps import TakeSerialSteps


class TestLogPDF:

    n_dims = 5

    def posterior(self, x, data):
        return -jnp.sum(jnp.square(x - data["data"]))

    def test_value_and_grad(self):
        logpdf = LogPDF(
            self.posterior, [Variable("x_" + str(i), True) for i in range(self.n_dims)]
        )
        inputs = jnp.arange(self.n_dims).astype(jnp.float32)
        data = {"data": jnp.ones(self.n_dims)}
        values, grads = jax.value_and_grad(logpdf)(inputs, data)
        assert values == self.posterior(inputs, data)

    def test_resource(self):
        mala = MALA(1.0)
        logpdf = LogPDF(
            self.posterior, [Variable("x_" + str(i), True) for i in range(self.n_dims)]
        )
        rng_key = jax.random.PRNGKey(0)
        initial_position = jnp.zeros(self.n_dims)
        data = {"data": jnp.ones(self.n_dims)}
        resources = {
            "test_position": Buffer("test_position", (self.n_dims, 1), 1),
            "test_log_prob": Buffer("test_log_prob", (self.n_dims, 1), 1),
            "test_acceptance": Buffer("test_acceptance", (self.n_dims, 1), 1),
            "MALA": mala,
            "logpdf": logpdf,
        }
        stepper = TakeSerialSteps(
            "logpdf", "MALA", ["test_position", "test_log_prob", "test_acceptance"], 1
        )
        key, resources, positions = stepper(rng_key, resources, initial_position, data)

    def test_tempered_pdf(self):
        logpdf = TemperedPDF(
            self.posterior,
            lambda x, data: jnp.zeros(x.shape[0]),
            n_dims=self.n_dims,
            n_temps=5,
            max_temp=100,
        )
        inputs = jnp.arange(self.n_dims).astype(jnp.float32)
        data = {"data": jnp.ones(self.n_dims)}
        values = logpdf(inputs, data)
        assert values[0] == self.posterior(inputs, data)
        assert values.shape == (5,)


class TestBuffer:
    def test_buffer(self):
        buffer = Buffer("test", (10, 10), cursor_dim=1)
        assert buffer.name == "test"
        assert buffer.data.shape == (10, 10)
        assert buffer.cursor == 0
        assert buffer.cursor_dim == 1

    def test_update_buffer(self):
        buffer = Buffer("test", (10, 10), cursor_dim=0)
        buffer.update_buffer(jnp.ones((10, 10)))
        assert (buffer.data == jnp.ones((10, 10))).all()
