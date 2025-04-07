import jax
import jax.numpy as jnp
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF, Variable, TemperedPDF
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.states import State
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
        sampler_state = State(
            {
                "target_positions": "test_positions",
                "target_log_probs": "test_log_probs",
                "target_acceptances": "test_acceptances",
            },
            name="sampler_state",
        )
        resources = {
            "test_positions": Buffer("test_positions", (self.n_dims, 1), 1),
            "test_log_probs": Buffer("test_log_probs", (self.n_dims, 1), 1),
            "test_acceptances": Buffer("test_acceptances", (self.n_dims, 1), 1),
            "MALA": mala,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }
        stepper = TakeSerialSteps(
            "logpdf",
            "MALA",
            "sampler_state",
            ["target_positions", "target_log_probs", "target_acceptances"],
            1,
        )
        key, resources, positions = stepper(rng_key, resources, initial_position, data)

    def test_tempered_pdf(self):
        n_temps = 5
        logpdf = TemperedPDF(
            self.posterior,
            lambda x, data: jnp.zeros(1),
            n_dims=self.n_dims,
            n_temps=n_temps,
            max_temp=100,
        )
        inputs = jnp.ones((n_temps, self.n_dims)).astype(jnp.float32)
        data = {"data": jnp.ones(self.n_dims)}
        temperatures = jnp.arange(n_temps) + 1.0
        values = jax.vmap(logpdf.tempered_log_pdf, in_axes=(0, 0, None))(
            temperatures, inputs, data
        )
        assert (
            values[:, 0] == jax.vmap(self.posterior, in_axes=(0, None))(inputs, data)
        ).all()
        assert values.shape == (5, 1)


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
