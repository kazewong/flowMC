import jax
import jax.numpy as jnp

# from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
# from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.buffers import Buffer
from flowMC.resource.log_pdf import LogPDF
# from flowMC.strategy.optimization import optimization_Adam
from flowMC.strategy.take_steps import TakeLocalSteps

def log_posterior(x, data={}):
    return -0.5 * jnp.sum(x**2)


class TestStrategies:
    # def test_Adam_optimization(self):
    #     n_dim = 2
    #     n_chains = 20

    #     key = jax.random.PRNGKey(42)

    #     local_sampler = MALA(
    #         log_posterior,
    #         True,
    #         step_size=1,
    #     )

    #     key, subkey = jax.random.split(key)
    #     global_sampler = NFProposal(
    #         local_sampler.logpdf,
    #         jit=True,
    #         model=MaskedCouplingRQSpline(n_dim, 4, [32, 32], 4, subkey),
    #     )

    #     Adam_obj = optimization_Adam(n_steps=10000, learning_rate=1e-2)

    #     key, subkey = jax.random.split(key)
    #     initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1 + 10

    #     key, subkey = jax.random.split(key)

    #     key, optimized_positions, local_sampler, global_sampler, data = Adam_obj(
    #         subkey, local_sampler, global_sampler, initial_position, {}
    #     )

    #     vmapped_logp = jax.vmap(log_posterior)

    #     assert vmapped_logp(optimized_positions).mean() > vmapped_logp(initial_position).mean()

    def test_take_local_MALA_step(self):
        PDF_resource = LogPDF(log_posterior)
        buffer_resource = Buffer("test_buffer", 5, 10, 2)
        kernel = MALA(1.0)

        resources = {
            "LogPDF": PDF_resource,
            "test_buffer": buffer_resource,
            "MALA": kernel,
        }

        strategy = TakeLocalSteps("MALA", "test_buffer", 3, 1)
        strategy(rng_key=jax.random.split(jax.random.PRNGKey(42),5), resources=resources, initial_position=jax.random.normal(jax.random.PRNGKey(42), shape=(5, 2)), data={})

        strategy(rng_key=jax.random.split(jax.random.PRNGKey(42),5), resources=resources, initial_position=jax.random.normal(jax.random.PRNGKey(42), shape=(5, 2)), data={})