import jax
import jax.numpy as jnp

# from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
# from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.buffers import Buffer
# from flowMC.strategy.optimization import optimization_Adam
from flowMC.strategy.take_steps import TakeSerialSteps

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

        n_chains = 5
        n_steps = 25
        n_dims = 2
        n_batch = 5

        test_position = Buffer("test_position", n_chains, n_steps, n_dims)
        test_log_prob = Buffer("test_log_prob", n_chains, n_steps, 1)
        test_acceptance = Buffer("test_acceptance", n_chains, n_steps, 1)
        kernel = MALA(1.0)

        resources = {
            "test_position": test_position,
            "test_log_prob": test_log_prob,
            "test_acceptance": test_acceptance,
            "MALA": kernel,
        }

        test_log_prob.update_buffer(log_posterior(jnp.zeros((n_chains, n_steps, 1))), n_steps)
        strategy = TakeSerialSteps(log_posterior, kernel, "test", n_batch)
        key = jax.random.PRNGKey(42)
        positions = jnp.array(test_position.buffer[:,0])

        for i in range(n_batch):
            key, subkey1, subkey2 = jax.random.split(key,3)
            _, resources, positions = strategy(rng_key=jax.random.split(subkey1, n_chains), resources=resources,
            initial_position=positions, data={})

            # print(test_acceptance.buffer[:,:,0])
            # print(test_log_prob.buffer[:,:,0])


        new_kernel = MALA(0.5)
        strategy.update_kernel(new_kernel)
        key, subkey1, subkey2 = jax.random.split(key,3)
        strategy.set_current_position(0)
        _, resources, positions = strategy(rng_key=jax.random.split(subkey1, n_chains), resources=resources,
        initial_position=positions, data={})