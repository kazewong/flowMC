import jax
import jax.numpy as jnp

from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.proposal.MALA import MALA
from flowMC.proposal.NF_proposal import NFProposal
from flowMC.strategy.optimization import optimization_Adam


def log_posterior(x, data={}):
    return -0.5 * jnp.sum(x**2)


class TestStrategies:
    def test_Adam_optimization(self):
        n_dim = 2
        n_chains = 20

        key = jax.random.PRNGKey(42)

        local_sampler = MALA(
            log_posterior,
            True,
            step_size=1,
        )

        key, subkey = jax.random.split(key)
        global_sampler = NFProposal(
            local_sampler.logpdf,
            jit=True,
            model=MaskedCouplingRQSpline(n_dim, 4, [32, 32], 4, subkey),
        )

        Adam_obj = optimization_Adam(n_steps=10000, learning_rate=1e-2)

        key, subkey = jax.random.split(key)
        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1 + 10

        key, subkey = jax.random.split(key)

        key, optimized_positions, local_sampler, global_sampler, data = Adam_obj(
            subkey, local_sampler, global_sampler, initial_position, {}
        )

        vmapped_logp = jax.vmap(log_posterior)

        assert vmapped_logp(optimized_positions).mean() > vmapped_logp(initial_position).mean()