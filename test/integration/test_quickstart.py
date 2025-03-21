import jax
import jax.numpy as jnp
from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundles import RQSpline_MALA_Bundle


def log_posterior(x, data: dict):
    return -0.5 * jnp.sum((x - data["data"]) ** 2)


n_dims = 2
n_local_steps = 10
n_global_steps = 10
n_training_loops = 5
n_production_loops = 5
n_epochs = 10
n_chains = 10
rq_spline_hidden_units = [64, 64]
rq_spline_n_bins = 8
rq_spline_n_layers = 3
data = {"data": jnp.arange(n_dims).astype(jnp.float32)}

rng_key = jax.random.PRNGKey(42)
rng_key, subkey = jax.random.split(rng_key)
initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

rng_key, subkey = jax.random.split(rng_key)
bundle = RQSpline_MALA_Bundle(
    subkey,
    n_chains,
    n_dims,
    log_posterior,
    n_local_steps,
    n_global_steps,
    n_training_loops,
    n_production_loops,
    n_epochs,
    rq_spline_hidden_units=rq_spline_hidden_units,
    rq_spline_n_bins=rq_spline_n_bins,
    rq_spline_n_layers=rq_spline_n_layers,
    verbose=True,
)

nf_sampler = Sampler(
    n_dims,
    n_chains,
    rng_key,
    resource_strategy_bundles=bundle,
)

nf_sampler.sample(initial_position, data)
