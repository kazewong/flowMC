import jax
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.resource_strategy_bundle.RQSpline_MALA_PT import RQSpline_MALA_PT_Bundle


def logpdf(x, _):
    return -0.5 * (x**2).sum()


def test_rqspline_mala_bundle_initialization():
    rng_key = jax.random.PRNGKey(0)
    n_chains = 2
    n_dims = 3
    n_local_steps = 10
    n_global_steps = 5
    n_training_loops = 2
    n_production_loops = 1
    n_epochs = 3

    bundle = RQSpline_MALA_Bundle(
        rng_key=rng_key,
        n_chains=n_chains,
        n_dims=n_dims,
        logpdf=logpdf,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_training_loops=n_training_loops,
        n_production_loops=n_production_loops,
        n_epochs=n_epochs,
    )

    assert repr(bundle) == "RQSpline_MALA Bundle"


def test_rqspline_mala_pt_bundle_initialization():
    rng_key = jax.random.PRNGKey(0)
    n_chains = 2
    n_dims = 3
    n_local_steps = 10
    n_global_steps = 5
    n_training_loops = 2
    n_production_loops = 1
    n_epochs = 3

    bundle = RQSpline_MALA_PT_Bundle(
        rng_key=rng_key,
        n_chains=n_chains,
        n_dims=n_dims,
        logpdf=logpdf,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_training_loops=n_training_loops,
        n_production_loops=n_production_loops,
        n_epochs=n_epochs,
    )

    assert repr(bundle) == "RQSpline MALA PT Bundle"
