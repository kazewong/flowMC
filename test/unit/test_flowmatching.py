import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from flowMC.resource.model.flowmatching.base import FlowMatchingModel, Solver, Path, CondOTScheduler
from flowMC.resource.model.common import MLP


def test_flowmatching_model_training():
    n_features = 2
    n_hidden = 16
    key = jax.random.PRNGKey(0)

    # Dummy data
    n_samples = 100
    x0 = jax.random.normal(key, (n_samples, n_features))
    x1 = jax.random.normal(key, (n_samples, n_features)) * 0.5 + 1
    t = jax.random.uniform(key, (n_samples, 1))

    # Initialize model
    key, subkey = jax.random.split(key)
    mlp = MLP([n_features + 1, n_hidden, n_features], key=subkey)
    solver = Solver(mlp)
    path = Path(CondOTScheduler())
    fm = FlowMatchingModel(solver, path)

    # Train model
    key, subkey = jax.random.split(key)
    optim = optax.adam(1e-3)
    state = optim.init(eqx.filter(fm, eqx.is_array))
    _, fm, _, loss = fm.train(subkey, (x0, x1, t), optim, state, 10, 32)

    assert loss[-1] < loss[0]


def test_flowmatching_model_sampling():
    n_features = 2
    n_hidden = 16
    key = jax.random.PRNGKey(0)

    # Initialize model
    key, subkey = jax.random.split(key)
    mlp = MLP([n_features + 1, n_hidden, n_features], key=subkey)
    solver = Solver(mlp)
    path = Path(CondOTScheduler())
    fm = FlowMatchingModel(solver, path)

    # Sample from model
    key, subkey = jax.random.split(key)
    samples = fm.sample(subkey, 10)

    assert samples.shape == (10, n_features)
    assert isinstance(samples, jnp.ndarray)


def test_flowmatching_model_log_prob():
    n_features = 2
    n_hidden = 16
    key = jax.random.PRNGKey(0)

    # Initialize model
    key, subkey = jax.random.split(key)
    mlp = MLP([n_features + 1, n_hidden, n_features], key=subkey)
    solver = Solver(mlp)
    path = Path(CondOTScheduler())
    fm = FlowMatchingModel(solver, path)

    # Get log_prob
    key, subkey = jax.random.split(key)
    samples = fm.sample(subkey, 10)
    log_prob = jax.vmap(fm.log_prob)(samples)

    assert log_prob.shape == (10,)
    assert isinstance(log_prob, jnp.ndarray)


def test_flowmatching_model_save_load():
    n_features = 2
    n_hidden = 16
    key = jax.random.PRNGKey(0)

    # Initialize model
    key, subkey = jax.random.split(key)
    mlp = MLP([n_features + 1, n_hidden, n_features], key=subkey)
    solver = Solver(mlp)
    path = Path(CondOTScheduler())
    fm = FlowMatchingModel(solver, path)

    # Save and load model
    fm.save_model("test_model")

    key, subkey = jax.random.split(key)
    mlp2 = MLP([n_features + 1, n_hidden, n_features], key=subkey)
    solver2 = Solver(mlp2)
    path2 = Path(CondOTScheduler())
    fm2 = FlowMatchingModel(solver2, path2)

    fm2 = fm2.load_model("test_model")

    assert all(
        [
            jnp.allclose(p1, p2)
            for p1, p2 in zip(
                eqx.filter(fm, eqx.is_array), eqx.filter(fm2, eqx.is_array)
            )
        ]
    )
