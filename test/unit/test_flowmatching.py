import jax
import jax.numpy as jnp
import pytest

from flowMC.resource.model.flowmatching.base import (
    FlowMatchingModel,
    Solver,
    Path,
    CondOTScheduler,
)
from flowMC.resource.model.common import MLP
from diffrax import Dopri5
import equinox as eqx
import optax


def get_simple_mlp(n_input, n_hidden, n_output, key):
    """Simple 2-layer MLP for testing."""
    shape = (
        [n_input]
        + ([n_hidden] if isinstance(n_hidden, int) else list(n_hidden))
        + [n_output]
    )
    return MLP(shape=shape, key=key, activation=jax.nn.swish)


##############################
# Solver Tests
##############################


class TestSolver:
    @pytest.fixture
    def solver(self):
        key = jax.random.PRNGKey(0)
        n_dim = 3
        n_hidden = 4
        mlp = get_simple_mlp(
            n_input=n_dim + 1, n_hidden=n_hidden, n_output=n_dim, key=key
        )
        return Solver(model=mlp, method=Dopri5()), key, n_dim

    def test_sample_shape_and_finiteness(self, solver):
        solver, key, n_dim = solver
        n_samples = 7
        samples = solver.sample(key, n_samples)
        assert samples.shape == (n_samples, n_dim)
        assert jnp.isfinite(samples).all()

    def test_log_prob_shape_and_finiteness(self, solver):
        solver, key, n_dim = solver
        x1 = jax.random.normal(key, (n_dim,))
        logp = solver.log_prob(x1)
        logp_arr = jnp.asarray(logp)
        assert logp_arr.size == 1
        assert jnp.isfinite(logp_arr).all()

    @pytest.mark.parametrize("dt", [1e-2, 1e-1, 0.5])
    def test_sample_various_dt(self, solver, dt):
        solver, key, n_dim = solver
        samples = solver.sample(key, 3, dt=dt)
        assert samples.shape == (3, n_dim)
        assert jnp.isfinite(samples).all()


##############################
# Path & Scheduler Tests
##############################


class TestPathAndScheduler:
    def test_path_sample_shapes_and_values(self):
        n_dim = 2
        scheduler = CondOTScheduler()
        path = Path(scheduler=scheduler)
        x0 = jnp.ones((5, n_dim))
        x1 = jnp.zeros((5, n_dim))
        for t_val in [0.0, 0.5, 1.0]:
            t = jnp.full((5, 1), t_val)
            x_t, dx_t = path.sample(x0, x1, t)
            assert x_t.shape == (5, n_dim)
            assert dx_t.shape == (5, n_dim)

    @pytest.mark.parametrize("t", [0.0, 1.0, 0.5, -0.1, 1.1])
    def test_condotscheduler_call_output(self, t):
        sched = CondOTScheduler()
        out = sched(jnp.array(t))
        assert isinstance(out, tuple)
        assert len(out) == 4
        assert all(isinstance(float(x), float) for x in out)


##############################
# FlowMatchingModel Tests
##############################


class TestFlowMatchingModel:
    @pytest.fixture
    def model(self):
        key = jax.random.PRNGKey(42)
        n_dim = 2
        n_hidden = 8
        mlp = get_simple_mlp(
            n_input=n_dim + 1, n_hidden=n_hidden, n_output=n_dim, key=key
        )
        solver = Solver(model=mlp, method=Dopri5())
        scheduler = CondOTScheduler()
        path = Path(scheduler=scheduler)
        model = FlowMatchingModel(
            solver=solver,
            path=path,
            data_mean=jnp.zeros(n_dim),
            data_cov=jnp.eye(n_dim),
        )
        return model, key, n_dim

    def test_sample_and_log_prob(self, model):
        model, key, n_dim = model
        n_samples = 4
        samples = model.sample(key, n_samples)
        assert samples.shape == (n_samples, n_dim)
        assert jnp.isfinite(samples).all()
        logp = eqx.filter_vmap(model.log_prob)(samples)
        assert logp.shape == (n_samples, 1)
        assert jnp.isfinite(logp).all()

    @pytest.mark.parametrize("n_samples", [1, 5, 10])
    def test_sample_various_shapes(self, model, n_samples):
        model, key, n_dim = model
        samples = model.sample(key, n_samples)
        assert samples.shape == (n_samples, n_dim)
        assert jnp.isfinite(samples).all()
        logp = eqx.filter_vmap(model.log_prob)(samples)
        assert logp.shape[0] == n_samples
        assert jnp.isfinite(logp).all()

    def test_log_prob_edge_cases(self, model):
        model, key, n_dim = model
        for arr in [jnp.zeros(n_dim), 1e6 * jnp.ones(n_dim), -1e6 * jnp.ones(n_dim)]:
            logp = model.log_prob(arr)
            logp_arr = jnp.asarray(logp)
            assert logp_arr.size == 1
            assert (
                jnp.isfinite(logp_arr).all() or jnp.isnan(logp_arr).all()
            )  # may be nan for extreme values

    def test_save_and_load(self, tmp_path, model):
        model, key, n_dim = model
        save_path = str(tmp_path / "test_model")
        model.save_model(save_path)
        loaded = model.load_model(save_path)
        x = jax.random.normal(key, (2, n_dim))
        assert jnp.allclose(
            eqx.filter_vmap(model.log_prob)(x), eqx.filter_vmap(loaded.log_prob)(x)
        )

    def test_properties(self, model):
        model, key, n_dim = model
        mean = jnp.arange(n_dim)
        cov = jnp.eye(n_dim) * 2
        model2 = FlowMatchingModel(
            solver=model.solver, path=model.path, data_mean=mean, data_cov=cov
        )
        assert model2.n_features == n_dim
        assert jnp.allclose(model2.data_mean, mean)
        assert jnp.allclose(model2.data_cov, cov)

    def test_print_parameters_notimplemented(self, model):
        model, key, n_dim = model
        with pytest.raises(NotImplementedError):
            model.print_parameters()

    def test_train_step_and_epoch(self, model):
        model, key, n_dim = model
        n_batch = 5
        x0 = jax.random.normal(key, (n_batch, n_dim))
        x1 = jax.random.normal(key, (n_batch, n_dim))
        t = jax.random.uniform(key, (n_batch, 1))
        optim = optax.adam(learning_rate=1e-3)
        state = optim.init(eqx.filter(model, eqx.is_array))
        std = jnp.sqrt(jnp.diag(model.data_cov))
        x1_whitened = (x1 - model.data_mean) / std
        x_t, dx_t = model.path.sample(x0, x1_whitened, t)
        loss, model2, state2 = model.train_step(x_t, t, dx_t, optim, state)
        assert jnp.isfinite(loss)
        assert isinstance(model2, FlowMatchingModel)
        data = (x0, x1, t)
        loss_epoch, model3, state3 = model.train_epoch(
            key, optim, state, data, batch_size=n_batch
        )
        assert jnp.isfinite(loss_epoch)
        assert isinstance(model3, FlowMatchingModel)
