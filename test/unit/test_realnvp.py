import jax
import jax.numpy as jnp
from flowMC.nfmodel.realNVP import RealNVP, AffineCoupling

def test_affine_coupling():
    n_features = 3
    n_hidden = 4
    mask = jnp.array([1, 0, 1])
    dt = 1
    x = jnp.array([[1, 2, 3], [4, 5, 6]])

    layer = AffineCoupling(n_features, n_hidden, mask, dt)
    layer.init(jax.random.PRNGKey(0), x)
    y, log_det = layer(x)

    assert y.shape == x.shape
    assert log_det.shape == (2,)

    y_inv, log_det_inv = layer.inverse(y)

    assert y_inv.shape == x.shape
    assert log_det_inv.shape == (2,)
    assert jnp.allclose(x, y_inv)
    assert jnp.allclose(log_det, -log_det_inv)


def test_realnvp():
    n_features = 3
    n_hidden = 4
    n_layer = 2
    dt = 1
    x = jnp.array([[1, 2, 3], [4, 5, 6]])

    rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0), 2)
    model = RealNVP(n_layer, n_features, n_hidden, dt)
    model_init = model.init(rng_subkey, jnp.ones((1, self.n_dim)))
    params = model_init["params"]
    variables = model_init["variables"]

    y, log_det = model.apply({"params": params, "variables": variables}, x)


    assert y.shape == x.shape
    assert log_det.shape == (2,)

    y_inv, log_det_inv = model.inverse(y)

    assert y_inv.shape == x.shape
    assert log_det_inv.shape == (2,)
    assert jnp.allclose(x, y_inv)
    assert jnp.allclose(log_det, -log_det_inv)

    rng_key = jax.random.PRNGKey(0)
    samples = model.sample(rng_key, 2)

    assert samples.shape == (2, 3)

    log_prob = model.log_prob(samples)

    assert log_prob.shape == (2,)

    assert jnp.allclose(log_prob, jax.scipy.stats.multivariate_normal.logpdf(
        samples, jnp.zeros(n_features), jnp.eye(n_features)
    ))