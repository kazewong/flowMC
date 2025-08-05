import jax
import jax.numpy as jnp

from flowMC.resource.nf_model.realNVP import AffineCoupling, RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline


def test_affine_coupling_forward_and_inverse():
    n_features = 2
    n_hidden = 4
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.where(jnp.arange(n_features) % 2 == 0, 1.0, 0.0)
    key = jax.random.PRNGKey(0)
    dt = 0.5
    layer = AffineCoupling(n_features, n_hidden, mask, key, dt)

    y_forward, log_det_forward = jax.vmap(layer.forward)(x)
    x_recon, log_det_inverse = jax.vmap(layer.inverse)(y_forward)

    assert jnp.allclose(x, jnp.round(x_recon, decimals=5))
    assert jnp.allclose(log_det_forward, -log_det_inverse)


def test_realnvp():
    n_features = 3
    n_hidden = 4
    n_layers = 2
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0), 2)
    model = RealNVP(n_features, n_layers, n_hidden, rng_key)

    y, log_det = jax.vmap(model)(x)

    assert y.shape == x.shape
    assert log_det.shape == (2,)

    y_inv, log_det_inv = jax.vmap(model.inverse)(y)

    assert y_inv.shape == x.shape
    assert log_det_inv.shape == (2,)
    assert jnp.allclose(x, y_inv)
    assert jnp.allclose(log_det, -log_det_inv)

    rng_key = jax.random.PRNGKey(0)
    samples = model.sample(rng_key, 2)

    assert samples.shape == (2, 3)

    log_prob = jax.vmap(model.log_prob)(samples)

    assert log_prob.shape == (2,)


def test_rqspline():
    n_features = 3
    hidden_layes = [16, 16]
    n_layers = 2
    n_bins = 8

    rng_key, rng_subkey = jax.random.split(jax.random.PRNGKey(0), 2)
    model = MaskedCouplingRQSpline(
        n_features, n_layers, hidden_layes, n_bins, jax.random.PRNGKey(10)
    )

    rng_key = jax.random.PRNGKey(0)
    samples = model.sample(rng_key, 2)

    assert samples.shape == (2, 3)

    log_prob = jax.vmap(model.log_prob)(samples)

    assert log_prob.shape == (2,)
