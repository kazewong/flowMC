import equinox as eqx  # Equinox utilities
import jax
import jax.numpy as jnp  # JAX NumPy
import optax  # Optimizers

from flowMC.resource.nf_model.realNVP import RealNVP
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline


def test_realNVP():
    key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    data = jax.random.normal(key1, (100, 2))

    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    model = RealNVP(2, 4, 32, rng)
    optim = optax.adam(learning_rate, momentum)
    state = optim.init(eqx.filter(model, eqx.is_array))

    rng, best_model, state, loss_values = model.train(
        init_rng, data, optim, state, num_epochs, batch_size, verbose=True
    )
    rng_key_nf = jax.random.PRNGKey(124098)
    model.sample(rng_key_nf, 10000)


def test_rqSpline():
    n_dim = 2
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    data = jax.random.normal(key1, (batch_size, n_dim))

    n_layers = 4
    hidden_dim = 32
    num_bins = 4

    model = MaskedCouplingRQSpline(
        n_dim,
        n_layers,
        [hidden_dim, hidden_dim],
        num_bins,
        rng,
        data_mean=jnp.mean(data, axis=0),
        data_cov=jnp.cov(data.T),
    )
    optim = optax.adam(learning_rate, momentum)
    state = optim.init(eqx.filter(model, eqx.is_array))

    rng, best_model, state, loss_values = model.train(
        init_rng, data, optim, state, num_epochs, batch_size, verbose=True
    )
    rng_key_nf = jax.random.PRNGKey(124098)
    model.sample(rng_key_nf, 10000)
