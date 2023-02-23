# Import rqSpline model and train it on a multivariate normal distribution

import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
import optax  # Optimizers
import flax  # Deep Learning library for JAX
from flax.training import train_state  # Useful dataclass to keep train state

from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.nfmodel.utils import make_training_loop, sample_nf

def normal(x, mu, sigma):
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))


def test_rqSpline():
    n_dim = 5

    # Generate data
    rng_key, rng_data = jax.random.split(jax.random.PRNGKey(0))
    data = jax.random.multivariate_normal(rng_data, jnp.zeros(n_dim), jnp.eye(n_dim), (10000,))

    key, rng_model, rng_init, rng_train, rng_nf_sample = jax.random.split(
        jax.random.PRNGKey(0), 5
    )

    # Model parameters
    n_layers = 2
    n_hiddens = [16, 16]
    n_bins = 8

    model = RQSpline(n_dim, n_layers, n_hiddens, n_bins)

    variables = model.init(rng_model, jnp.ones((1, n_dim)))["variables"]
    variables = variables.unfreeze()
    variables["base_mean"] = jnp.mean(data, axis=0)
    variables["base_cov"] = jnp.cov(data.T)
    variables = flax.core.freeze(variables)

    def create_train_state(rng, learning_rate, momentum):
        params = model.init(rng, jnp.ones((1, n_dim)))["params"]
        tx = optax.adam(learning_rate, momentum)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Optimization parameters
    num_epochs = 3
    batch_size = 10000
    learning_rate = 0.001
    momentum = 0.9
    state = create_train_state(rng_init, learning_rate, momentum)
    
    train_flow, train_epoch, train_step = make_training_loop(model)
    rng, state, loss_values = train_flow(
        rng_train, state, variables, data, num_epochs, batch_size
    )