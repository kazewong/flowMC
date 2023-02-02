from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import RQSpline
import jax
import jax.numpy as jnp  # JAX NumPy

from flowMC.nfmodel.utils import *
from flax.training import train_state  # Useful dataclass to keep train state
import optax  # Optimizers
import flax

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt



def test_realNVP():

    data = jnp.array(make_moons(n_samples=100, noise=0.05)[0])

    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)

    model = RealNVP(2, 2, 16, 1)
    def create_train_state(rng, learning_rate, momentum):
        params = model.init(rng, jnp.ones((1, 2)))["params"]
        tx = optax.adam(learning_rate, momentum)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    state = create_train_state(init_rng, learning_rate, momentum)

    variables = model.init(rng, jnp.ones((1, 2)))["variables"]
    variables = variables.unfreeze()
    variables["base_mean"] = jnp.mean(data, axis=0)
    variables["base_cov"] = jnp.cov(data.T)
    variables = flax.core.freeze(variables)

    train_flow, train_epoch, train_step = make_training_loop(model)
    rng, state, loss_values = train_flow(
        rng, state, variables, data, num_epochs, batch_size
    )

    rng_key_nf = jax.random.PRNGKey(124098)
    sample_nf(model, state.params, rng_key_nf, 10000, variables)

def test_rqSpline():


    data = jnp.array(make_moons(n_samples=100, noise=0.05)[0])

    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    model = RQSpline(2, 4, [16, 16], 8)

    def create_train_state(rng, learning_rate, momentum):
        params = model.init(rng, jnp.ones((1, 2)))["params"]
        tx = optax.adam(learning_rate, momentum)
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    state = create_train_state(init_rng, learning_rate, momentum)

    variables = model.init(rng, jnp.ones((1, 2)))["variables"]
    variables = variables.unfreeze()
    variables["base_mean"] = jnp.mean(data, axis=0)
    variables["base_cov"] = jnp.cov(data.T)
    variables = flax.core.freeze(variables)

    train_flow, train_epoch, train_step = make_training_loop(model)
    rng, state, loss_values = train_flow(
        rng, state, variables, data, num_epochs, batch_size
    )
    rng_key_nf = jax.random.PRNGKey(124098)
    sample_nf(model, state.params, rng_key_nf, 10000, variables)
