from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import jax
import jax.numpy as jnp  # JAX NumPy

from flowMC.nfmodel.utils import *
from flax.training import train_state  # Useful dataclass to keep train state
import optax  # Optimizers
import flax

def test_realNVP():

    key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    data = jax.random.normal(key1, (100,2))

    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    model = RealNVP(4, 2, 32, rng, 1., base_cov = jnp.cov(data.T), base_mean = jnp.mean(data, axis=0))
    optim = optax.adam(learning_rate, momentum)

    train_flow, train_epoch, train_step = make_training_loop(model, optim)
    rng, best_model, loss_values = train_flow(
        rng, model, data, num_epochs, batch_size, verbose = True
    )
    rng_key_nf = jax.random.PRNGKey(124098)
    model.sample(rng_key_nf, 10000)


def test_rqSpline():

    key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    data = jax.random.normal(key1, (100,2))

    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    momentum = 0.9

    model = MaskedCouplingRQSpline(2, 4, [32,32], 4 , rng, base_mean = jnp.mean(data, axis=0), base_cov = jnp.cov(data.T))
    optim = optax.adam(learning_rate, momentum)


    train_flow, train_epoch, train_step = make_training_loop(model, optim)
    rng, best_model, loss_values = train_flow(
        rng, model, data, num_epochs, batch_size, verbose = True
    )
    rng_key_nf = jax.random.PRNGKey(124098)
    model.sample(rng_key_nf, 10000)
