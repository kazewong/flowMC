from flowMC.nfmodel.realNVP import RealNVP
import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
from jax.scipy.stats import multivariate_normal

from flowMC.nfmodel.utils import *
from flax import linen as nn  # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import optax  # Optimizers

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

"""
Training a masked RealNVP flow to fit the dual moons dataset.
"""

num_epochs = 300
batch_size = 10000
learning_rate = 0.001
momentum = 0.9
n_layers = 10
n_hidden = 100
dt = 1 / n_layers

model = RealNVP(10, 2, n_hidden, 1)
data = make_moons(n_samples=100000, noise=0.05)

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)


def create_train_state(rng, learning_rate, momentum):
    params = model.init(rng, jnp.ones((1, 2)))["params"]
    tx = optax.adam(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


state = create_train_state(init_rng, learning_rate, momentum)

variables = model.init(rng, jnp.ones((1, 2)))["variables"]


data = jnp.array(data[0])

rng, state, loss_values = train_flow(
    rng, model, state, data, num_epochs, batch_size, variables
)


rng_key_nf = jax.random.PRNGKey(124098)
nf_samples = sample_nf(model, state.params, rng_key_nf, 10000, variables)
plt.figure(figsize=(10, 9))
plt.scatter(data[:, 0], data[:, 1], s=0.1, c="r", label="Data")
plt.scatter(nf_samples[1][0][:, 0], nf_samples[1][0][:, 1], s=0.1, c="b", label="NF")
plt.show()
