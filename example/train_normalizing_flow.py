from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import jax
import jax.numpy as jnp  # JAX NumPy

from flowMC.nfmodel.utils import *
import equinox as eqx
import optax  # Optimizers
from flowMC.nfmodel.utils import make_training_loop

from sklearn.datasets import make_moons

"""
Training a Masked Coupling RQSpline flow to fit the dual moons dataset.
"""

num_epochs = 3000
batch_size = 10000
learning_rate = 0.001
momentum = 0.9
n_layers = 10
n_hidden = 128
dt = 1 / n_layers

data = make_moons(n_samples=20000, noise=0.05)
data = jnp.array(data[0])

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)

model = MaskedCouplingRQSpline(2, 10, [128,128], 8 , init_rng, data_cov = jnp.cov(data.T), data_mean = jnp.mean(data, axis=0))

optim = optax.adam(learning_rate)
train_flow, _, _ = make_training_loop(optim)

key, model, loss = train_flow(rng, model, data, num_epochs, batch_size, verbose=True)

nf_samples = model.sample(jax.random.PRNGKey(124098),5000)
 