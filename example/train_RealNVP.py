from flowMC.nfmodel.realNVP import RealNVP
from flowMC.nfmodel.rqSpline import RQSpline
import jax
import jax.numpy as jnp  # JAX NumPy

from flowMC.nfmodel.utils import *
import equinox as eqx
import optax  # Optimizers

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

"""
Training a masked RealNVP flow to fit the dual moons dataset.
"""

num_epochs = 1000
batch_size = 10000
learning_rate = 0.001
momentum = 0.9
n_layers = 10
n_hidden = 128
dt = 1 / n_layers

data = make_moons(n_samples=20000, noise=0.05)
data = jnp.array(data[0])

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)

model = RealNVP(n_layers, 2, n_hidden, rng, 1., base_cov = jnp.cov(data.T), base_mean = jnp.mean(data, axis=0))
# model = RQSpline(2, 6, [128,128], 8, rng)


@eqx.filter_value_and_grad
def loss_fn(model, x):
    return -jnp.mean(jax.vmap(model.log_prob)(x))

@eqx.filter_jit
def make_step(model, x, opt_state):
    loss, grads = loss_fn(model, x)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

optim = optax.adam(learning_rate)
opt_state = optim.init(eqx.filter(model,eqx.is_array))
for step in range(num_epochs):
    loss, model, opt_state = make_step(model, data, opt_state)
    loss = loss.item()
    print(f"step={step}, loss={loss}")


nf_samples = model.sample(jax.random.PRNGKey(124098),1000)
