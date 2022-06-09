
from nfsampler.nfmodel.realNVP import RealNVP
import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.stats import multivariate_normal

from nfsampler.nfmodel.utils import sample_nf
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers

from sklearn.datasets import make_moons

"""
Training a masked autoregressive flow to fit the dual moons dataset.
"""


model = RealNVP(10, 2, 64, 1)
data = make_moons(n_samples=10000, noise=0.05)
mean = jnp.mean(data[0], axis=0)[None,:]
cov = jnp.cov(data[0].T)[None,:]
key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0),3)

def create_train_state(rng, learning_rate, momentum):
    params = model.init(rng, jnp.ones((1,2)))['params']
    tx = optax.adam(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


learning_rate = 0.01
momentum = 0.9

state = create_train_state(init_rng, learning_rate, momentum)
variables = {'base_mean':mean, 'base_cov':cov}
# variables = model.init(rng, jnp.ones((1,2)))['variables']
# variables = {'base_mean':variables['base_mean'], 'base_cov':variables['base_cov'][None,:]}
@jax.jit
def train_step(state, batch):
    def loss(params):
        log_det = model.apply({'params': params,'variables': variables}, batch, method=model.log_prob)
        return -jnp.mean(log_det)
    grad_fn = jax.value_and_grad(loss)
    value, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return state

@jax.jit
def eval_step(params, batch):
    log_det = model.apply({'params': params,'variables': variables}, batch, method=model.log_prob)
    return -jnp.mean(log_det)

def train_epoch(state, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds)
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, train_ds_size)
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  for perm in perms:
    batch = train_ds[perm, ...]
    state = train_step(state, batch)

  return state

num_epochs = 1000
batch_size = 10000

for epoch in range(1, num_epochs + 1):
    print('Epoch %d' % epoch)
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state = train_epoch(state, data[0], batch_size, epoch, input_rng)
    print('Loss: %.3f' % eval_step(state.params, data[0]))


rng_key_nf = jax.random.PRNGKey(124098)
nf_samples = sample_nf(model, state.params, variables, rng_key_nf, batch_size)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(data[0][:,0], data[0][:,1], s=0.1, c='r',label='data')
plt.scatter(nf_samples[1][:,0], nf_samples[1][:,1], s=0.1, c='b',label='NF') # x-y is flipped in this configuration
plt.show()