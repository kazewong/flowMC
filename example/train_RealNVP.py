
from flowMC.nfmodel.realNVP import RealNVP
import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.stats import multivariate_normal

from flowMC.nfmodel.utils import sample_nf
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

"""
Training a masked RealNVP flow to fit the dual moons dataset.
"""

num_epochs = 10000
batch_size = 10000
learning_rate = 0.0001
momentum = 0.9
n_layers = 4
n_hidden = 100
dt = 1 / n_layers

fig = plt.figure(figsize=(30,10))
axes = [plt.subplot(1,3,i+1) for i in range(3)]

model = RealNVP(10, 2, 100, 1)
data = make_moons(n_samples=10000, noise=0.05) 
x_train = data[0] + 10

# plot training data
for i in range(3):
    ax = plt.sca(axes[i])
    plt.scatter(x_train[:,0], x_train[:,1], s=0.1, c='r',label='data')

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0),3)

def create_train_state(rng, learning_rate, momentum):
    params = model.init(rng, jnp.ones((1,2)))['params']
    tx = optax.adam(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

state = create_train_state(init_rng, learning_rate, momentum)

# test zero and one cov
mean = jnp.zeros((1, 2))
cov = jnp.eye(2)[None, :]
variables = {'base_mean':mean, 'base_cov':cov}
rng_key_nf = jax.random.PRNGKey(124098)
nf_samples = sample_nf(model, state.params, variables, rng_key_nf, batch_size)
plt.sca(axes[0])
plt.scatter(nf_samples[1][:,1], nf_samples[1][:,0], s=0.1, c='b',label='NF') # 

# test data cov and mean
mean = jnp.mean(x_train, axis=0)[None,:]
cov = jnp.cov(x_train.T)[None,:]
variables = {'base_mean':mean, 'base_cov':cov}
rng_key_nf = jax.random.PRNGKey(124098)
nf_samples = sample_nf(model, state.params, variables, rng_key_nf, batch_size)
plt.sca(axes[1])
plt.scatter(nf_samples[1][:,1], nf_samples[1][:,0], s=0.1, c='b',label='NF') # 

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

for epoch in range(1, num_epochs + 1):
    
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state = train_epoch(state, x_train, batch_size, epoch, input_rng)
    if epoch % int(num_epochs/10) == 0:
        print('Epoch %d' % epoch, end=' ')
        print('Loss: %.3f' % eval_step(state.params, x_train))


rng_key_nf = jax.random.PRNGKey(124098)
nf_samples = sample_nf(model, state.params, variables, rng_key_nf, batch_size)
plt.sca(axes[2])
plt.scatter(nf_samples[1][:,1], nf_samples[1][:,0], s=0.1, c='b',label='NF') # x-y is flipped in this configuration
plt.savefig('blu_whitening_moons.png')