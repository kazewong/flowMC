
from nfsampler.nfmodel.maf import MaskedAutoregressiveFlow
from nfsampler.sampler.Gaussian_random_walk import rw_metropolis_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.stats import multivariate_normal
import numpy as np  

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
import optax                           # Optimizers
from functools import partial
from jax.scipy.stats import norm


"""
Training a masked autoregressive flow to fit the dual moons dataset.
"""

"""
Define hyper-parameters here.
"""

learning_rate = 0.01
momentum = 0.9
num_epochs = 300
batch_size = 10000

def train_flow(rng, model, state, data):

    @jax.jit
    def train_step(state, batch):
        def loss(params):
            y, log_det = model.apply({'params': params},batch)
            mean = jnp.zeros((batch.shape[0],model.n_dim))
            cov = jnp.repeat(jnp.eye(model.n_dim)[None,:],batch.shape[0],axis=0)
            log_det = log_det + multivariate_normal.logpdf(y,mean,cov)
            return -jnp.mean(log_det)
        grad_fn = jax.value_and_grad(loss)
        value, grad = grad_fn(state.params)
        state = state.apply_gradients(grads=grad)
        return state

    @jax.jit
    def eval_step(params, batch):
        y, log_det = model.apply({'params': params},batch)
        mean = jnp.zeros((batch.shape[0],model.n_dim))
        cov = jnp.repeat(jnp.eye(model.n_dim)[None,:],batch.shape[0],axis=0)
        log_det = log_det + multivariate_normal.logpdf(y,mean,cov)
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
        print('Epoch %d' % epoch)
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        state = train_epoch(state, data, batch_size, epoch, input_rng)
        print('Loss: %.3f' % eval_step(state.params, data))

    return rng, state

def neal_funnel(x):
    y_dist = partial(norm.logpdf, loc=0, scale=3)
    x_dist = partial(norm.logpdf, loc=0, scale=jnp.exp(x[0]/2))
    y_pdf = y_dist(x[0])
    x_pdf = x_dist(x[1:])
    return y_pdf + jnp.sum(x_pdf,axis=0)

n_dim = 20
n_samples = 10000
n_chains = 2
precompiled = False
rng_key = jax.random.PRNGKey(42)
rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,2)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

initial_position = jnp.zeros((n_dim, n_chains)) # (n_dim, n_chains)

model = MaskedAutoregressiveFlow(n_dim,64,6)
params = model.init(init_rng_keys_nf, jnp.ones((1,n_dim)))['params']

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
                    out_axes=0)

tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

print(rng_keys_mcmc)
rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, neal_funnel, initial_position)
flat_chain = positions.reshape(-1,n_dim)
rng_keys_nf, state1 = train_flow(rng_key_nf, model, state, flat_chain)

rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, neal_funnel, positions.T[:,-1])
flat_chain = positions.reshape(-1,n_dim)
rng_keys_nf, state1 = train_flow(rng_key_nf, model, state, flat_chain)
