
from nfsampler.nfmodel.maf import MaskedAutoregressiveFlow
from nfsampler.sampler.Gaussian_random_walk import rw_metropolis_sampler
import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.special import logsumexp
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
batch_size = 800

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
        return value,state

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
            value, state = train_step(state, batch)

        return value, state

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d' % epoch)
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        value, state = train_epoch(state, data, batch_size, epoch, input_rng)
        print('Train loss: %.3f' % value)

    return rng, state

def dual_moon_pe(x):
    """
    Term 2 separate the distriubiotn
    """
    term1 = 0.5 * ((jnp.linalg.norm(x, axis=-1) - 2) / 0.1) ** 2
    term2 = -0.5 * ((x[..., :1] + jnp.array([-5., 5.])) / 0.6) ** 2
    term3 = -0.5 * ((x[..., 1:2] + jnp.array([-5., 5.])) / 0.8) ** 2
    return -(term1 - logsumexp(term2, axis=-1) - logsumexp(term3, axis=-1))

n_dim = 3
n_samples = 1000
n_chains = 8
precompiled = False
rng_key = jax.random.PRNGKey(42)
rng_key_mcmc, rng_key_nf = jax.random.split(rng_key,2)

rng_keys_mcmc = jax.random.split(rng_key_mcmc, n_chains)  # (nchains,)
rng_keys_nf, init_rng_keys_nf = jax.random.split(rng_key_nf,2)

initial_position = jnp.zeros((n_dim, n_chains)) #(n_dim, n_chains)

model = MaskedAutoregressiveFlow(n_dim,64,4)
params = model.init(init_rng_keys_nf, jnp.ones((1,n_dim)))['params']

run_mcmc = jax.vmap(rw_metropolis_sampler, in_axes=(0, None, None, 1),
                    out_axes=0)

tx = optax.adam(learning_rate, momentum)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

print(rng_keys_mcmc)
rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, dual_moon_pe, initial_position)
flat_chain = positions.reshape(-1,n_dim)
rng_keys_nf, state1 = train_flow(rng_key_nf, model, state, flat_chain)

# rng_keys_mcmc, positions, log_prob = run_mcmc(rng_keys_mcmc, n_samples, dual_moon_pe, positions.T[:,-1])
# flat_chain = positions.reshape(-1,n_dim)
# rng_keys_nf, state1 = train_flow(rng_key_nf, model, state1, flat_chain)
