import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal

def train_step(model, batch, state, variables):
    def loss(params):
        log_det = model.apply({'params': params,'variables': variables}, batch, method=model.log_prob)
        return - jnp.mean(log_det)

    grad_fn = jax.value_and_grad(loss)
    value, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return value, state

train_step = jax.jit(train_step,static_argnums=(0,))

def train_flow(rng, model, state, data, num_epochs, batch_size, variables):

    def train_epoch(state, train_ds, batch_size, epoch, rng, variables):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size

        perms = jax.random.permutation(rng, train_ds_size)
        perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))
        for perm in perms:
            batch = train_ds[perm, ...]
            value, state = train_step(model, state, batch, variables)

        return value, state


    loss_values = []
    for epoch in range(1, num_epochs + 1):
        #print('Epoch %d' % epoch)
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        value, state = train_epoch(state, data, batch_size, epoch, input_rng, variables)
        #print('Train loss: %.3f' % value)
        loss_values.append(value)

    return rng, state, loss_values

def sample_nf(model, param, variables, rng_key, n_sample):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply({'params': param,'variables': variables}, subkey, n_sample, method=model.sample)
    samples = jnp.flip(samples[0],axis=1)
    return rng_key,samples