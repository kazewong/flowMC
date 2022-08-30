import jax
import jax.numpy as jnp                # JAX NumPy
import jax.random as random            # JAX random
from tqdm import tqdm


def train_step(model, batch, state, variables):
    def loss(params):
        log_det = model.apply({'params': params,'variables': variables}, batch, method=model.log_prob)
        return -jnp.mean(log_det)

    grad_fn = jax.value_and_grad(loss)
    value, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return value, state

train_step = jax.jit(train_step, static_argnums=(0,))

def train_epoch(model, state, train_ds, batch_size, epoch, rng, variables):
    """Train for a single epoch."""
    train_ds_size = len(train_ds)
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    for perm in perms:
        batch = train_ds[perm, ...]
        value, state = train_step(model, batch, state, variables)

    return value, state

def train_flow(rng, model, state, data, num_epochs, batch_size, variables):
    loss_values = jnp.zeros(num_epochs)
    for epoch in tqdm(range(1, num_epochs + 1),desc='Training NF',miniters=int(num_epochs/10)):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        value, state = train_epoch(model, state, data, batch_size, epoch, input_rng, variables)
        #print('Train loss: %.3f' % value)
        loss_values = loss_values.at[epoch].set(value)
    
    return rng, state, loss_values

def sample_nf(model, param, rng_key, n_sample, variables):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply({'params': param,'variables': variables}, subkey, n_sample, method=model.sample)
    # samples = jnp.flip(samples[0],axis=1)
    return rng_key,samples