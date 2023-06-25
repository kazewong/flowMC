import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
from tqdm import trange
import optax
import equinox as eqx
from flowMC.nfmodel.base import NFModel
from typing import Callable, Tuple
from jaxtyping import Array, PRNGKeyArray


def make_training_loop(model: eqx.Module, optim: optax.GradientTransformation) -> Callable:
    """
    Create a function that trains an NF model.

    Args:
        model (eqx.Model): NF model to train.
        optim (optax.GradientTransformation): Optimizer.

    Returns:
        train_flow: Function that trains the model.
    """
    @eqx.filter_value_and_grad
    def loss_fn(model, x):
        return -jnp.mean(jax.vmap(model.log_prob)(x))

    @eqx.filter_jit
    def train_step(model, x, opt_state):
        loss, grads = loss_fn(model, x)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    def train_epoch(rng, model, state, train_ds, batch_size):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                value, model, state = train_step(model, batch, state)
        else:
            value, model, state = train_step(model, train_ds, state)
            
        return value, model, state

    def train_flow(rng, model, data, num_epochs, batch_size, verbose: bool = False):
        state = optim.init(eqx.filter(model,eqx.is_array))
        loss_values = jnp.zeros(num_epochs)
        if verbose:
            pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        else:
            pbar = range(num_epochs)
        best_model = model
        best_loss = 1e9
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, model, state = train_epoch(input_rng, model, state, data, batch_size)
            # print('Train loss: %.3f' % value)
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_model = model
                best_loss = loss_values[epoch]
            if verbose:
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")
                else:
                    if epoch == num_epochs:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, best_model, loss_values

    return train_flow, train_epoch, train_step


def sample_nf(model: NFModel, rng_key: jax.random.PRNGKey, n_sample: int) -> Tuple[jax.random.PRNGKeyArray, Array]:
    """  
    Sample from a NF model given a set of parameters and accompanying variables. 

    Args:
        model (NFModel): NF model to sample from.
        rng_key (jax.random.PRNGKey): Random number generator key.
        n_sample (int): Number of samples to draw.

    Returns:
        rng_key (jax.random.PRNGKey): Random number generator key.
        samples (Array): Samples from the NF model.
    """
    rng_key, subkey = random.split(rng_key)
    samples = model.sample(subkey, n_sample)
    return rng_key, samples

def eval_nf(model: NFModel, x: jnp.ndarray) -> Array:
    """
    Evaluate a NF model given a set of parameters and accompanying variables.

    Args:
        model (NFModel): NF model to evaluate.
        x (jnp.ndarray): Input data.

    Returns:
        log_prob (Array): Log probability of the input data.
    """
    log_prob = model.log_prob(x)
    return log_prob