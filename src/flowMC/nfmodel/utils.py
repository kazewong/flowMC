import jax
import jax.numpy as jnp  # JAX NumPy
import jax.random as random  # JAX random
from tqdm import trange


def make_training_loop(model):
    """
    Create a function that trains an NF model.

    Args:
        model: a neural network model with a `loss` method.

    Returns:
        train_flow (Callable): wrapper function that trains the model.

    """

    def train_step(batch, state, variables):
        def loss(params):
            log_det = model.apply(
                {"params": params, "variables": variables}, batch, method=model.log_prob
            )
            return -jnp.mean(log_det)

        grad_fn = jax.value_and_grad(loss)
        value, grad = grad_fn(state.params)
        state = state.apply_gradients(grads=grad)
        return value, state

    train_step = jax.jit(train_step)

    def train_epoch(rng, state, variables, train_ds, batch_size):
        """Train for a single epoch."""
        train_ds_size = len(train_ds)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = train_ds[perm, ...]
                value, state = train_step(batch, state, variables)
        else:
            value, state = train_step(train_ds, state, variables)

        return value, state

    def train_flow(rng, state, variables, data, num_epochs, batch_size):
        loss_values = jnp.zeros(num_epochs)
        pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        best_state = state
        best_loss = 1e9
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, state = train_epoch(input_rng, state, variables, data, batch_size)
            # print('Train loss: %.3f' % value)
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_state = state
                best_loss = loss_values[epoch]
            if num_epochs > 10:
                if epoch % int(num_epochs / 10) == 0:
                    pbar.set_description(f"Training NF, current loss: {value:.3f}")
            else:
                if epoch == num_epochs:
                    pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, best_state, loss_values

    return train_flow, train_epoch, train_step


def sample_nf(model, param, rng_key, n_sample, variables):
    rng_key, subkey = random.split(rng_key)
    samples = model.apply(
        {"params": param, "variables": variables}, subkey, n_sample, method=model.sample
    )
    # samples = jnp.flip(samples[0],axis=1)
    return rng_key, samples
