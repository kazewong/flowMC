from abc import abstractmethod
import copy
import equinox as eqx
from typing import overload, Optional
from typing_extensions import Self
from jaxtyping import Array, PRNGKeyArray, Float
import optax
from tqdm import trange, tqdm
import jax.numpy as jnp
import jax


class NFModel(eqx.Module):
    """
    Base class for normalizing flow models.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    def __call__(self, x: Float[Array, "n_dim"]) -> tuple[Float[Array, "n_dim"], Float]:
        """
        Forward pass of the model.

        Args:
            x (Float[Array, "n_dim"]): Input data.

        Returns:
            tuple[Float[Array, "n_dim"], Float]: Output data and log determinant of the Jacobian.
        """
        return self.forward(x)

    @abstractmethod
    def log_prob(self, x: Float[Array, "n_dim"]) -> Float:
        return NotImplemented

    @abstractmethod
    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> Array:
        return NotImplemented

    @abstractmethod
    def forward(
        self, x: Float[Array, "n_dim"], key: Optional[PRNGKeyArray] = None
    ) -> tuple[Float[Array, "n_dim"], Float]:
        """
        Forward pass of the model.

        Args:
            x (Float[Array, "n_dim"]): Input data.

        Returns:
            tuple[Float[Array, "n_dim"], Float]: Output data and log determinant of the Jacobian.
        """
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Float[Array, "n_dim"]) -> tuple[Float[Array, "n_dim"], Float]:
        """
        Inverse pass of the model.

        Args:
            x (Float[Array, "n_dim"]): Input data.

        Returns:
            tuple[Float[Array, "n_dim"], Float]: Output data and log determinant of the Jacobian.
        """
        return NotImplemented

    @abstractmethod
    def n_features(self) -> int:
        return NotImplemented

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str):
        self = eqx.tree_deserialise_leaves(path + ".eqx", self)

    @eqx.filter_value_and_grad
    def loss_fn(self, x):
        return -jnp.mean(self.log_prob(x))

    @eqx.filter_jit
    def train_step(
        self: Self,
        x: Float[Array, "n_batch n_dim"],
        optim: optax.GradientTransformation,
        state: optax.OptState,
    ) -> tuple[Float, Self, optax.OptState]:
        """Train for a single step.

        Args:
            model (eqx.Model): NF model to train.
            x (Array): Training data.
            opt_state (optax.OptState): Optimizer state.

        Returns:
            loss (Array): Loss value.
            model (eqx.Model): Updated model.
            opt_state (optax.OptState): Updated optimizer state.
        """
        loss, grads = self.loss_fn(x)
        updates, state = optim.update(grads, state)
        model = eqx.apply_updates(self, updates)
        return loss, model, state

    def train_epoch(
        self: Self,
        rng: PRNGKeyArray,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        data: Float[Array, "n_example n_dim"],
        batch_size: Float,
    ) -> tuple[Float, Self, optax.OptState]:
        """Train for a single epoch."""
        model = self
        train_ds_size = len(data)
        steps_per_epoch = train_ds_size // batch_size
        if steps_per_epoch > 0:
            perms = jax.random.permutation(rng, train_ds_size)

            perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
            perms = perms.reshape((steps_per_epoch, batch_size))
            for perm in perms:
                batch = data[perm, ...]
                value, model, state = model.train_step(batch, optim, state)
        else:
            value, model, state = model.train_step(data, optim, state)

        return value, model, state

    def train(
        self: Self,
        rng: PRNGKeyArray,
        data: Array,
        optim: optax.GradientTransformation,
        state: optax.OptState,
        num_epochs: int,
        batch_size: int,
        verbose: bool = True,
    ) -> tuple[PRNGKeyArray, Self, optax.OptState, Array]:
        """Train a normalizing flow model.

        Args:
            rng (PRNGKeyArray): JAX PRNGKey.
            model (eqx.Module): NF model to train.
            data (Array): Training data.
            num_epochs (int): Number of epochs to train for.
            batch_size (int): Batch size.
            verbose (bool): Whether to print progress.

        Returns:
            rng (PRNGKeyArray): Updated JAX PRNGKey.
            model (eqx.Model): Updated NF model.
            loss_values (Array): Loss values.
        """
        loss_values = jnp.zeros(num_epochs)
        if verbose:
            pbar = trange(num_epochs, desc="Training NF", miniters=int(num_epochs / 10))
        else:
            pbar = range(num_epochs)
        best_model = model = self
        best_loss = 1e9
        for epoch in pbar:
            # Use a separate PRNG key to permute image data during shuffling
            rng, input_rng = jax.random.split(rng)
            # Run an optimization step over a training batch
            value, model, state = model.train_epoch(input_rng, optim, state, data, batch_size)
            loss_values = loss_values.at[epoch].set(value)
            if loss_values[epoch] < best_loss:
                best_model = model
                best_loss = loss_values[epoch]
            if verbose:
                assert isinstance(pbar, tqdm)
                if num_epochs > 10:
                    if epoch % int(num_epochs / 10) == 0:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")
                else:
                    if epoch == num_epochs:
                        pbar.set_description(f"Training NF, current loss: {value:.3f}")

        return rng, best_model, state, loss_values


class Bijection(eqx.Module):
    """
    Base class for bijective transformations.

    This is an abstract template that should not be directly used."""

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(
        self, x: Array, key: Optional[PRNGKeyArray] = None
    ) -> tuple[Array, Array]:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Array) -> tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array) -> tuple[Array, Array]:
        return NotImplemented


class Distribution(eqx.Module):
    """
    Base class for probability distributions.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        return self.log_prob(x)

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        return NotImplemented

    @abstractmethod
    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> Float[Array, " n_samples n_features"]:
        return NotImplemented
