from abc import abstractmethod, abstractproperty
import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray, Float
class NFModel(eqx.Module):

    """
    Base class for normalizing flow models.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """
        Forward pass of the model.

        Args:
            x (Array): Input data.

        Returns:
            tuple[Array, Array]: Output data and log determinant of the Jacobian.
        """
        return self.forward(x)

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        return NotImplemented

    @abstractmethod
    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> Array:
        return NotImplemented

    @abstractmethod
    def forward(self, x: Array) -> tuple[Array, Array]:
        """
        Forward pass of the model.

        Args:
            x (Array): Input data.

        Returns:
            tuple[Array, Array]: Output data and log determinant of the Jacobian."""
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array) -> tuple[Array, Array]:
        """
        Inverse pass of the model.

        Args:
            x (Array): Input data.

        Returns:
            tuple[Array, Array]: Output data and log determinant of the Jacobian."""
        return NotImplemented

    @abstractproperty
    def n_features(self) -> int:
        return NotImplemented

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path + ".eqx", self)

    def load_model(self, path: str) -> eqx.Module:
        return eqx.tree_deserialise_leaves(path + ".eqx", self)


class Bijection(eqx.Module):

    """
    Base class for bijective transformations.

    This is an abstract template that should not be directly used."""

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array) -> tuple[Array, Array]:
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

    def __call__(self, x: Array) -> Array:
        return self.log_prob(x)

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        return NotImplemented

    @abstractmethod
    def sample(self, rng_key: PRNGKeyArray, n_samples: int) -> Float[Array, " n_samples n_features"]:
        return NotImplemented
