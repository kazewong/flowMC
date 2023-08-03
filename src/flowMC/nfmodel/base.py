from abc import abstractmethod
from typing import Tuple
import equinox as eqx
import jax
from jaxtyping import Array
class NFModel(eqx.Module):

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.forward(x)
    
    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        return NotImplemented
    
    @abstractmethod
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        return NotImplemented

    @abstractmethod
    def forward(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

    def save_model(self, path: str):
        eqx.tree_serialise_leaves(path+".eqx", self)

    def load_model(self, path: str) -> eqx.Module:
        return eqx.tree_deserialise_leaves(path+".eqx", self)

class Bijection(eqx.Module):

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

class Distribution(eqx.Module):

    @abstractmethod
    def __init__(self):
        return NotImplemented

    def __call__(self, x: Array) -> Array:
        return self.log_prob(x)

    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        return NotImplemented

    @abstractmethod
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        return NotImplemented