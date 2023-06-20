from abc import abstractmethod
from typing import Tuple
import equinox as eqx
import jax
from jaxtyping import Array
class NFModel(eqx.Module):

    @abstractmethod
    def __init__(self):
        return NotImplemented

    @abstractmethod
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented
    
    @abstractmethod
    def log_prob(self, x: Array) -> Array:
        return NotImplemented
    
    @abstractmethod
    def sample(self, rng_key: jax.random.PRNGKey, n_samples: int) -> Array:
        return NotImplemented

    @abstractmethod
    def inverse(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

    @abstractmethod
    def inverse_vmap(self, x: Array) -> Tuple[Array, Array]:
        return NotImplemented

