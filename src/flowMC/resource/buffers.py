from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np
from jaxtyping import Array, Float
import jax.numpy as jnp

TBuffer = TypeVar("TBuffer", bound="Buffer")

class Buffer(Resource):

    name: str
    buffer: Float[Array, "n_chains n_steps n_dims"]
    current_position: int = 0

    @property
    def n_chains(self) -> int:
        return self.buffer.shape[0]
    
    @property
    def n_steps(self) -> int:
        return self.buffer.shape[1]
    
    @property
    def n_dims(self) -> int:
        return self.buffer.shape[2]

    def __init__(self, name: str, n_chains: int, n_steps: int, n_dims: int):
        self.name = name
        self.buffer = jnp.zeros((n_chains, n_steps, n_dims)) + jnp.nan

    def update_buffer(self, updates: Array, length: int, start: int = 0):
        self.buffer = self.buffer.at[:, start: start + length].set(updates)

    def print_parameters(self):
        print(
            f"Buffer: {self.n_chains} chains, {self.n_steps} steps, {self.n_dims} dimensions"
        )

    def get_distribution(self, n_bins: int = 100):
        assert self.n_dims == 1, "Only 1D buffers are supported for now"
        return np.histogram(self.buffer.flatten(), bins=n_bins)

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            buffer=self.buffer,
        )

    def load_resource(self: TBuffer, path: str) -> TBuffer:
        data = np.load(path)
        buffer: Float[Array, "n_chains n_steps n_dims"] = data["buffer"]
        result = Buffer(data["name"], buffer.shape[0], buffer.shape[1], buffer.shape[2])
        result.buffer = buffer
        return result # type: ignore
