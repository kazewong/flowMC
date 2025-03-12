from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np
from jaxtyping import Array, Float
import jax.numpy as jnp

TBuffer = TypeVar("TBuffer", bound="Buffer")


class Buffer(Resource):
    name: str
    data: Float[Array, "n_chains n_steps n_dims"]
    current_position: int = 0

    def __repr__(self):
        return str(self.data)

    @property
    def n_chains(self) -> int:
        return self.data.shape[0]

    @property
    def n_steps(self) -> int:
        return self.data.shape[1]

    @property
    def n_dims(self) -> int:
        return self.data.shape[2]

    def __init__(self, name: str, n_chains: int, n_steps: int, n_dims: int):
        self.name = name
        self.data = jnp.zeros((n_chains, n_steps, n_dims)) - jnp.inf

    def __call__(self):
        return self.data

    def update_buffer(self, updates: Array, length: int, start: int = 0):
        self.data = self.data.at[:, start : start + length].set(updates)

    def print_parameters(self):
        print(
            f"Buffer: {self.n_chains} chains,"
            "{self.n_steps} steps, {self.n_dims} dimensions"
        )

    def get_distribution(self, n_bins: int = 100):
        assert self.n_dims == 1, "Only 1D buffers are supported for now"
        return np.histogram(self.data.flatten(), bins=n_bins)

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            buffer=self.data,
        )

    def load_resource(self: TBuffer, path: str) -> TBuffer:
        data = np.load(path)
        buffer: Float[Array, "n_chains n_steps n_dims"] = data["buffer"]
        result = Buffer(data["name"], buffer.shape[0], buffer.shape[1], buffer.shape[2])
        result.data = buffer
        return result  # type: ignore
