from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np
from jaxtyping import Array, Float
import jax.numpy as jnp
import jax

TBuffer = TypeVar("TBuffer", bound="Buffer")


class Buffer(Resource):
    name: str
    data: Float[Array, " ..."]
    cursor: int = 0
    cursor_dim: int = 0

    def __repr__(self):
        return "Buffer " + self.name + " with shape " + str(self.data.shape)

    @property
    def shape(self):
        return self.data.shape

    def __init__(self, name: str, shape: tuple[int, ...], cursor_dim: int = 0):
        self.cursor_dim = cursor_dim
        self.name = name
        self.data = jnp.zeros(shape) - jnp.inf

    def __call__(self):
        return self.data

    def update_buffer(self, updates: Array, start: int = 0):
        """Update the buffer with new data.

        This will modify the buffer in place.
        The cursor is expected to propagate the buffer in the cursor_dim
        with length equal to the length of the updates in its first dimension.
        """
        self.data = jax.lax.dynamic_update_slice_in_dim(
            self.data, updates, start, self.cursor_dim
        )

    def print_parameters(self):
        print(
            f"Buffer: {self.name} with shape {self.data.shape} and cursor"
            f" {self.cursor} at dimension {self.cursor_dim}"
        )

    def get_distribution(self, n_bins: int = 100):
        return np.histogram(self.data.flatten(), bins=n_bins)

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            data=self.data,
        )

    def load_resource(self: TBuffer, path: str) -> TBuffer:
        data = np.load(path)
        buffer: Float[Array, " ..."] = data["data"]
        result = Buffer(data["name"], buffer.shape)
        result.data = buffer
        return result  # type: ignore
