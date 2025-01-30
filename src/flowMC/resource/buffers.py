from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np
from jaxtyping import Array, Float

TBuffer = TypeVar("TBuffer", bound="Buffer")

class Buffer(Resource):

    name: str
    buffer: np.ndarray
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
        self.buffer = np.zeros((n_chains, n_steps, n_dims))
        self.current_position = 0

    def update_buffer(self, updates: Array, length: int):
        start = self.current_position
        self.buffer[:, start: start + length] = updates
        self.current_position = start + length

    def print_parameters(self):
        print(
            f"Buffer: {self.n_chains} chains, {self.n_steps} steps, {self.n_dims} dimensions"
        )

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            buffer=self.buffer,
        )

    def load_resource(self: TBuffer, path: str) -> TBuffer:
        data = np.load(path)
        buffer = data["buffer"]
        result = Buffer(data["name"], buffer.shape[0], buffer.shape[1], buffer.shape[2])
        result.buffer = buffer
        return result # type: ignore
