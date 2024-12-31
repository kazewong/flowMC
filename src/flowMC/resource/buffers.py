from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np

TBuffer = TypeVar("TBuffer", bound="Buffer")

class Buffer(Resource):

    name: str
    n_chains: int
    n_steps: int
    n_dims: int

    def __init__(self, name: str, n_chains: int, n_steps: int, n_dims: int):
        self.name = name
        self.n_chains = n_chains
        self.n_steps = n_steps
        self.n_dims = n_dims

    def print_parameters(self):
        print(
            f"Buffer: {self.n_chains} chains, {self.n_steps} steps, {self.n_dims} dimensions"
        )

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            n_chains=self.n_chains,
            n_steps=self.n_steps,
            n_dims=self.n_dims,
        )

    def load_resource(self: TBuffer, path: str) -> TBuffer:
        data = np.load(path)
        return Buffer(data["name"], data["n_chains"], data["n_steps"], data["n_dims"]) # type: ignore
