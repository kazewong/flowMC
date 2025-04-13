from flowMC.resource.base import Resource
from typing import TypeVar
import numpy as np

TState = TypeVar("TState", bound="State")


class State(Resource):
    """A Resource class that holds the state of the system.
    This is essentially a wrapper around a dictionary such that it can be
    handled by flowMC.

    We restrict the type of the state to be simple types including integers, booleans and strings.
    The main reason for this is State is expected to be used to indiciate stage of individual
    strategies instead of storing parameters to resources.
    I.e. State should hold whehter the sampler is in training phase or production phase.
    But not mass matrix of a kernel per se.
    """

    name: str
    data: dict[str, int | bool | str]

    def __repr__(self):
        return "State " + self.name + " with shape " + str(len(self.data))

    def __init__(self, data: dict[str, int | bool | str], name: str = "State"):
        """Initialize the state.

        Args:
            data (dict): The data to initialize the state with.
            name (str): The name of the state.
        """

        self.name = name
        self.data = data

    def update(self, key: list[str], value: list[int | bool | str]):
        """Update the state with new data.

        This will modify the state in place.

        Args:
            key (str): The key to update.
            value (int | bool | str): The value to update.
        """
        for k, v in zip(key, value):
            self.data[k] = v
            print(f"Updated state {k} to {v}")

    def print_parameters(self):
        print(f"State: {self.name} with shape {len(self.data)} and data {self.data}")

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            data=self.data,  # type: ignore
        )

    def load_resource(self: TState, path: str) -> TState:
        data = np.load(path)
        result = State(data["name"], data["data"])
        return result  # type: ignore
