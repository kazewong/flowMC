from abc import ABC, abstractmethod

from typing import Self


class Resource(ABC):
    """Base class for resources. Resources are objects such as local sampler and neural
    networks.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def print_parameters(self):
        """Function to print the tunable parameters of the resource."""
        raise NotImplementedError

    @abstractmethod
    def save_resource(self, path: str):
        """Function to save the resource.

        Args:
            path (str): Path to save the resource.
        """
        raise NotImplementedError

    @abstractmethod
    def load_resource(self, path: str) -> Self:
        """Function to load the resource.

        Args:
            path (str): Path to load the resource.
        """
        raise NotImplementedError
