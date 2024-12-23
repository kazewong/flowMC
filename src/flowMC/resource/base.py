from abc import ABC, abstractmethod

from jaxtyping import Array, Float, PRNGKeyArray, PyTree


class Resource(ABC):

    """
    Base class for resources.
    Resources are objects such as local sampler and neural networks.

    This is an abstract template that should not be directly used.    
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def print_parameters(self):
        """
        Function to print the tunable parameters of the resource.
        """
        raise NotImplementedError