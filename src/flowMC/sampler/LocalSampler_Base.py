from abc import abstractmethod
from typing import Callable
class LocalSamplerBase:

    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        """
        Initialize the sampler class
        """
        self.logpdf = logpdf
        self.jit = jit
        self.params = params

    @abstractmethod
    def make_kernel(self, return_aux = False) -> Callable:
        """
        Make the kernel of the sampler for one update
        """

    @abstractmethod
    def make_update(self) -> Callable:
        """
        Make the update function for multiple steps
        """

    @abstractmethod
    def make_sampler(self) -> Callable:
        """
        Make the sampler for multiple chains given initial positions
        """