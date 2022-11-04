from abc import abstractmethod
from typing import Callable
class LocalSamplerBase:

    @abstractmethod
    def __init__(self, logpdf: Callable, jit: bool, params: dict) -> Callable:
        """
        
        """

    @abstractmethod
    def make_kernel(self, return_aux = False) -> Callable:
        """
        
        """

    @abstractmethod
    def make_update(self) -> Callable:
        """
        """

    @abstractmethod
    def make_sampler(self) -> Callable:
        """
        """