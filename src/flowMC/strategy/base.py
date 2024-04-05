from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Base class for strategies, which are basically wrapper blocks that modify the state of the sampler

    This is an abstract template that should not be directly used.
    
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError