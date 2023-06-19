from abc import abstractmethod
from equinox import Module

class NFModel(Module):

    @abstractmethod
    def __call__(self, x):
        return NotImplemented
    
    @abstractmethod
    def log_prob(self, x):
        return NotImplemented
    
    @abstractmethod
    def sample(self, n_samples):
        return NotImplemented

