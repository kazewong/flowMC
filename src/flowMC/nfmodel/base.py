from abc import abstractmethod
import equinox as eqx

class NFModel(eqx.Module):

    @abstractmethod
    def __init__(self):
        return NotImplemented

    @abstractmethod
    def __call__(self, x):
        return NotImplemented
    
    @abstractmethod
    def log_prob(self, x):
        return NotImplemented
    
    @abstractmethod
    def sample(self, n_samples):
        return NotImplemented

