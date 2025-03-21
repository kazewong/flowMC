from dataclasses import dataclass
from typing import Callable, Optional
from flowMC.resource.base import Resource
from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp
import jax

@dataclass
class Variable:
    name: str
    continuous: bool

@jax.tree_util.register_pytree_node_class
class LogPDF(Resource):

    """LogPDF class 
    
    """
    log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]]
    variables: list[Variable]

    @property
    def n_dims(self):
        return len(self.variables)

    def __repr__(self):
        return super().__repr__()
    
    def __init__(self, log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]], variables: Optional[list[Variable]] = None, n_dims: Optional[int] = None):
        self.log_pdf = log_pdf
        if variables is None and n_dims is not None:
            self.variables = [Variable("x_"+str(i), True) for i in range(n_dims)]
        elif variables is not None:
            self.variables = variables
        else:
            raise ValueError("Either variables or n_dims must be provided")

    def __call__(self, x: Float[Array, " n_dim"], data: PyTree) -> Float[Array, "1"]:
        return self.log_pdf(x, data)

    def print_parameters(self):
        print("LogPDF with variables:")
        for var in self.variables:
            print(var.name, var.continuous)
    
    def save_resource(self, path):
        raise NotImplementedError
    
    def load_resource(self, path):
        raise NotImplementedError

    def tree_flatten(self):
        children = ()
        aux_data = (self.log_pdf, self.variables)
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data[0], aux_data[1])