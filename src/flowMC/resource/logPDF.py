from dataclasses import dataclass
from typing import Callable
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

    """
    
    """

    log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]]
    variables: list[Variable]
    data: PyTree

    @property
    def n_dim(self):
        return len(self.variables)

    def __repr__(self):
        return super().__repr__()
    
    def __init__(self, log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]], variables: PyTree, data: PyTree):
        self.log_pdf = log_pdf
        self.variables = variables
        self.data = data

    def __call__(self, x: Float[Array, " n_dim"]) -> Float[Array, "1"]:
        return self.log_pdf(x, self.data)

    def tree_flatten(self):
        children = (self.log_pdf, self.variables, self.data)
        aux_data = {}
        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)