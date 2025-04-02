from dataclasses import dataclass
from typing import Callable, Optional
from flowMC.resource.base import Resource
from jaxtyping import Array, Float, PyTree
import jax


@dataclass
class Variable:
    """A dataclass that holds the information of a variable in the log-pdf function.

    This main purpose of this class is to let the users name their variables,
    and specify whether they are continuous or not.
    """

    name: str
    continuous: bool


@jax.tree_util.register_pytree_node_class
class LogPDF(Resource):
    """A resource class that holds the log-pdf function.
    The main purpose of this class is to wrap the log-pdf function into the unified Resource interface.

    Args:
        log_pdf (Callable[[Float[Array, "n_dim"], PyTree], Float[Array, "1"]): The log-pdf function
        variables (list[Variable]): The list of variables in the log-pdf function
    """

    log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]]
    variables: list[Variable]

    @property
    def n_dims(self):
        return len(self.variables)

    def __repr__(self):
        return "LogPDF with " + str(self.n_dims) + " dimensions"

    def __init__(
        self,
        log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        variables: Optional[list[Variable]] = None,
        n_dims: Optional[int] = None,
    ):
        """
        Args:
            log_pdf (Callable[[Float[Array, "n_dim"], PyTree], Float[Array, "1"]): The log-pdf function
            variables (list[Variable], optional): The list of variables in the log-pdf function. Defaults to None. n_dims must be provided if this argument is None.
            n_dims (int, optional): The number of dimensions of the log-pdf function. Defaults to None. If variables is provided, this argument is ignored.
        """
        self.log_pdf = log_pdf
        if variables is None and n_dims is not None:
            self.variables = [Variable("x_" + str(i), True) for i in range(n_dims)]
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


@jax.tree_util.register_pytree_node_class
class TemperedPDF(LogPDF):

    log_prior: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]]

    def __init__(
        self,
        log_likelihood: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        log_prior: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        variables=None,
        n_dims=None,
        n_temps=5,
        max_temp=100,
    ):
        super().__init__(log_likelihood, variables, n_dims)
        self.log_prior = log_prior

    def __call__(self, x, data):
        return super().__call__(x, data)

    def tempered_log_pdf(self, temperatures, x, data):
        base_pdf = super().__call__(x, data)
        return (1.0 / temperatures) * base_pdf + self.log_prior(x, data)

    def tree_flatten(self):  # type: ignore
        children = ()
        aux_data = (self.log_pdf, self.log_prior, self.variables)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*aux_data, *children)
