from flowMC.resource.base import Resource
from typing import Callable, Self
from jaxtyping import Array, Float, PRNGKeyArray

class LogPDF(Resource):
    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
    ):
        """
        Initialize the sampler class
        """
        self.logpdf = logpdf

    def __call__(
        self,
        position: Float[Array, "nstep  n_dim"],
        data: dict,
    ) -> Float[Array, "nstep 1"]:
        """
        Kernel for one step in the proposal cycle.
        """
        return self.logpdf(position, data)
    
    def print_parameters(self):
        raise NotImplementedError

    def save_resource(self, path):
        raise NotImplementedError

    def load_resource(self, path) -> Self:
        raise NotImplementedError