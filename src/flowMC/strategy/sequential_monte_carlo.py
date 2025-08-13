from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable

class SequentialMonteCarlo(Resource):
    def __init__(self):
        raise NotImplementedError
        
    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        raise NotImplementedError