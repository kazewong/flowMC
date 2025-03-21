from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp
from typing import Callable


class ParallelTempering(Strategy):
    """
    
    """


    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]

    def __init__(self):
        super().__init__()

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
        
        return rng_key, resources, final_position

    def _individal_step(self):
        raise NotImplementedError

    def _exchange(self):
        raise NotImplementedError