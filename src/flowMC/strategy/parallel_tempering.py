from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF
from flowMC.strategy.base import Strategy
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp


class ParallelTempering(Strategy):
    """
    
    """

    n_temperature: int


    def __init__(self):
        raise NotImplementedError

    def __call__(
        self,
        rng_key: PRNGKeyArray,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dims"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        
        return rng_key, resources, final_position

    def _individal_step(self, positions: Float[Array, "n_temps n_dims"], logpdf: LogPDF):
        raise NotImplementedError

    def _exchange(self, positions: Float[Array, "n_temps n_dims"], logpdf: LogPDF):
        log_prob_temps = jax.vmap(logpdf, in_axes=(0, None))(positions, None) * evaluate_temps(temps_pdf, positions, temperatures)