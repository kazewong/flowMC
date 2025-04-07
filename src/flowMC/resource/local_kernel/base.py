from abc import abstractmethod
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from flowMC.resource.base import Resource
from flowMC.resource.logPDF import LogPDF
from typing import Callable


class ProposalBase(eqx.Module, Resource):
    @abstractmethod
    def __init__(
        self,
    ):
        """Initialize the sampler class."""

    @abstractmethod
    def kernel(
        self,
        rng_key: PRNGKeyArray,
        position: Float[Array, "nstep  n_dim"],
        log_prob: Float[Array, "nstep 1"],
        logpdf: LogPDF | Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        data: PyTree,
    ) -> tuple[
        Float[Array, "nstep  n_dim"], Float[Array, "nstep 1"], Int[Array, "n_step 1"]
    ]:
        """Kernel for one step in the proposal cycle."""
