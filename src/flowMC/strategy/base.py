from abc import abstractmethod
from flowMC.proposal.base import ProposalBase
from flowMC.proposal.NF_proposal import NFProposal
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
class Strategy:
    """
    Base class for strategies, which are basically wrapper blocks that modify the state of the sampler

    This is an abstract template that should not be directly used.
    
    """

    @abstractmethod
    def __name__(self):
        raise NotImplementedError

    @abstractmethod
    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(
        self,
        rng_key: PRNGKeyArray,
        local_sampler: ProposalBase,
        global_sampler: NFProposal,
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "n_chains n_dim"],
        ProposalBase,
        NFProposal,
        PyTree,
    ]:
        raise NotImplementedError