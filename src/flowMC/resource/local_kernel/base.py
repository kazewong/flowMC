from abc import abstractmethod
from typing import Callable

import jax
from jaxtyping import Array, Float, Int, PRNGKeyArray, PyTree
from flowMC.resource.base import Resource


@jax.tree_util.register_pytree_node_class
class ProposalBase(Resource):
    def __init__(
        self,
        **kwargs,
    ):
        """Initialize the sampler class."""
        self.kwargs = kwargs

    @abstractmethod
    def kernel(
        self,
        rng_key: PRNGKeyArray,
        log_pdf: Callable[[Float[Array, " n_dim"], PyTree], Float[Array, "1"]],
        position: Float[Array, "nstep  n_dim"],
        log_prob: Float[Array, "nstep 1"],
        data: PyTree,
    ) -> tuple[
        Float[Array, "nstep  n_dim"], Float[Array, "nstep 1"], Int[Array, "n_step 1"]
    ]:
        """Kernel for one step in the proposal cycle."""

    def tree_flatten(self):
        children = ()

        aux_data = {"kwargs": self.kwargs}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
