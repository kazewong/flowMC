from abc import ABC, abstractmethod
from typing import Callable

import jax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from tqdm import tqdm

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.strategy.base import Strategy
from flowMC.strategy.global_tuning import LocalGlobalNFSample


class ResourceStrategyBundle(ABC):
    """
    Resource-Strategy Bundle is aim to be the highest level of abstraction in the flowMC library.
    It is a collection of resources and strategies that are used to perform a specific task.
    """

    resources: dict[str, Resource]
    strategies: list[Strategy]


class LocalGlobalNFSamplingBundle(ResourceStrategyBundle):

    def __str__(self):
        return "Local Global NF Sampling"

    def __init__(
        self,
        rng_key: PRNGKeyArray,
        n_chains: int,
        n_steps: int,
        n_dim: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        local_sampler: ProposalBase,
        global_sampler: ProposalBase,
        local_names: list[str],
        training_names: list[str],
        global_names: list[str],
        n_local_steps: int,
        n_global_steps: int,
        n_training_loops: int,
        n_production_loops: int,
        n_epochs: int,
    ):

        positions = Buffer("positions", n_chains, n_steps, n_dim)
        log_prob = Buffer("log_prob", n_chains, n_steps, 1)
        local_accs = Buffer("local_accs", n_chains, n_steps, 1)
        global_accs = Buffer("global_accs", n_chains, n_steps, 1)
        local_sampler = MALA(step_size=1e-1)
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(n_dim, 3, [64, 64], 8, subkey)
        global_sampler = NFProposal(model)
        optimizer = Optimizer(model=model)

        self.resources = {
            "positions": positions,
            "log_prob": log_prob,
            "local_accs": local_accs,
            "global_accs": global_accs,
            "model": model,
            "optimizer": optimizer,
        }

        self.strategies = [
            LocalGlobalNFSample(
                logpdf,
                local_sampler,
                global_sampler,
                local_names,
                training_names,
                global_names,
                n_local_steps,
                n_global_steps,
                n_training_loops,
                n_production_loops,
                n_epochs,
            )
        ]
