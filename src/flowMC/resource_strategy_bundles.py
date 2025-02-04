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


class RQSpline_MALA_Bundle(ResourceStrategyBundle):

    def __str__(self):
        return "Local Global NF Sampling"

    def __init__(
        self,
        rng_key: PRNGKeyArray,
        n_chains: int,
        n_dim: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_local_steps: int,
        n_global_steps: int,
        n_training_loops: int,
        n_production_loops: int,
        n_epochs: int,
        rq_spline_hidden_units: list[int],
        rq_spline_n_bins: int,
        rq_spline_n_layers: int,
    ):

        n_training_steps = n_local_steps * n_training_loops
        n_production_steps = n_local_steps * n_production_loops

        positions_training = Buffer("positions_training", n_chains, n_training_steps, n_dim)
        log_prob_training = Buffer("log_prob_training", n_chains, n_training_steps, 1)
        local_accs_training = Buffer("local_accs_training", n_chains, n_training_steps, 1)
        global_accs_training = Buffer("global_accs_training", n_chains, n_training_steps, 1)

        position_production = Buffer("positions_production", n_chains, n_production_steps, n_dim)
        log_prob_production = Buffer("log_prob_production", n_chains, n_production_steps, 1)
        local_accs_production = Buffer("local_accs_production", n_chains, n_production_steps, 1)
        global_accs_production = Buffer("global_accs_production", n_chains, n_production_steps, 1)

        local_sampler = MALA(step_size=1e-1)
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(n_dim, rq_spline_n_layers, rq_spline_hidden_units, rq_spline_n_bins, subkey)
        global_sampler = NFProposal(model)
        optimizer = Optimizer(model=model)

        self.resources = {
            "positions_training": positions_training,
            "log_prob_training": log_prob_training,
            "local_accs_training": local_accs_training,
            "global_accs_training": global_accs_training,
            "positions_production": position_production,
            "log_prob_production": log_prob_production,
            "local_accs_production": local_accs_production,
            "global_accs_production": global_accs_production,
            "model": model,
            "optimizer": optimizer,
        }

        self.strategies = [
            LocalGlobalNFSample(
                logpdf,
                local_sampler,
                global_sampler,
                ["positions_training", "log_prob_training", "local_accs_training"],
                ["model", "positions_training", "optimizer"],
                ["positions_training", "log_prob_training", "global_accs_training"],
                n_local_steps,
                n_global_steps,
                n_training_loops,
                n_epochs,
                True
            ),
            LocalGlobalNFSample(
                logpdf,
                local_sampler,
                global_sampler,
                ["positions_production", "log_prob_production", "local_accs_production"],
                ["model", "positions_production", "optimizer"],
                ["positions_production", "log_prob_production", "global_accs_production"],
                n_local_steps,
                n_global_steps,
                n_production_loops,
                n_epochs,
                False
            ),
            
        ]
