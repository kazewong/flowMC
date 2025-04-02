from abc import ABC
from typing import Callable

import jax
from jaxtyping import Array, Float, PRNGKeyArray

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.strategy.base import Strategy
from flowMC.strategy.global_tuning import LocalGlobalNFSample


class ResourceStrategyBundle(ABC):
    """Resource-Strategy Bundle is aim to be the highest level of abstraction in the
    flowMC library.

    It is a collection of resources and strategies that are used to perform a specific
    task.
    """

    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: list[str]


class RQSpline_MALA_Bundle(ResourceStrategyBundle):
    """A bundle that uses a Rational Quadratic Spline as a normalizing flow model and
    the Metropolis Adjusted Langevin Algorithm as a local sampler.

    This is the base algorithm described in https://www.pnas.org/doi/full/10.1073/pnas.2109420119



    """

    def __repr__(self):
        return "Local Global NF Sampling"

    def __init__(
        self,
        rng_key: PRNGKeyArray,
        n_chains: int,
        n_dims: int,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        n_local_steps: int,
        n_global_steps: int,
        n_training_loops: int,
        n_production_loops: int,
        n_epochs: int,
        mala_step_size: float = 1e-1,
        rq_spline_hidden_units: list[int] = [32, 32],
        rq_spline_n_bins: int = 8,
        rq_spline_n_layers: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        verbose: bool = False,
    ):
        n_training_steps = (
            n_local_steps * n_training_loops + n_global_steps * n_training_loops
        )
        n_production_steps = (
            n_local_steps * n_production_loops + n_global_steps * n_production_loops
        )
        n_total_epochs = n_training_loops * n_epochs

        positions_training = Buffer(
            "positions_training", (n_chains, n_training_steps, n_dims), 1
        )
        log_prob_training = Buffer("log_prob_training", (n_chains, n_training_steps), 1)
        local_accs_training = Buffer(
            "local_accs_training", (n_chains, n_training_steps), 1
        )
        global_accs_training = Buffer(
            "global_accs_training", (n_chains, n_training_steps), 1
        )
        loss_buffer = Buffer("loss_buffer", (n_total_epochs,), 0)

        position_production = Buffer(
            "positions_production", (n_chains, n_production_steps, n_dims), 1
        )
        log_prob_production = Buffer(
            "log_prob_production", (n_chains, n_production_steps), 1
        )
        local_accs_production = Buffer(
            "local_accs_production", (n_chains, n_production_steps), 1
        )
        global_accs_production = Buffer(
            "global_accs_production", (n_chains, n_production_steps), 1
        )

        local_sampler = MALA(step_size=mala_step_size)
        rng_key, subkey = jax.random.split(rng_key)
        model = MaskedCouplingRQSpline(
            n_dims, rq_spline_n_layers, rq_spline_hidden_units, rq_spline_n_bins, subkey
        )
        global_sampler = NFProposal(model)
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf = LogPDF(logpdf, n_dims=n_dims)

        self.resources = {
            "logpdf": logpdf,
            "positions_training": positions_training,
            "log_prob_training": log_prob_training,
            "local_accs_training": local_accs_training,
            "global_accs_training": global_accs_training,
            "loss_buffer": loss_buffer,
            "positions_production": position_production,
            "log_prob_production": log_prob_production,
            "local_accs_production": local_accs_production,
            "global_accs_production": global_accs_production,
            "local_sampler": local_sampler,
            "global_sampler": global_sampler,
            "model": model,
            "optimizer": optimizer,
        }

        self.strategies = {
            "training_sampler": LocalGlobalNFSample(
                "logpdf",
                "local_sampler",
                "global_sampler",
                ["positions_training", "log_prob_training", "local_accs_training"],
                ["model", "positions_training", "optimizer"],
                ["positions_training", "log_prob_training", "global_accs_training"],
                n_local_steps,
                n_global_steps,
                n_training_loops,
                n_epochs,
                loss_buffer_name="loss_buffer",
                batch_size=batch_size,
                n_max_examples=n_max_examples,
                training=True,
                verbose=verbose,
            ),
            "production_sampler": LocalGlobalNFSample(
                "logpdf",
                "local_sampler",
                "global_sampler",
                [
                    "positions_production",
                    "log_prob_production",
                    "local_accs_production",
                ],
                ["model", "positions_production", "optimizer"],
                [
                    "positions_production",
                    "log_prob_production",
                    "global_accs_production",
                ],
                n_local_steps,
                n_global_steps,
                n_production_loops,
                n_epochs,
                batch_size=batch_size,
                n_max_examples=n_max_examples,
                training=False,
                verbose=verbose,
            ),
        }
        self.strategy_order = ["training_sampler", "production_sampler"]
