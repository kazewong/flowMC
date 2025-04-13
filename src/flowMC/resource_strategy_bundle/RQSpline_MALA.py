from typing import Callable

import jax
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx

from flowMC.resource.base import Resource
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.local_kernel.MALA import MALA
from flowMC.resource.nf_model.NF_proposal import NFProposal
from flowMC.resource.nf_model.rqSpline import MaskedCouplingRQSpline
from flowMC.resource.optimizer import Optimizer
from flowMC.strategy.lambda_function import Lambda
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.train_model import TrainModel
from flowMC.strategy.update_state import UpdateState
from flowMC.resource_strategy_bundle.base import ResourceStrategyBundle


class RQSpline_MALA_Bundle(ResourceStrategyBundle):
    """A bundle that uses a Rational Quadratic Spline as a normalizing flow model and
    the Metropolis Adjusted Langevin Algorithm as a local sampler.

    This is the base algorithm described in https://www.pnas.org/doi/full/10.1073/pnas.2109420119

    """

    def __repr__(self):
        return "RQSpline_MALA Bundle"

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
        local_thinning: int = 1,
        global_thinning: int = 1,
        n_NFproposal_batch_size: int = 10000,
        verbose: bool = False,
    ):
        n_training_steps = (
            n_local_steps // local_thinning * n_training_loops
            + n_global_steps // global_thinning * n_training_loops
        )
        n_production_steps = (
            n_local_steps // local_thinning * n_production_loops
            + n_global_steps // global_thinning * n_production_loops
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
        global_sampler = NFProposal(
            model, n_NFproposal_batch_size=n_NFproposal_batch_size
        )
        optimizer = Optimizer(model=model, learning_rate=learning_rate)
        logpdf = LogPDF(logpdf, n_dims=n_dims)

        sampler_state = State(
            {
                "target_positions": "positions_training",
                "target_log_prob": "log_prob_training",
                "target_local_accs": "local_accs_training",
                "target_global_accs": "global_accs_training",
                "training": True,
            },
            name="sampler_state",
        )

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
            "sampler_state": sampler_state,
        }

        local_stepper = TakeSerialSteps(
            "logpdf",
            "local_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_local_accs"],
            n_local_steps,
            thinning=local_thinning,
            verbose=verbose,
        )

        global_stepper = TakeGroupSteps(
            "logpdf",
            "global_sampler",
            "sampler_state",
            ["target_positions", "target_log_prob", "target_global_accs"],
            n_global_steps,
            thinning=global_thinning,
            verbose=verbose,
        )

        model_trainer = TrainModel(
            "model",
            "positions_training",
            "optimizer",
            loss_buffer_name="loss_buffer",
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            verbose=verbose,
        )

        update_state = UpdateState(
            "sampler_state",
            [
                "target_positions",
                "target_log_prob",
                "target_local_accs",
                "target_global_accs",
                "training",
            ],
            [
                "positions_production",
                "log_prob_production",
                "local_accs_production",
                "global_accs_production",
                False,
            ],
        )

        def reset_steppers(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Reset the steppers to the initial position."""
            local_stepper.set_current_position(0)
            global_stepper.set_current_position(0)
            return rng_key, resources, initial_position

        reset_steppers_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: reset_steppers(
                rng_key, resources, initial_position, data
            )
        )

        update_global_step = Lambda(
            lambda rng_key, resources, initial_position, data: global_stepper.set_current_position(
                local_stepper.current_position
            )
        )
        update_local_step = Lambda(
            lambda rng_key, resources, initial_position, data: local_stepper.set_current_position(
                global_stepper.current_position
            )
        )

        def update_model(
            rng_key: PRNGKeyArray,
            resources: dict[str, Resource],
            initial_position: Float[Array, "n_chains n_dim"],
            data: dict,
        ) -> tuple[
            PRNGKeyArray,
            dict[str, Resource],
            Float[Array, "n_chains n_dim"],
        ]:
            """Update the model."""
            model = resources["model"]
            resources["global_sampler"] = eqx.tree_at(
                lambda x: x.model,
                resources["global_sampler"],
                model,
            )
            return rng_key, resources, initial_position

        update_model_lambda = Lambda(
            lambda rng_key, resources, initial_position, data: update_model(
                rng_key, resources, initial_position, data
            )
        )

        self.strategies = {
            "local_stepper": local_stepper,
            "global_stepper": global_stepper,
            "model_trainer": model_trainer,
            "update_state": update_state,
            "update_global_step": update_global_step,
            "update_local_step": update_local_step,
            "reset_steppers": reset_steppers_lambda,
            "update_model": update_model_lambda,
        }

        training_phase = [
            "local_stepper",
            "update_global_step",
            "model_trainer",
            "update_model",
            "global_stepper",
            "update_local_step",
        ]
        production_phase = [
            "local_stepper",
            "update_global_step",
            "global_stepper",
            "update_local_step",
        ]
        strategy_order = []
        for _ in range(n_training_loops):
            strategy_order.extend(training_phase)

        strategy_order.append("reset_steppers")
        strategy_order.append("update_state")
        for _ in range(n_production_loops):
            strategy_order.extend(production_phase)

        self.strategy_order = strategy_order
