import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree
from tqdm import tqdm

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource.local_kernel.base import ProposalBase
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.train_model import TrainModel
from typing import Callable

class LocalGlobalNFSample(Strategy):

    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    local_stepper: TakeSerialSteps
    global_stepper: TakeGroupSteps
    training_stepper: TrainModel

    n_loops: int

    def __str__(self):
        return "Local Global NF Sampling"

    def __init__(
        self,
        logpdf: Callable[[Float[Array, " n_dim"], dict], Float],
        local_sampler: ProposalBase,
        global_sampler: ProposalBase,
        local_names: list[str],
        training_names: list[str],
        global_names: list[str],
        n_local_steps: int,
        n_global_steps: int,
        n_loops: int,
        n_epochs: int,
        training: bool = True,
    ):
        self.logpdf = logpdf
        self.local_stepper = TakeSerialSteps(
            logpdf,
            local_sampler,
            local_names,
            n_local_steps,
        )
        self.global_stepper = TakeGroupSteps(
            logpdf,
            global_sampler,
            global_names,
            n_global_steps,
        )
        self.training_stepper = TrainModel(
            training_names[0],
            training_names[1],
            training_names[2],
            n_epochs,
        )
        self.n_loops = n_loops
        self.training = training

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
        for _ in tqdm(
            range(self.n_loops),
            desc="Tuning Phase",
        ):
            rng_key, resources, initial_position = self.local_stepper(
                rng_key,
                resources,
                initial_position,
                data,
            )
            self.global_stepper.set_current_position(self.local_stepper.current_position)
            if self.training is True:
                rng_key, resources, initial_position = self.training_stepper(
                    rng_key,
                    resources,
                    initial_position,
                    data,
                )
            rng_key, resources, initial_position = self.global_stepper(
                rng_key,
                resources,
                initial_position,
                data,
            )
            self.local_stepper.set_current_position(self.global_stepper.current_position)

        return rng_key, resources, initial_position

