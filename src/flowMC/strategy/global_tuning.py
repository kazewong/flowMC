from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.strategy.take_steps import TakeSerialSteps, TakeGroupSteps
from flowMC.strategy.train_model import TrainModel


class LocalGlobalNFSample(Strategy):
    n_loops: int

    def __repr__(self):
        return "Local Global NF Sampling"

    def __init__(
        self,
        logpdf_name: str,
        local_kernel_name: str,
        global_kernel_name: str,
        local_buffers_names: list[str],
        training_buffers_names: list[str],
        global_buffers_names: list[str],
        n_local_steps: int,
        n_global_steps: int,
        n_loops: int,
        n_epochs: int,
        loss_buffer_name: str = "",
        batch_size: int = 10000,
        n_max_examples: int = 10000,
        training: bool = True,
        verbose: bool = False,
    ):
        self.local_stepper = TakeSerialSteps(
            logpdf_name,
            local_kernel_name,
            local_buffers_names,
            n_local_steps,
            verbose=verbose,
        )
        self.global_stepper = TakeGroupSteps(
            logpdf_name,
            global_kernel_name,
            global_buffers_names,
            n_global_steps,
            verbose=verbose,
        )
        self.training_stepper = TrainModel(
            training_buffers_names[0],
            training_buffers_names[1],
            training_buffers_names[2],
            loss_buffer_name=loss_buffer_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_max_examples=n_max_examples,
            verbose=verbose,
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
        if self.training is True:
            iterator = tqdm(
                range(self.n_loops),
                desc="Tuning Phase",
            )
        else:
            iterator = tqdm(
                range(self.n_loops),
                desc="Sampling Phase",
            )

        for _ in iterator:
            rng_key, resources, initial_position = self.local_stepper(
                rng_key,
                resources,
                initial_position,
                data,
            )
            self.global_stepper.set_current_position(
                self.local_stepper.current_position
            )
            if self.training is True:
                rng_key, resources, initial_position = self.training_stepper(
                    rng_key,
                    resources,
                    initial_position,
                    data,
                )
                resources["global_sampler"].model = resources["model"]  # type: ignore
            rng_key, resources, initial_position = self.global_stepper(
                rng_key,
                resources,
                initial_position,
                data,
            )
            self.local_stepper.set_current_position(
                self.global_stepper.current_position
            )

        return rng_key, resources, initial_position
