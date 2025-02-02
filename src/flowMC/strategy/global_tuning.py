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

class GlobalTuning(Strategy):

    logpdf: Callable[[Float[Array, " n_dim"], dict], Float]
    local_stepper: TakeSerialSteps
    global_stepper: TakeGroupSteps
    training_stepper: TrainModel

    def __str__(self):
        return "GlobalTuning"

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
        n_loop: int,
        n_epochs: int,
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
        self.n_loop = n_loop

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
            range(self.n_loop),
            desc="Global Tuning",
        ):
            rng_key, resources, initial_position = self.local_stepper(
                rng_key,
                resources,
                initial_position,
                data,
            )
            self.global_stepper.set_current_position(self.local_stepper.current_position)
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



# class GlobalSampling(Strategy):

#     n_dim: int
#     n_chains: int
#     n_local_steps: int
#     n_global_steps: int
#     n_loop: int
#     output_thinning: int
#     verbose: bool

#     @property
#     def __name__(self):
#         return "GlobalSampling"

#     def __init__(
#         self,
#         **kwargs,
#     ):
#         class_keys = list(self.__class__.__annotations__.keys())
#         for key, value in kwargs.items():
#             if key in class_keys:
#                 if not key.startswith("__"):
#                     setattr(self, key, value)

#     def __call__(
#         self,
#         rng_key: PRNGKeyArray,
#         local_sampler: ProposalBase,
#         global_sampler: NFProposal,
#         initial_position: Float[Array, "n_chains n_dim"],
#         data: dict,
#     ) -> tuple[
#         PRNGKeyArray,
#         Float[Array, "n_chains n_dim"],
#         ProposalBase,
#         NFProposal,
#         PyTree,
#     ]:

#         summary = {}
#         summary["chains"] = jnp.empty((self.n_chains, 0, self.n_dim))
#         summary["log_prob"] = jnp.empty((self.n_chains, 0))
#         summary["local_accs"] = jnp.empty((self.n_chains, 0))
#         summary["global_accs"] = jnp.empty((self.n_chains, 0))

#         current_position = initial_position
#         for _ in tqdm(
#             range(self.n_loop),
#             desc="Global Sampling",
#         ):
#             rng_key, rng_keys_mcmc = jax.random.split(rng_key)
#             rng_keys_mcmc = jax.random.split(rng_keys_mcmc, self.n_chains)
#             (
#                 rng_keys_mcmc,
#                 positions,
#                 log_prob,
#                 local_acceptance,
#             ) = local_sampler.sample(
#                 rng_keys_mcmc,
#                 self.n_local_steps,
#                 current_position,
#                 data,
#                 verbose=self.verbose,
#             )

#             summary["chains"] = jnp.append(
#                 summary["chains"],
#                 positions[:, :: self.output_thinning],
#                 axis=1,
#             )
#             summary["log_prob"] = jnp.append(
#                 summary["log_prob"],
#                 log_prob[:, :: self.output_thinning],
#                 axis=1,
#             )

#             summary["local_accs"] = jnp.append(
#                 summary["local_accs"],
#                 local_acceptance[:, 1 :: self.output_thinning],
#                 axis=1,
#             )

#             current_position = summary["chains"][:, -1]

#             rng_key, rng_keys_nf = jax.random.split(rng_key)
#             (
#                 rng_keys_nf,
#                 nf_chain,
#                 log_prob,
#                 global_acceptance,
#             ) = global_sampler.sample(
#                 rng_keys_nf,
#                 self.n_global_steps,
#                 positions[:, -1],
#                 data,
#                 verbose=self.verbose,
#             )

#             summary["chains"] = jnp.append(
#                 summary["chains"],
#                 nf_chain[:, :: self.output_thinning],
#                 axis=1,
#             )
#             summary["log_prob"] = jnp.append(
#                 summary["log_prob"],
#                 log_prob[:, :: self.output_thinning],
#                 axis=1,
#             )

#             summary["global_accs"] = jnp.append(
#                 summary["global_accs"],
#                 global_acceptance[:, 1 :: self.output_thinning],
#                 axis=1,
#             )

#             current_position = summary["chains"][:, -1]

#         return rng_key, summary['chains'][:, -1], local_sampler, global_sampler, summary
