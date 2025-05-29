from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF
from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource.nf_model.base import NFModel
from jaxtyping import Array, Float, PRNGKeyArray
import jax
import jax.numpy as jnp
import logging


class NeuralImportanceSampling(Strategy):
    """A strategy for importance sampling in Monte Carlo simulations."""

    logpdf_name: str
    model_name: str
    target_buffer_name: str
    target_weight_name: str
    n_steps: int

    def __repr__(self) -> str:
        return "NeuralImportanceSampling"

    def __init__(
        self,
        logpdf_name: str,
        model_name: str,
        target_buffer_name: str,
        target_weight_name: str,
        n_steps: int,
    ):
        """Initialize the importance sampling strategy.

        Args:
            logpdf: A callable that computes the log probability density function.
            proposal: A neural flow model to use as the proposal distribution.
            n_steps: Number of steps for the importance sampling.
        """

        self.logpdf_name = logpdf_name
        self.model_name = model_name
        self.n_steps = n_steps
        self.target_buffer_name = target_buffer_name
        self.target_weight_name = target_weight_name

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
        rng_key, subkey = jax.random.split(rng_key)
        assert isinstance(
            target_buffer := resources[self.target_buffer_name], Buffer
        ), (
            f"Expected {self.target_buffer_name} to be a Buffer, got"
            f" {type(target_buffer)}"
        )
        assert isinstance(
            model := resources[self.model_name], NFModel
        ), f"Expected {self.model_name} to be a NFModel, got {type(model)}"
        assert isinstance(target_weight := resources[self.target_weight_name], Buffer), (
            f"Expected {self.target_weight_name} to be a Buffer, got"
            f" {type(target_weight)}"
        )
        assert isinstance(logpdf := resources[self.logpdf_name], LogPDF), (
            f"Expected {self.logpdf_name} to be a LogPDF, got"
            f" {type(logpdf)}"
        )

        proposed_position = model.sample(subkey, self.n_steps)
        nf_pdf = model.log_prob(proposed_position)
        target_pdf = logpdf(proposed_position, data)
        weights = target_pdf - nf_pdf
        if jnp.any(jnp.isnan(weights)):
            logging.warning(
                "NaN values found in weights. This may indicate an issue with the model or logpdf function."
            )
        target_weight.update_buffer(weights)
        target_buffer.update_buffer(proposed_position)

        return rng_key, resources, initial_position
