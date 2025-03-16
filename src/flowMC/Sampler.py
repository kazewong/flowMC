import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from flowMC.resource_strategy_bundles import ResourceStrategyBundle


class Sampler:
    """Top level API that the users primarily interact with.

    Args:
        n_dim (int): Dimension of the parameter space.
        n_chains (int): Number of chains to sample.
        rng_key (PRNGKeyArray): Jax PRNGKey.
        logpdf (Callable[[Float[Array, "n_dim"], dict], Float):
            Log probability function.
        resources (dict[str, Resource]): Resources to be used by the sampler.
        strategies (list[Strategy]): List of strategies to be used by the sampler.
        verbose (bool): Whether to print out progress. Defaults to False.
        logging (bool): Whether to log the progress. Defaults to True.
        outdir (str): Directory to save the logs. Defaults to "./outdir/".
    """

    # Essential parameters
    n_dim: int
    n_chains: int
    rng_key: PRNGKeyArray
    resources: dict[str, Resource]
    strategies: list[Strategy]

    # Logging hyperparameters
    verbose: bool = False
    logging: bool = True
    outdir: str = "./outdir/"

    def __init__(
        self,
        n_dim: int,
        n_chains: int,
        rng_key: PRNGKeyArray,
        resources: None | dict[str, Resource] = None,
        strategies: None | list[Strategy] = None,
        resource_strategy_bundles: None | ResourceStrategyBundle = None,
        **kwargs,
    ):
        # Copying input into the model

        self.n_dim = n_dim
        self.n_chains = n_chains
        self.rng_key = rng_key

        if resources is not None and strategies is not None:
            print(
                "Resources and strategies provided. Ignoring resource strategy bundles."
            )
            self.resources = resources
            self.strategies = strategies
        else:
            print(
                "Resources or strategies not provided. Using resource strategy bundles."
            )
            if resource_strategy_bundles is None:
                raise ValueError(
                    "Resource strategy bundles not provided."
                    "Please provide either resources and strategies or resource strategy bundles."
                )
            self.resources = resource_strategy_bundles.resources
            self.strategies = resource_strategy_bundles.strategies

        # Set and override any given hyperparameters
        class_keys = list(self.__class__.__dict__.keys())
        for key, value in kwargs.items():
            if key in class_keys:
                if not key.startswith("__"):
                    setattr(self, key, value)

    def sample(self, initial_position: Float[Array, "n_chains n_dim"], data: dict):
        """Sample from the posterior using the local sampler.

        Args:
            initial_position (Device Array): Initial position.
            data (dict): Data to be used by the likelihood functions
        """

        initial_position = jnp.atleast_2d(initial_position)  # type: ignore
        rng_key = self.rng_key
        last_step = initial_position
        for strategy in self.strategies:
            (
                rng_key,
                self.resources,
                last_step,
            ) = strategy(rng_key, self.resources, last_step, data)

    # TODO: Implement quick access and summary functions that operates on buffer

    def serialize(self):
        """Serialize the sampler object."""
        raise NotImplementedError

    def deserialize(self):
        """Deserialize the sampler object."""
        raise NotImplementedError
