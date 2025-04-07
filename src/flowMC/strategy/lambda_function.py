from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable


class Lambda(Strategy):
    """A strategy that applies a function to the resources.

    This should be used for simple functions or calling methods from
    a class.
    If you find yourself writing a Lambda strategy that is more than a few lines
    long, consider writing a custom strategy instead.

    """

    def __init__(self, lambda_function: Callable):
        """Initialize the lambda strategy.

        Args:
            lambda: A callable that takes a resource and applies the lambda function.
        """
        self.lambda_function = lambda_function

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
        self.lambda_function(rng_key, resources, initial_position, data)
        return rng_key, resources, initial_position
