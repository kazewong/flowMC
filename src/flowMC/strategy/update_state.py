from flowMC.resource.states import State
from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from jaxtyping import Array, Float, PRNGKeyArray


class UpdateState(Strategy):
    """Update a state resource in place.

    This strategy is meant to be used to update the state not too frequently.
    If you are looking for an option that iterates over some parameters,
    say the paramters of a neural network, you should write a custom strategy
    that does that.
    """

    def __init__(
        self, state_name: str, keys: list[str], values: list[int | bool | str]
    ):
        """Initialize the update state strategy.

        Args:
            state_name (str): The name of the state resource to update.
            keys (list[str]): The keys to update in the state resource.
            values (list[int | bool | str]): The values to update in the state resource.
        """
        self.state_name = state_name
        self.keys = keys
        self.values = values

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
        """Update the state resource in place."""
        assert isinstance(
            state := resources[self.state_name], State
        ), f"Resource {self.state_name} is not a State resource."

        state.update(self.keys, self.values)
        return rng_key, resources, initial_position
