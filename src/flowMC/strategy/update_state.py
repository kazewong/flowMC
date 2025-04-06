from flowMC.resource.states import State
from flowMC.strategy.base import Strategy
from flowMC.resource.base import Resource
from jaxtyping import Array, Float, PRNGKeyArray


class UpdateState(Strategy):
    """Update a state resource in place."""

    def __init__(
        self, state_name: str, keys: list[str], values: list[int | bool | str]
    ):
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
