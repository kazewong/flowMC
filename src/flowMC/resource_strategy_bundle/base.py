import enum
from abc import ABC

from flowMC.resource.base import Resource
from flowMC.strategy.base import Strategy


class Phase(enum.Enum):
    """Enumeration for the different phases of the resource-strategy bundle."""

    INITIALIZATION = "initialization"
    INTERMEDIATE = "intermediate"
    PRODUCTION = "production"
    TRAINING = "training"


class ResourceStrategyBundle(ABC):
    """Resource-Strategy Bundle is aim to be the highest level of abstraction in the
    flowMC library.

    It is a collection of resources and strategies that are used to perform a specific
    task.
    """

    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: list[tuple[str, str]]
