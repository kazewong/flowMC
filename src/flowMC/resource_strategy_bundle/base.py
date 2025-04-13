from abc import ABC

from flowMC.resource.base import Resource
from flowMC.strategy.base import Strategy


class ResourceStrategyBundle(ABC):
    """Resource-Strategy Bundle is aim to be the highest level of abstraction in the
    flowMC library.

    It is a collection of resources and strategies that are used to perform a specific
    task.
    """

    resources: dict[str, Resource]
    strategies: dict[str, Strategy]
    strategy_order: list[str]
