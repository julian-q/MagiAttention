from dataclasses import dataclass
from enum import Enum

import torch.nn as nn


class OverlapAlgorithm(Enum):
    """This enum is used to specify the algorithm for multi-stage overlapping"""

    UNIFORM = "uniform"
    GREEDY = "greedy"


@dataclass
class StageCost:
    comm_cost: float = 0.0
    calc_cost: float = 0.0


class OverlapSolver(nn.Module):
    """The implementation of the algorithms for multi-stage overlapping specified by `alg`."""

    def __init__(
        self,
        alg: OverlapAlgorithm = OverlapAlgorithm.GREEDY,
    ) -> None:
        super().__init__()

        self.alg = alg

        self.solve_func = {
            OverlapAlgorithm.UNIFORM: self._solve_with_uniform,
            OverlapAlgorithm.GREEDY: self._solve_with_greedy,
        }[self.alg]

        # return values
        self.partitions: list[list[int]] = None  # type: ignore

    def solve(
        self,
        costs: list[StageCost],
        num_stages: int | None = None,
        **kwargs,
    ) -> list[list[int]]:
        self.solve_func(costs, num_stages, **kwargs)

        return self.partitions

    def _solve_with_uniform(
        self, costs: list[StageCost], num_stages: int | None = None, **kwargs
    ) -> None:
        pass

    def _solve_with_greedy(
        self, costs: list[StageCost], num_stages: int | None = None, **kwargs
    ) -> None:
        pass
