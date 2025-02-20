from dataclasses import dataclass

from zeus.meta.solver.dispatch_solver import (
    BSDispatchAlg,
    DispatchAlg,
    DispatchConfig,
    DPDispatchAlg,
    LBDispatchAlg,
    MinHeapDispatchAlg,
)
from zeus.meta.solver.overlap_solver import (
    GreedyOverlapAlg,
    OverlapAlg,
    OverlapConfig,
    UniformOverlapAlg,
)

__all__ = [
    "DistAttnConfig",
    "DispatchConfig",
    "DispatchAlg",
    "LBDispatchAlg",
    "DPDispatchAlg",
    "BSDispatchAlg",
    "MinHeapDispatchAlg",
    "OverlapConfig",
    "OverlapAlg",
    "UniformOverlapAlg",
    "GreedyOverlapAlg",
]


@dataclass(frozen=True)
class DistAttnConfig:
    """The overall config dataclass for dist-attn
    containing sub-configs for sub-modules to be assigned
    """

    dispatch_config: DispatchConfig = DispatchConfig()
    overlap_config: OverlapConfig = (
        OverlapConfig()
    )  # TODO: add distinct overlap config for fwd/bwd in the future
    deterministic: bool = False

    def __post_init__(self):
        assert (
            not self.deterministic
        ), "For now, deterministic mode is not supported by ffa."
