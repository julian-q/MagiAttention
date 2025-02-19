from dataclasses import dataclass, field

import torch

from .enum import AttnOverlapMode, OverlapAlgorithm


@dataclass(frozen=True)
class OverlapConfig:
    enable: bool = True  # if False, turn off the multi-stage overlapping mode

    mode: AttnOverlapMode = AttnOverlapMode.STATIC

    degree: int | None = 1
    dynamic_max_degree: int | None = (
        8  # only used in dynamic mode, if None, then no limit
    )

    min_chunk_size: int = 512
    max_num_chunks: int = 64

    # TODO: use another non-trivial alg as default in the future
    alg: OverlapAlgorithm = OverlapAlgorithm.UNIFORM
    alg_kwargs: dict = field(default_factory=dict)

    calc_cost_factor: float = (
        1.0  # define: calc_cost = calc_cost_factor * calc_area (unit: μs)
    )
    comm_cost_factor: float = (
        1.0  # define: comm_cost = comm_cost_factor * comm_size (unit: μs)
    )

    def __post_init__(self):
        if not self.enable:
            # HACK: force auto-set other attrs to disable mso
            object.__setattr__(self, "mode", AttnOverlapMode.STATIC)
            object.__setattr__(self, "degree", 1)
            object.__setattr__(self, "max_num_chunks", 1)

        if self.mode is AttnOverlapMode.STATIC:
            assert self.degree is not None, (
                "When using static overlap mode, "
                f"the {self.degree=} should be set explicitly."
            )
            assert (
                self.degree <= self.max_num_chunks
            ), f"The {self.max_num_chunks=} should be no less than {self.degree=}."
        elif self.mode is AttnOverlapMode.DYNAMIC:
            assert self.degree is None, (
                "When using dynamic overlap mode, "
                f"the {self.degree=} should not be set."
            )
            assert self.dynamic_max_degree is None or (
                self.dynamic_max_degree > 0
            ), f"The {self.dynamic_max_degree=} should be greater than 0 if specified."

        assert (
            self.min_chunk_size > 0
        ), f"The {self.min_chunk_size=} should be greater than 0."
        assert (
            self.max_num_chunks > 0
        ), f"The {self.max_num_chunks=} should be greater than 0."
        assert (
            self.calc_cost_factor > 0.0 and self.comm_cost_factor > 0.0
        ), f"The {self.calc_cost_factor=} and {self.comm_cost_factor=} should be both greater than 0."


@dataclass(frozen=True)
class DistFlashAttnConfig:
    """
    静态config, 在程序初始化的时候就应该被定义
    """

    num_heads: int
    head_dim: int
    dtype: torch.dtype
    overlap_config: OverlapConfig
    deterministic: bool = False

    def __post_init__(self):
        pass
