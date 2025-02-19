from dataclasses import dataclass

import torch


@dataclass
class AttnArg:
    q_ranges: list[tuple[int, int]]
    k_ranges: list[tuple[int, int]]
    is_causal_mapping: list[bool]

    # 用户不应该设置以下变量, 这些变量是__post_init__自动生成的
    max_seqlen_q: int = None  # type: ignore
    max_seqlen_k: int = None  # type: ignore
    skip_attn: bool = None  # type: ignore
    q_ranges_tensor: torch.Tensor = None  # type: ignore
    k_ranges_tensor: torch.Tensor = None  # type: ignore
    is_causal_mapping_tensor: torch.Tensor = None  # type: ignore

    total_area: int = 0

    def __post_init__(self):
        # shape check
        batch_size = len(self.q_ranges)
        assert len(self.q_ranges) == len(self.k_ranges) == len(self.is_causal_mapping)

        self.q_ranges_tensor = torch.tensor(
            self.q_ranges, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.k_ranges_tensor = torch.tensor(
            self.k_ranges, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.is_causal_mapping_tensor = torch.tensor(
            self.is_causal_mapping, dtype=torch.bool, device=torch.cuda.current_device()
        )

        if batch_size > 0:
            assert self.q_ranges_tensor.shape == torch.Size(
                [batch_size, 2]
            ), f"{self.q_ranges_tensor.shape=}, {batch_size=}"
            assert self.k_ranges_tensor.shape == torch.Size(
                [batch_size, 2]
            ), f"{self.k_ranges_tensor.shape=}, {batch_size=}"
            assert self.is_causal_mapping_tensor.shape == torch.Size(
                [batch_size]
            ), f"{self.is_causal_mapping_tensor.shape=}, {batch_size=}"
            self.skip_attn = False
            self.max_seqlen_q = max(
                q_range[1] - q_range[0] for q_range in self.q_ranges
            )
            self.max_seqlen_k = max(
                k_range[1] - k_range[0] for k_range in self.k_ranges
            )
        elif batch_size == 0:
            self.skip_attn = True
            self.max_seqlen_q = 0
            self.max_seqlen_k = 0
        else:
            raise ValueError(f"Invalid batch size: {batch_size}")

        # 检查每一个k_ranges的left < right
        for k_ranges in self.k_ranges:
            assert k_ranges[0] < k_ranges[1]

    def to_ffa_args(self) -> dict:
        return {
            "q_ranges": self.q_ranges_tensor,
            "k_ranges": self.k_ranges_tensor,
            "is_causal_mapping": self.is_causal_mapping_tensor,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
        }


@dataclass
class AttnCalcMeta:
    local_attn_arg: AttnArg
    remote_attn_args_list: list[AttnArg]

    @property
    def overlap_degree(self) -> int:
        return len(self.remote_attn_args_list)

    def __post_init__(self):
        pass
