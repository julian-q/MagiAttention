from dataclasses import dataclass

import torch

import dffa
from dffa.common.ranges import AttnRanges


@dataclass(repr=False)
class AttnArg:
    q_ranges: AttnRanges
    k_ranges: AttnRanges
    is_causal_mapping: list[bool]
    # REVIEW(xiaowu): Is shard_seqlen_q an appropriate name?
    shard_seqlen_q: int

    # total area of the attn arg, -1 means unknown
    total_area: int = -1

    # NOTE: 以下变量是__post_init__自动生成的，并作为ffa的args
    # max_seqlen_q: int
    # max_seqlen_k: int
    # skip_attn: bool
    # q_ranges_tensor: torch.Tensor
    # k_ranges_tensor: torch.Tensor
    # is_causal_mapping_tensor: torch.Tensor
    # out_zero_fill_ranges: list[tuple[int, int]]

    def __post_init__(self):
        # shape check
        assert len(self.q_ranges) == len(self.k_ranges) == len(self.is_causal_mapping)
        # filter out k_ranges with seqlen == 0
        self.q_ranges = AttnRanges.from_ranges(
            [
                q_range
                for q_range, k_range in zip(self.q_ranges, self.k_ranges)
                if k_range.seqlen > 0
            ]
        )
        self.k_ranges = AttnRanges.from_ranges(
            [
                k_range
                for q_range, k_range in zip(self.q_ranges, self.k_ranges)
                if k_range.seqlen > 0
            ]
        )
        self.is_causal_mapping = [
            is_causal_mapping
            for q_range, k_range, is_causal_mapping in zip(
                self.q_ranges, self.k_ranges, self.is_causal_mapping
            )
            if k_range.seqlen > 0
        ]

        batch_size = len(self.q_ranges)

        # init tensors
        self.q_ranges_tensor = self.q_ranges.to_tensor(
            device=torch.cuda.current_device()
        )
        self.k_ranges_tensor = self.k_ranges.to_tensor(
            device=torch.cuda.current_device()
        )
        self.is_causal_mapping_tensor = torch.tensor(
            self.is_causal_mapping, dtype=torch.bool, device=torch.cuda.current_device()
        )

        # sanity check
        if dffa.is_sanity_check_enable():
            # 检查每一个k_ranges的left < right
            for k_ranges in self.k_ranges:
                assert k_ranges.start < k_ranges.end

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

        # init max seqlen
        if batch_size > 0:
            self.skip_attn = False
            self.max_seqlen_q = max(
                q_range.end - q_range.start for q_range in self.q_ranges
            )
            self.max_seqlen_k = max(
                k_range.end - k_range.start for k_range in self.k_ranges
            )
        elif batch_size == 0:  # no calc needed
            self.skip_attn = True
            self.max_seqlen_q = 0
            self.max_seqlen_k = 0
        else:
            raise ValueError(f"Invalid batch size: {batch_size}")

        # init ffa args dict
        self.ffa_args_dict = {
            "q_ranges": self.q_ranges_tensor,
            "k_ranges": self.k_ranges_tensor,
            "is_causal_mapping": self.is_causal_mapping_tensor,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
        }

        # init out zero-fill ranges
        # TODO: put this logic into kernel
        start, end = 0, self.shard_seqlen_q
        self.out_zero_fill_ranges: list[tuple[int, int]] = []
        for q_range in self.q_ranges:
            if start < q_range.start:
                self.out_zero_fill_ranges.append((start, q_range.start))
            start = q_range.end
        if start < end:
            self.out_zero_fill_ranges.append((start, end))

        self.out_zero_fill_ranges = (
            AttnRanges.from_ranges(self.out_zero_fill_ranges).merge().to_naive_ranges()
        )

    def to_ffa_args(self) -> dict:
        return self.ffa_args_dict

    def __repr__(self) -> str:
        return (
            f"AttnArg(q_ranges={self.q_ranges}, k_ranges={self.k_ranges}, is_causal_mapping={self.is_causal_mapping}, "
            f"shard_seqlen_q={self.shard_seqlen_q}, total_area={self.total_area}, "
            f"max_seqlen_q={self.max_seqlen_q}, max_seqlen_k={self.max_seqlen_k}, skip_attn={self.skip_attn}, "
            f"out_zero_fill_ranges={self.out_zero_fill_ranges})"
        )


@dataclass
class AttnCalcMeta:
    local_attn_arg: AttnArg
    remote_attn_args_list: list[AttnArg]

    @property
    def overlap_degree(self) -> int:
        return len(self.remote_attn_args_list)

    def __post_init__(self):
        pass
