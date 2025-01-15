# mypy: ignore-errors
from dataclasses import dataclass
from typing import List

import torch.distributed as dist

from ..dispatch.kv_transfer import KVTransferTable  # 不应该导入KVTransferTable
from ..meta.containers.bucket import AttnBucket
from .enum import AttnMaskType, AttnRole, AttnType
from .ranges import AttnRanges


@dataclass
class DispatchMeta:
    """The meta info of sequence dispatch for distributed attention"""

    attn_role: AttnRole
    attn_type: AttnType
    attn_mask_type: List[AttnMaskType]

    ranges: AttnRanges
    ranges_permed: AttnRanges

    batch_size: int
    total_seqlen: int

    cp_rank: int
    cp_size: int
    cp_group_nccl: dist.ProcessGroup
    cp_group_gloo: dist.ProcessGroup

    chunk_size: int
    num_chunks: int

    overlap_degree: int
    num_remote_tokens: int
    overlap_split_size_list: List[int]

    seqlens: List[int]
    seqlens_permed: List[int]
    seqlens_perm_idxs: List[int]
    seqlens_unperm_idxs: List[int]

    cu_seqlens: List[int]
    cu_seqlens_permed: List[int]

    partitions_permed: List[List[int]]
    partitions_perm_idxs: List[int]
    partitions_unperm_idxs: List[int]

    global_bucket: AttnBucket
    buckets_per_rank: List[AttnBucket]

    host_qk_ranges_global_per_rank: List[AttnRanges]
    host_qk_ranges_local_per_rank: List[AttnRanges]
    host_req_k_ranges_global_per_rank: List[AttnRanges]
    remote_k_ranges_global_per_rank: List[AttnRanges]
    remote_k_ranges_local_per_rank: List[AttnRanges]

    kv_transfer_table: KVTransferTable

    kv_input_split_size_list: List[int]
    kv_output_split_size_list: List[int]
    kv_dst_indices_list: List[List[int]]
    kv_src_index_list: List[int]

    local_attn_arg_q_ranges: AttnRanges
    local_attn_arg_k_ranges: AttnRanges
    local_attn_arg_is_causal_mapping: List[bool]
    local_attn_arg_max_seqlen_q: int
    local_attn_arg_max_seqlen_k: int

    remote_attn_args_q_ranges_list: List[AttnRanges]
    remote_attn_args_k_ranges_list: List[AttnRanges]
    remote_attn_args_is_causal_mapping_list: List[List[bool]]
    remote_attn_args_max_seqlen_q_list: List[int]
    remote_attn_args_max_seqlen_k_list: List[int]

    def __post_init__(self):
        assert len(self.kv_input_split_size_list) == len(
            self.kv_dst_indices_list
        ), f"The {len(self.kv_input_split_size_list)=} should be equal to {len(self.kv_dst_indices_list)=}."  # noqa

        assert len(self.kv_output_split_size_list) == len(
            self.kv_src_index_list
        ), f"The {len(self.kv_output_split_size_list)=} should be equal to {len(self.kv_src_index_list)=}."  # noqa

        assert (
            len(self.local_attn_arg_is_causal_mapping)
            == self.local_attn_arg_q_ranges.size
            == self.local_attn_arg_k_ranges.size
        ), (
            f"The {len(self.local_attn_arg_is_causal_mapping)=} should be equal to "
            f"{self.local_attn_arg_q_ranges.size=}, as well as {self.local_attn_arg_k_ranges.size=}."  # noqa
        )

    def __repr__(self, width: int = 30) -> str:
        """Customized __repr__ method for BaseConfig,
        displaying all fields with their values in alphabetical order.
        """
        class_name = self.__class__.__name__
        repr_str = f"{'*' * width}   {class_name}   {'*' * width}\n"
        title_len = len(repr_str) - 1

        field_names = sorted(self.__dataclass_fields__.keys())
        for field_name in field_names:
            field_value = getattr(self, field_name)
            if isinstance(field_value, str):
                field_value = repr(field_value)
            repr_str += f"{field_name}: {field_value}\n"

        repr_str += f"{'*' * title_len}\n"

        return repr_str
