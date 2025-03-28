from dataclasses import dataclass

from dffa.comm.primitive.utils import (
    _calc_group_cast_a2a_input_meta_args,
    _calc_group_cast_a2a_output_meta_args,
    _calc_group_reduce_a2a_input_meta_args,
    _calc_group_reduce_a2a_output_meta_args,
)


@dataclass
class GroupCollectiveArg:
    """The arg dataclass for group cast/reduce collective ops"""

    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]
    world_size: int

    # NOTE: 以下变量是__post_init__自动生成的，并作为group collective to a2a的meta args
    # group_cast_args_dict_kv_packed: dict
    # group_reduce_args_dict_kv_packed: dict

    def __post_init__(self):
        # shape check
        assert len(self.input_split_size_list) == len(self.dst_indices_list), (
            f"Got inconsistent input split size: "
            f"{len(self.input_split_size_list)=} and "
            f"{len(self.dst_indices_list)=}."
        )
        assert len(self.output_split_size_list) == len(self.src_index_list), (
            f"Got inconsistent output split size: "
            f"{len(self.output_split_size_list)=} and "
            f"{len(self.src_index_list)=}."
        )

        # -------   group cast args dict for packed kv  ------- #

        self.group_cast_args_dict_kv_packed = {
            k: v * 2
            for k, v in {
                "input_split_size_list": self.input_split_size_list,
                "output_split_size_list": self.output_split_size_list,
                "dst_indices_list": self.dst_indices_list,
                "src_index_list": self.src_index_list,
            }.items()
        }

        (
            self.group_cast_args_dict_kv_packed["a2a_input_split_size"],
            self.group_cast_args_dict_kv_packed["perm_before_a2a_kwargs"],
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.group_cast_args_dict_kv_packed[
                "input_split_size_list"
            ],
            dst_indices_list=self.group_cast_args_dict_kv_packed["dst_indices_list"],
            world_size=self.world_size,
            device="cuda",
        )

        (
            self.group_cast_args_dict_kv_packed["a2a_output_split_size"],
            self.group_cast_args_dict_kv_packed["unperm_after_a2a_kwargs"],
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.group_cast_args_dict_kv_packed[
                "output_split_size_list"
            ],
            src_index_list=self.group_cast_args_dict_kv_packed["src_index_list"],
            world_size=self.world_size,
            device="cuda",
        )

        # -------   group reduce args dict for packed kv  ------- #

        self.group_reduce_args_dict_kv_packed = {
            k: v * 2
            for k, v in {
                "input_split_size_list": self.output_split_size_list,
                "output_split_size_list": self.input_split_size_list,
                "dst_index_list": self.src_index_list,
                "src_indices_list": self.dst_indices_list,
            }.items()
        }

        (
            self.group_reduce_args_dict_kv_packed["a2a_input_size_ranges_with_rank"],
            self.group_reduce_args_dict_kv_packed["a2a_input_split_size"],
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=self.group_reduce_args_dict_kv_packed[
                "input_split_size_list"
            ],
            dst_index_list=self.group_reduce_args_dict_kv_packed["dst_index_list"],
            world_size=self.world_size,
        )

        (
            self.group_reduce_args_dict_kv_packed["a2a_output_split_size"],
            self.group_reduce_args_dict_kv_packed["a2a_output_reduce_ranges_list"],
            self.group_reduce_args_dict_kv_packed["output_size_ranges"],
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=self.group_reduce_args_dict_kv_packed[
                "output_split_size_list"
            ],
            src_indices_list=self.group_reduce_args_dict_kv_packed["src_indices_list"],
            world_size=self.world_size,
        )

    def to_group_cast_args(self) -> dict:
        return self.group_cast_args_dict_kv_packed

    def to_group_reduce_args(self) -> dict:
        return self.group_reduce_args_dict_kv_packed


@dataclass
class CommMeta:
    num_remote_tokens_per_stage: list[int]
    group_collective_args_list: list[GroupCollectiveArg]

    @property
    def overlap_degree(self) -> int:
        return len(self.num_remote_tokens_per_stage)

    def __post_init__(self):
        assert len(self.num_remote_tokens_per_stage) == len(
            self.group_collective_args_list
        ), (
            f"Got inconsistent overlap degree: "
            f"{len(self.num_remote_tokens_per_stage)=} and "
            f"{len(self.group_collective_args_list)=}."
        )
