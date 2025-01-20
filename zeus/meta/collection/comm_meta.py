from dataclasses import dataclass, field


@dataclass
class GroupCastCollectiveArg:
    input_split_size_list: list[int]
    output_split_size_list: list[int]
    dst_indices_list: list[list[int]]
    src_index_list: list[int]


@dataclass
class CommMeta:
    num_remote_tokens_per_overlap_stage: list[int] = field(
        metadata={"help": "num tokens per overlap stage"}
    )
    group_cast_collective_args_list: list[GroupCastCollectiveArg]
