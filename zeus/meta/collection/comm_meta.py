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
        default_factory=list, metadata={"help": "num tokens per overlap stage"}
    )
    group_cast_collective_args_list: list[GroupCastCollectiveArg] = field(
        default_factory=list,
        metadata={"help": "group cast collective args list for each overlap stage"},
    )

    @property
    def overlap_degree(self) -> int:
        return len(self.num_remote_tokens_per_overlap_stage)

    def __post_init__(self):
        assert len(self.num_remote_tokens_per_overlap_stage) == len(
            self.group_cast_collective_args_list
        ), (
            f"Got inconsistent overlap degree: "
            f"{len(self.num_remote_tokens_per_overlap_stage)=} and "
            f"{len(self.group_cast_collective_args_list)=}."
        )
