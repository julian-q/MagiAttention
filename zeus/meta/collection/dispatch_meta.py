from dataclasses import dataclass

from zeus.common.enum import AttnMaskType, AttnRole, AttnType
from zeus.common.ranges import AttnRanges
from zeus.meta.container.bucket import AttnBucket


@dataclass
class DispatchMeta:
    """The meta info of sequence dispatch for distributed attention

    Args:
        TODO: finish docstring

    NOTE: global_bucket
    """

    attn_role: AttnRole
    attn_type: AttnType
    attn_mask_type: list[AttnMaskType]

    ranges: AttnRanges

    batch_size: int
    total_seqlen: int
    shard_seqlen: int

    chunk_size: int
    num_chunks: int

    cp_rank: int
    cp_size: int

    seqlens: list[int]
    seqlens_permed: list[int]
    seqlens_perm_idxs: list[int]
    seqlens_unperm_idxs: list[int]

    cu_seqlens: list[int]
    cu_seqlens_permed: list[int]

    partitions: list[list[int]]
    partitions_perm_idxs: list[int]
    partitions_unperm_idxs: list[int]

    global_bucket: AttnBucket
    buckets_per_rank: list[AttnBucket]
    host_ranges_per_rank: list[AttnRanges]

    def __post_init__(self) -> None:
        assert len(self.seqlens) == len(self.seqlens_permed) == self.batch_size
        assert (
            len(self.seqlens_perm_idxs)
            == len(self.seqlens_unperm_idxs)
            == self.batch_size
        )
        assert (
            len(self.cu_seqlens) == len(self.cu_seqlens_permed) == self.batch_size + 1
        )

        assert len(self.partitions) == self.cp_size
        assert (
            len(self.partitions_perm_idxs)
            == len(self.partitions_unperm_idxs)
            == self.num_chunks
        )
        assert len(self.buckets_per_rank) == self.cp_size
        assert len(self.host_ranges_per_rank) == self.cp_size

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
