# mypy: ignore-errors
from dataclasses import dataclass, field

from zeus.common.ranges import AttnRanges

from .chunk import AttnChunk
from .slice import AttnSlice


@dataclass(repr=False)
class AttnBucket:
    cp_rank: int | None = None

    q_chunks: list[AttnChunk] = field(default_factory=list)

    @property
    def q_ranges(self) -> AttnRanges:
        q_ranges = AttnRanges()
        for chunk in self.q_chunks:
            q_ranges.extend(chunk.q_ranges, check=False)
        return q_ranges

    @property
    def k_ranges(self) -> AttnRanges:
        k_ranges = AttnRanges()
        for chunk in self.q_chunks:
            k_ranges.extend(chunk.k_ranges, check=False)
        return k_ranges

    @property
    def attn_slices(self) -> list[AttnSlice]:
        _attn_slices = []
        for chunk in self.q_chunks:
            _attn_slices.extend(chunk.attn_slices)
        return _attn_slices

    @property
    def area(self) -> int:
        return sum(chunk.area for chunk in self.q_chunks)

    @property
    def areas(self) -> list[int]:
        return [chunk.area for chunk in self.q_chunks]

    def __repr__(self, indent: str = "") -> str:
        repr_str = (
            f"{indent}AttnBucket(cp_rank={self.cp_rank}, area={self.area}, q_chunks=[\n"
        )
        for i, chunk in enumerate(self.q_chunks):
            repr_str += (
                f"{indent}    └── {chunk.__repr__(indent + '    ')}\n"
                if i == len(self.q_chunks) - 1
                else f"{indent}    ├── {chunk.__repr__(indent + '    ')}\n"
            )
        repr_str += f"{indent}])"
        return repr_str
