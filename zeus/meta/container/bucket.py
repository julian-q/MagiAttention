# mypy: ignore-errors
from dataclasses import dataclass, field
from typing import List

from zeus.common.range import AttnRange
from zeus.common.ranges import (
    AttnRanges,
    NaiveRanges,
    find_hole_ranges,
    find_hole_ranges_new,
)

from .chunk import AttnChunk


@dataclass(repr=False)
class AttnBucket:
    cp_rank: int | None = None

    q_chunks: List[AttnChunk] = field(default_factory=list)

    @property
    def q_ranges(self) -> AttnRanges:
        q_ranges = AttnRanges(as_cu_seqlens=False)
        for chunk in self.q_chunks:
            q_ranges.extend(chunk.q_ranges, check=False)
        return q_ranges

    @property
    def k_ranges(self) -> AttnRanges:
        k_ranges = AttnRanges(as_cu_seqlens=False)
        for chunk in self.q_chunks:
            k_ranges.extend(chunk.k_ranges, check=False)
        return k_ranges

    @property
    def remote_k_ranges(self) -> AttnRanges:
        remote_k_ranges = AttnRanges(as_cu_seqlens=False)

        q_ranges = self.q_ranges
        k_ranges = self.k_ranges
        axis_start, axis_end = min(q_ranges.start, k_ranges.start), max(
            q_ranges.end, k_ranges.end
        )
        hole_ranges: NaiveRanges = find_hole_ranges(
            all_ones_ranges=k_ranges.ranges,
            all_zeros_ranges=q_ranges.ranges,
            axis_start=axis_start,
            axis_end=axis_end,
        )

        hole_ranges_new = find_hole_ranges_new(
            ranges1=k_ranges,
            ranges2=q_ranges,
        )
        assert hole_ranges_new == AttnRanges.from_ranges(
            hole_ranges
        ), f"{hole_ranges_new} != {hole_ranges}, \nranges1: {k_ranges}, \nranges2: {q_ranges}"

        for range in hole_ranges:
            remote_k_ranges.append(AttnRange.from_range(range), check=False)

        return remote_k_ranges

    @property
    def area(self) -> int:
        return sum(chunk.area for chunk in self.q_chunks)

    @property
    def areas(self) -> List[int]:
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
