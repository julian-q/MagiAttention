# mypy: ignore-errors
from dataclasses import dataclass, field
from typing import List

from ...common.enum import AttnMaskType
from ...common.ranges import (
    AttnRange,
    AttnRanges,
    NaiveRanges,
    find_hole_ranges,
    find_hole_ranges_new,
)


@dataclass(repr=False)
class AttnSlice:
    slice_id: int | None = None

    mask_type: AttnMaskType | None = None

    q_range: AttnRange | None = None
    k_range: AttnRange | None = None

    area: int = 0

    @property
    def remote_k_ranges(self) -> AttnRanges:
        remote_k_ranges = AttnRanges(as_cu_seqlens=False)

        # k_range - q_range
        diff_ranges = self.q_range.diff_by(self.k_range)

        for range in diff_ranges:
            remote_k_ranges.append(range, check=False)

        return remote_k_ranges

    def __repr__(self) -> str:
        return (
            f"AttnSlice(slice_id={self.slice_id}, "
            f"q_range={self.q_range}, k_range={self.k_range}, mask_type={self.mask_type}, area={self.area})"
        )


@dataclass(repr=False)
class AttnChunk:
    chunk_id: int | None = None

    q_slices: List[AttnSlice] = field(default_factory=list)

    @property
    def q_ranges(self) -> AttnRanges:
        q_ranges = AttnRanges(as_cu_seqlens=False)
        for slice in self.q_slices:
            q_ranges.append(slice.q_range, check=False)
        return q_ranges

    @property
    def k_ranges(self) -> AttnRanges:
        k_ranges = AttnRanges(as_cu_seqlens=False)
        for slice in self.q_slices:
            k_ranges.append(slice.k_range, check=False)
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
        ), f"{hole_ranges_new} != {hole_ranges}"

        for range in hole_ranges:
            remote_k_ranges.append(AttnRange.from_range(range), check=False)

        return remote_k_ranges

    @property
    def area(self) -> int:
        return sum(slice.area for slice in self.q_slices)

    def __repr__(self, indent: str = "") -> str:
        repr_str = f"{indent}AttnChunk(chunk_id={self.chunk_id}, area={self.area}, q_slices=[\n"
        for i, slice in enumerate(self.q_slices):
            repr_str += (
                f"{indent}    └── {slice}\n"
                if i == len(self.q_slices) - 1
                else f"{indent}    ├── {slice}\n"
            )
        repr_str += f"{indent}])"
        return repr_str


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
