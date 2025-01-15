# mypy: ignore-errors
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)

import numpy as np
import torch

from zeus.utils import nvtx

from .range import AttnRange, NaiveRange, RangeError

NaiveRanges: TypeAlias = List[NaiveRange]


def is_valid_cu_seqlens(cu_seqlens: List[int]) -> bool:
    if len(cu_seqlens) == 0:
        return True

    if not cu_seqlens[0] == 0:
        return False

    if not all(cu_seqlens[i - 1] < cu_seqlens[i] for i in range(1, len(cu_seqlens))):
        return False

    return True


def check_valid_cu_seqlens(cu_seqlens: List[int]) -> None:
    if not is_valid_cu_seqlens(cu_seqlens):
        raise ValueError(
            f"The cu_seqlens {cu_seqlens} is invalid against the rule: 'cu_seqlens[0] == 0', \
            and 'cu_seqlens[i-1] < cu_seqlens[i], for any i in [1, len(cu_seqlens))'"  # noqa
        )


class AttnRanges:
    """A dataclass to manage a list of 'AttnRange' objects for attention computation"""

    def __init__(self, as_cu_seqlens: bool = False) -> None:
        """
        Args:
            as_cu_seqlens: If True, the ranges should be consecutive, mutually exclusive and complete,
                so as to be converted to cu_seqlens
        """
        self._ranges: List[AttnRange] = []
        self._as_cu_seqlens = as_cu_seqlens

    def append(self, range: AttnRange, check: bool = True) -> None:
        """Add the range to the end"""
        if check:
            self._ranges.append(range)
            try:
                self.check_valid(idx=self.size - 1)
            except Exception as e:
                self._ranges.pop()
                raise e
        else:
            self._ranges.append(range)

    def insert(self, idx: int, range: AttnRange, check: bool = True) -> None:
        """Insert the range to the 'idx'-th position,
        NOTE: if idx >= self.size, then use 'append' instead
        """
        if check:
            self._ranges.insert(idx, range)
            try:
                self.check_valid(idx=idx)
            except Exception as e:
                self._ranges.pop(idx)
                raise e
        else:
            self._ranges.insert(idx, range)

    def sort(self, reverse: bool = False) -> "AttnRanges":
        """Sort the ranges by 'range' in ascending order if 'reverse=False', \
        otherwise in descending order
        """
        return AttnRanges.from_ranges(
            sorted(self._ranges, key=lambda range: range.range, reverse=reverse)
        )

    @nvtx.instrument_nvtx
    def merge(self) -> "AttnRanges":
        _ranges = self.sort()._ranges

        _merged_ranges: List[AttnRange] = []

        start, end = None, None
        for range in _ranges:
            if start is None:
                start = range.start
                end = range.end
                _merged_ranges.append(AttnRange(start=start, end=end))
            elif range.start > end:  # a new range can be merged
                start = range.start
                end = range.end
                _merged_ranges.append(AttnRange(start=start, end=end))
            else:
                end = range.end
                _merged_ranges[-1].end = end

        return AttnRanges.from_ranges(_merged_ranges)

    def extend(self, ranges: "AttnRanges", check: bool = True) -> None:
        if check:
            self._ranges.extend(ranges._ranges)
            try:
                self.check_valid()
            except Exception as e:
                self._ranges = self._ranges[: -len(ranges._ranges)]
                raise e
        else:
            self._ranges.extend(ranges._ranges)

    def clear_empty(self) -> None:
        self._ranges = [range for range in self._ranges if not range.is_empty()]

    def to_cu_seqlens(self, check: bool = True) -> List[int]:
        if not self._as_cu_seqlens:
            raise ValueError(
                "This Ranges is not initialized to be 'as_cu_seqlens', "
                "thus not allowed to use this API."
            )

        if check:
            self.check_valid()

        return [0] + [range.end for range in self._ranges]

    @nvtx.instrument_nvtx
    def make_range_local(self, range: AttnRange, merged: bool = False) -> AttnRange:
        if not merged:
            tmp_ranges = AttnRanges.from_ranges(self.ranges)
            tmp_ranges = tmp_ranges.merge()
        else:
            tmp_ranges = self

        slice_start = 0
        for global_range in tmp_ranges._ranges:
            if range.is_subrange_of(global_range):
                start = slice_start + range.start - global_range.start
                return AttnRange(start=start, end=start + range.size)
            slice_start += global_range.size
        else:
            raise ValueError(
                f"The range {range} is not in the (even merged) ranges {tmp_ranges}"
            )

    def make_ranges_local(
        self,
        ranges: Union[List[AttnRange], "AttnRanges"],
        merged: bool = False,
    ) -> "AttnRanges":
        """ """
        local_ranges = AttnRanges()

        if not merged:
            tmp_ranges = AttnRanges.from_ranges(self.ranges)
            tmp_ranges = tmp_ranges.merge()
        else:
            tmp_ranges = self

        original_ranges = self._ranges
        self._ranges = tmp_ranges._ranges

        for range in ranges:
            local_range = self.make_range_local(range, merged=True)
            local_ranges.append(local_range)

        self._ranges = original_ranges

        return local_ranges

    @nvtx.instrument_nvtx
    def to_local_ranges(self) -> "AttnRanges":
        local_ranges = AttnRanges(as_cu_seqlens=True)

        start = 0
        for global_range in self._ranges:
            end = start + global_range.size
            local_ranges.append(
                AttnRange(start=start, end=end),
                check=False,
            )
            start = end

        return local_ranges

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        if self.is_empty():
            return torch.empty([0, 2], dtype=torch.int32, device=device)
        else:
            return torch.tensor(self.ranges, dtype=torch.int32, device=device)

    @staticmethod
    def split_axis_to_ranges(
        axis_start: int,
        axis_end: int,
        split_idxs: List[int],
    ) -> Tuple["AttnRanges", Dict[int, int]]:
        ranges = AttnRanges(as_cu_seqlens=True)
        start_to_range_idx_map = {}

        start, end = axis_start, None
        for idx in sorted(split_idxs):
            if idx > start:
                end = idx
                start_to_range_idx_map[start] = ranges.size
                ranges.append(AttnRange(start=start, end=end), check=False)
                start = idx

        if end is None or end < axis_end:
            start_to_range_idx_map[start] = ranges.size
            ranges.append(AttnRange(start=start, end=axis_end), check=False)

        return ranges, start_to_range_idx_map

    @staticmethod
    def from_cu_seqlens(
        cu_seqlens: List[int],
        as_cu_seqlens: bool = False,
    ) -> "AttnRanges":
        check_valid_cu_seqlens(cu_seqlens)

        ranges = AttnRanges(as_cu_seqlens=as_cu_seqlens)

        for i in range(1, len(cu_seqlens)):
            ranges.append(AttnRange(cu_seqlens[i - 1], cu_seqlens[i]), check=False)

        return ranges

    @staticmethod
    @nvtx.instrument_nvtx
    def from_ranges(
        ranges: Union[NaiveRanges, "AttnRanges"],
        as_cu_seqlens: bool = False,
    ) -> "AttnRanges":
        if isinstance(ranges, AttnRanges):  # just copy
            return ranges

        _ranges = [AttnRange.from_range(r) for r in ranges]
        attn_ranges = AttnRanges(as_cu_seqlens=as_cu_seqlens)
        attn_ranges._ranges = _ranges
        attn_ranges.check_valid()

        return attn_ranges

    @property
    def ranges(self) -> NaiveRanges:
        return [range.range for range in self._ranges]

    @property
    def last(self) -> AttnRange:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no last range")
        return self._ranges[-1]

    @last.setter
    def last(self, range: AttnRange) -> None:
        self._ranges[-1] = range

    @property
    def size(self) -> int:
        return len(self._ranges)

    @property
    def seqlen(self) -> int:
        return sum(range.size for range in self._ranges)

    @property
    def max_seqlen(self) -> int:
        if self.is_empty():
            return 0
        return max(range.size for range in self._ranges)

    @property
    def start(self) -> int:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no start")
        return min(range.start for range in self._ranges)

    @property
    def end(self) -> int:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no end")
        return max(range.end for range in self._ranges)

    def is_as_cu_seqlens(self) -> bool:
        return self._as_cu_seqlens

    def is_empty(self) -> bool:
        return self.size == 0

    def is_valid_idx(self, idx: int) -> bool:
        return 0 <= idx < self.size

    def check_valid_idx(self, idx: int) -> None:
        if not self.is_valid_idx(idx):
            raise IndexError(f"The index {idx} is out of the range [0, {self.size})")

    def is_valid(
        self,
        ranges: Optional["AttnRanges"] = None,
        idx: int | None = None,
    ) -> bool:
        ranges = self if ranges is None else ranges

        if idx is not None:
            ranges.check_valid_idx(idx)

        if ranges.is_empty():  # empty ranges are always valid
            return True

        if idx is not None:
            if not ranges[idx].is_valid():
                return False
        else:
            if not all(range.is_valid() for range in ranges):
                return False

        if self._as_cu_seqlens:
            if not ranges[0].start == 0:
                return False

            if idx is not None:
                if idx > 0:
                    if not ranges[idx - 1].end == ranges[idx].start:
                        return False
                if idx < ranges.size - 1:
                    if not ranges[idx].end == ranges[idx + 1].start:
                        return False
            else:
                if not ranges[0].start == 0:
                    return False
                if not all(
                    ranges[i - 1].end == ranges[i].start for i in range(1, len(ranges))
                ):
                    return False

        return True

    def check_valid(
        self,
        ranges: Optional["AttnRanges"] = None,
        idx: int | None = None,
    ) -> None:
        ranges = self if ranges is None else ranges

        if not self.is_valid(ranges=ranges, idx=idx):
            if self._as_cu_seqlens:
                if idx is not None:
                    raise ValueError(
                        f"The range {ranges[idx]} of {ranges=} is invalid, "
                        f"either against the 'cu_seqlens' check or against the rule: '0 <= start <= end'"
                    )
                else:
                    raise ValueError(
                        f"Some of the {ranges=} is invalid, "
                        f"either against the 'cu_seqlens' check or against the rule: '0 <= start <= end'"
                    )
            else:
                if idx is not None:
                    raise ValueError(
                        f"The range {ranges[idx]} of {ranges=} is invalid against the rule: '0 <= start <= end'"
                    )
                else:
                    raise ValueError(
                        f"Some of the {ranges=} is invalid against the rule: '0 <= start <= end'"
                    )

    @staticmethod
    def check_valid_qk_ranges(
        q_ranges: "AttnRanges",
        k_ranges: "AttnRanges",
        is_self_attn: bool = False,
    ) -> None:
        assert q_ranges.is_as_cu_seqlens(), "q_ranges should be 'as_cu_seqlens'"

        assert (
            q_ranges.size == k_ranges.size
        ), "q_ranges and k_ranges should have the same length"

        if is_self_attn:
            assert (max_end_q := q_ranges.end) >= (max_end_k := k_ranges.end), (
                f"For self-attn, The max end index of k_ranges ({max_end_k}) "
                f"should NOT be larger than the max end index of q_ranges {max_end_q}"
            )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> AttnRange:
        self.check_valid_idx(idx)
        return self._ranges[idx]

    def __iter__(self) -> Iterator[AttnRange]:
        return iter(self._ranges)

    def __repr__(self) -> str:
        return f"{self._ranges}"

    def pop(self, idx: int = -1) -> AttnRange:
        """Remove and return item at index (default last).

        Args:
            idx: The index of the element to remove. Default is -1 (last element).

        Returns:
            The removed AttnRange object.

        Raises:
            IndexError: If the list is empty or idx is out of range.
        """
        if self.is_empty():
            raise IndexError("pop from empty AttnRanges")

        if idx < 0:
            idx = len(self._ranges) + idx

        self.check_valid_idx(idx)
        return self._ranges.pop(idx)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRanges):
            return self._ranges == other._ranges
        return False


RangesType: TypeAlias = AttnRanges | NaiveRanges


@nvtx.instrument_nvtx
def find_hole_ranges(
    all_ones_ranges: NaiveRanges,
    all_zeros_ranges: NaiveRanges,
    axis_start: int,
    axis_end: int,
) -> NaiveRanges:
    """axis is a one-dim array, made of only 0s and 1s,
    find all the 'hole ranges' as follows:
    0. the axis is initialized as all 0s
    1. first of all, all_ones_ranges take up the positions in axis with 1s
    2. then, all_zeros_ranges take up the positions in axis with 0s
    3. finally, in the axis, there'll leave some consecutive 1s, \
        which are defined as the 'hole ranges'
    """
    assert (
        axis_start < axis_end
    ), f"axis_start ({axis_start}) should be smaller than axis_end ({axis_end})"
    axis = np.zeros(axis_end - axis_start, dtype=int)

    for all_ones_range in all_ones_ranges:
        start = all_ones_range[0] - axis_start
        end = all_ones_range[1] - axis_start
        axis[start:end] = 1

    for all_zeros_range in all_zeros_ranges:
        start = all_zeros_range[0] - axis_start
        end = all_zeros_range[1] - axis_start
        axis[start:end] = 0

    diff = np.diff(axis)

    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    if axis[0]:
        start_indices = np.insert(start_indices, 0, 0)

    if axis[-1]:
        end_indices = np.append(end_indices, len(axis))

    hole_ranges = list(zip(start_indices + axis_start, end_indices + axis_start))

    return hole_ranges


@nvtx.instrument_nvtx
def find_hole_ranges_new(
    ranges1: AttnRanges,
    ranges2: AttnRanges,
) -> AttnRanges:
    ranges1 = ranges1.merge()
    ranges2 = ranges2.merge()

    p1 = 0
    p2 = 0

    hole_ranges = AttnRanges()

    def get_hole_range(r1: AttnRange, r2: AttnRange) -> AttnRange:
        return AttnRange(start=r1.start, end=min(r1.end, r2.start))

    while p1 < len(ranges1) and p2 < len(ranges2):
        r1 = ranges1[p1]
        r2 = ranges2[p2]

        if r1.end > r2.end:
            p2 += 1
        else:
            p1 += 1

        if r1.start < r2.start:
            hole_ranges.append(get_hole_range(r1, r2))

        if r1.start < r2.end:
            try:
                r1.start = r2.end
            except RangeError:
                pass

    while p1 < len(ranges1):
        hole_ranges.append(ranges1[p1])
        p1 += 1

    hole_ranges = hole_ranges.merge()

    return hole_ranges


@nvtx.instrument_nvtx
def find_overlap_ranges(
    ranges1: NaiveRanges,
    ranges2: NaiveRanges,
    axis_start: int,
    axis_end: int,
) -> NaiveRanges:
    """axis is a one-dim array, made of only 0s and 1s,
    find all the 'overlap ranges' as follows:
    0. the axis is initialized as all 0s
    1. first of all, ranges1 take up the positions in axis with 1
    2. then, ranges2 add to the positions in axis also with 1
    3. finally, in the axis, there'll leave some consecutive 2s, turned to True, then turned to 1s, \
        which are defined as the 'overlap ranges'
    """
    assert (
        axis_start < axis_end
    ), f"axis_start ({axis_start}) should be smaller than axis_end ({axis_end})"
    axis = np.zeros(axis_end - axis_start, dtype=int)

    for range1 in ranges1:
        start = range1[0] - axis_start
        end = range1[1] - axis_start
        axis[start:end] += 1

    for range2 in ranges2:
        start = range2[0] - axis_start
        end = range2[1] - axis_start
        axis[start:end] += 1

    axis = (axis == 2).astype(int)

    diff = np.diff(axis)

    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    if axis[0]:
        start_indices = np.insert(start_indices, 0, 0)

    if axis[-1]:
        end_indices = np.append(end_indices, len(axis))

    hole_ranges = list(zip(start_indices + axis_start, end_indices + axis_start))

    return hole_ranges


@nvtx.instrument_nvtx
def find_overlap_ranges_new(
    ranges1: AttnRanges,
    ranges2: AttnRanges,
) -> AttnRanges:
    ranges1 = ranges1.merge()
    ranges2 = ranges2.merge()

    p1 = 0
    p2 = 0

    overlap_ranges = []

    def is_overlap(r1: AttnRange, r2: AttnRange) -> bool:
        return (r1.start < r2.end and r1.end > r2.start) or (
            r2.start < r1.end and r2.end > r1.start
        )

    def get_overlap_range(r1: AttnRange, r2: AttnRange) -> AttnRange:
        return AttnRange(start=max(r1.start, r2.start), end=min(r1.end, r2.end))

    while p1 < len(ranges1) and p2 < len(ranges2):
        r1 = ranges1[p1]
        r2 = ranges2[p2]

        if r1.end > r2.end:
            p2 += 1
        else:
            p1 += 1

        if is_overlap(r1, r2):
            overlap_ranges.append(get_overlap_range(r1, r2))

    overlap_ranges = AttnRanges.from_ranges(overlap_ranges)
    overlap_ranges = overlap_ranges.merge()

    return overlap_ranges


NestedIntList: TypeAlias = Union[List[int], Tuple[int, ...], Sequence["NestedIntList"]]
