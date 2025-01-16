# mypy: ignore-errors
from typing import Any, Iterator, List, Optional, Sequence, Tuple, TypeAlias, Union

import torch

from zeus.utils import nvtx

from .range import AttnRange, NaiveRange, RangeError

NaiveRanges: TypeAlias = List[NaiveRange]


def is_valid_cu_seqlens(cu_seqlens: List[int], seq_len: int) -> bool:
    if len(cu_seqlens) == 0:
        return True

    if not cu_seqlens[0] == 0:
        return False

    if not all(cu_seqlens[i - 1] < cu_seqlens[i] for i in range(1, len(cu_seqlens))):
        return False

    if not cu_seqlens[-1] == seq_len:
        return False

    return True


def check_valid_cu_seqlens(cu_seqlens: List[int], seq_len: int) -> None:
    if not is_valid_cu_seqlens(cu_seqlens, seq_len):
        raise ValueError(
            f"The cu_seqlens {cu_seqlens} is invalid against the rule: 'cu_seqlens[0] == 0', \
            and 'cu_seqlens[i-1] < cu_seqlens[i], for any i in [1, len(cu_seqlens))'"  # noqa
        )


class AttnRanges:
    """A dataclass to manage a list of 'AttnRange' objects for attention computation"""

    def __init__(self) -> None:
        self._ranges: List[AttnRange] = []

    def is_valid_idx(self, idx: int) -> bool:
        return 0 <= idx < self.size

    def check_valid_idx(self, idx: int) -> None:
        if not self.is_valid_idx(idx):
            raise IndexError(f"The index {idx} is out of the range [0, {self.size})")

    def is_valid(
        self,
    ) -> bool:
        if self.is_empty():  # empty ranges are always valid
            return True

        if not all(attn_range.is_valid() for attn_range in self._ranges):
            return False

        return True

    def check_valid(
        self,
    ) -> None:
        if not self.is_valid():
            raise ValueError(
                f"Some of the {self.ranges=} is invalid against the rule: '0 <= start <= end'"
            )

    # Inplace Operation(append, insert, pop, extend, sort, clear_empty)
    def append(self, attn_range: AttnRange, check: bool = True) -> None:
        """Add the attn_range to the end"""
        if check:
            attn_range.check_valid()
            self._ranges.append(attn_range)
        else:
            self._ranges.append(attn_range)

    def insert(self, idx: int, attn_range: AttnRange, check: bool = True) -> None:
        """Insert the attn_range to the 'idx'-th position,
        NOTE: if idx >= self.size, then use 'append' instead
        """
        if check:
            self.check_valid_idx(idx)
            attn_range.check_valid()
            self._ranges.insert(idx, range)
        else:
            self._ranges.insert(idx, range)

    def extend(self, ranges: "AttnRanges", check: bool = True) -> None:
        if check:
            ranges.check_valid()
            self._ranges.extend(ranges._ranges)
        else:
            self._ranges.extend(ranges._ranges)

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

    def clear_empty(self) -> None:
        self._ranges = [
            attn_range for attn_range in self._ranges if not attn_range.is_empty()
        ]

    @nvtx.instrument_nvtx
    def sort(self, reverse: bool = False) -> "AttnRanges":
        """Sort the attn_ranges by 'attn_range.start' in ascending order if 'reverse=False', \
        otherwise in descending order
        """
        return AttnRanges.from_ranges(
            sorted(
                self._ranges, key=lambda attn_range: attn_range.start, reverse=reverse
            )
        )

    @nvtx.instrument_nvtx
    def merge(self) -> "AttnRanges":
        _ranges = self.sort()._ranges

        _merged_ranges: List[AttnRange] = []

        start, end = None, None
        for attn_range in _ranges:
            if start is None:
                start = attn_range.start
                end = attn_range.end
                _merged_ranges.append(AttnRange(start=start, end=end))
            elif attn_range.start > end:  # a new range can be merged
                start = attn_range.start
                end = attn_range.end
                _merged_ranges.append(AttnRange(start=start, end=end))
            else:
                end = attn_range.end
                _merged_ranges[-1].end = end

        return AttnRanges.from_ranges(_merged_ranges)

    def is_sorted(self) -> bool:
        if not all(
            self._ranges[i - 1].start <= self._ranges[i].start
            for i in range(1, len(self._ranges))
        ):
            return False
        return True

    def is_merged(self) -> bool:
        sorted_ranges = self.sort()
        if not all(
            sorted_ranges._ranges[i - 1].end < sorted_ranges._ranges[i].start
            for i in range(1, len(sorted_ranges._ranges))
        ):
            return False
        else:
            return True

    def is_cu_seqlens(self, seqlen: int) -> bool:
        if not self._ranges[0].start == 0:
            return False
        if not all(
            self._ranges[i - 1].end == self._ranges[i].start
            for i in range(1, len(self._ranges))
        ):
            return False
        if not self._ranges[-1].end == seqlen:
            return False

        return True

    def to_cu_seqlens(self, seq_len: int) -> List[int]:
        assert self.is_cu_seqlens(
            seq_len
        ), "The ranges can not be converted to cu_seqlens"
        return [0] + [attn_range.end for attn_range in self._ranges]

    # 高级方法(make_range_local, make_ranges_local, to_local_ranges, find_hole_ranges, find_overlap_ranges)
    # NOTE: 这些高级方法都是为了能够更方便实现各种ranges的映射
    # NOTE: 什么是AttnRanges的local_ranges?
    #     因为AttnRanges中的每个attn_range都是描述实际内存中的一片连续存储,
    #     所以AttnRanges可以选择映射到实际内存中的某一片连续存储, 这片连续的存储满足:
    #         1. 能够存储AttnRanges中的所有attn_range
    #         2. 所有的attn_range在其中有序存储(按照range.start和range.end)
    #     所以AttnRanges的local_ranges就是每个attn_range在这片内存中的实际位置
    # Example::
    #     [[5, 10), [15, 20), [25, 30)]的local_ranges是[[0, 5), [5, 10), [10, 15)]
    @nvtx.instrument_nvtx
    def make_range_local(
        self,
        attn_range: AttnRange,
        is_self_merged: bool = False,
        prefix_offset: Optional[List[int]] = None,
    ) -> AttnRange:
        """
        将attn_range映射到self的local_ranges中, 并返回attn_range在local_ranges中的位置

        Args:
            attn_range(AttnRange): 需要被转换的attn_range
            is_self_merged(bool): 是否self已经merge
            prefix_offset(Optional[List[int]]): 如果prefix_offset为None, 则计算prefix_offset

        Returns:
            local_range(AttnRange): attn_range在self的local_ranges中的位置
        """

        def binary_search(arr: list, target: int) -> int:
            # left左侧都是小于等于target，right右侧都是大于target
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid].start > target.start:
                    right = mid - 1
                else:
                    left = mid + 1
            return right

        if not is_self_merged:
            merged_ranges = self.merge()
        else:
            merged_ranges = self

        if prefix_offset is None:
            prefix_offset = [0]
            for item in merged_ranges:
                prefix_offset.append(prefix_offset[-1] + item.size)
            prefix_offset.pop()
        else:
            assert len(prefix_offset) == len(merged_ranges)

        le_idx = binary_search(merged_ranges, attn_range)
        target_range = merged_ranges[le_idx]

        if attn_range.is_subrange_of(target_range):
            start = prefix_offset[le_idx] + attn_range.start - target_range.start
            local_range = AttnRange(start=start, end=start + attn_range.size)
            return local_range
        else:
            raise ValueError(
                f"The attn_range {attn_range} is not in the (even merged) attn_ranges {merged_ranges}"
            )

    def make_ranges_local(
        self,
        other_attn_ranges: "AttnRanges",
        is_self_merged: bool = False,
    ) -> "AttnRanges":
        """
        将other_attn_ranges中的每个attn_range映射到self的local_ranges中, 并返回每个attn_range
        在self的local_ranges中的位置

        Args:
            ranges(AttnRanges): 需要被转换的range, 必须是merge后的range
            is_self_merged(bool): 是否self已经merge

        Returns:
            local_ranges(AttnRanges): 每个attn_range在ref—local-ranges中的位置

        Complexity:
            assume len(self) = m, len(other_attn_ranges) = n
            then the complexity is O(m + n * log(m))
        """
        local_ranges = AttnRanges()

        if not is_self_merged:
            merged_ranges = self.merge()
        else:
            merged_ranges = self

        prefix_offset = [0]
        for item in merged_ranges:
            prefix_offset.append(prefix_offset[-1] + item.size)
        prefix_offset.pop()

        for attn_range in other_attn_ranges:
            local_range = merged_ranges.make_range_local(
                attn_range, is_self_merged=True, prefix_offset=prefix_offset
            )
            local_ranges.append(local_range)

        return local_ranges

    def find_hole_ranges(
        self,
        other: "AttnRanges",
    ) -> "AttnRanges":
        ranges1 = self.merge()
        ranges2 = other.merge()

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

    def find_overlap_ranges(
        self: "AttnRanges",
        other: "AttnRanges",
    ) -> "AttnRanges":
        ranges1 = self.merge()
        ranges2 = other.merge()

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

    # def to_local_ranges(self) -> "AttnRanges":
    # REVIEW(xiaowu): 检查这个方法的正确性，并判断是否需要
    #     local_ranges = AttnRanges()

    #     start = 0
    #     for global_range in self._ranges:
    #         end = start + global_range.size
    #         local_ranges.append(
    #             AttnRange(start=start, end=end),
    #             check=False,
    #         )
    #         start = end

    #     return local_ranges

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        if self.is_empty():
            return torch.empty([0, 2], dtype=torch.int32, device=device)
        else:
            return torch.tensor(
                self.to_naive_ranges(), dtype=torch.int32, device=device
            )

    @staticmethod
    def from_cu_seqlens(
        cu_seqlens: List[int],
        seq_len: int,
    ) -> "AttnRanges":
        check_valid_cu_seqlens(cu_seqlens, seq_len)

        ranges = AttnRanges()

        for i in range(1, len(cu_seqlens)):
            ranges.append(AttnRange(cu_seqlens[i - 1], cu_seqlens[i]), check=False)

        return ranges

    @staticmethod
    @nvtx.instrument_nvtx
    def from_ranges(
        ranges: Union[NaiveRanges, "AttnRanges"],
        check: bool = True,
    ) -> "AttnRanges":
        if isinstance(ranges, AttnRanges):  # just copy
            attn_ranges = ranges
        else:
            attn_ranges = AttnRanges()
            _ranges = [
                AttnRange.from_range(attn_range, check=False) for attn_range in ranges
            ]
            attn_ranges._ranges = _ranges

        if check:
            attn_ranges.check_valid()

        return attn_ranges

    def to_naive_ranges(self) -> NaiveRanges:
        return [attn_range.to_naive_range() for attn_range in self._ranges]

    @property
    def last(self) -> AttnRange:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no last attn_range")
        return self._ranges[-1]

    @last.setter
    def last(self, attn_range: AttnRange) -> None:
        self._ranges[-1] = attn_range

    @property
    def size(self) -> int:
        return len(self._ranges)

    @property
    def seqlen(self) -> int:
        return sum(attn_range.size for attn_range in self._ranges)

    @property
    def max_seqlen(self) -> int:
        if self.is_empty():
            return 0
        return max(attn_range.size for attn_range in self._ranges)

    @property
    def start(self) -> int:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no start")
        return min(attn_range.start for attn_range in self._ranges)

    @property
    def end(self) -> int:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no end")
        return max(attn_range.end for attn_range in self._ranges)

    def is_empty(self) -> bool:
        return self.size == 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> AttnRange:
        self.check_valid_idx(idx)
        return self._ranges[idx]

    def __iter__(self) -> Iterator[AttnRange]:
        return iter(self._ranges)

    def __repr__(self) -> str:
        return f"{self._ranges}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRanges):
            return self._ranges == other._ranges
        return False


RangesType: TypeAlias = AttnRanges | NaiveRanges

NestedIntList: TypeAlias = Union[List[int], Tuple[int, ...], Sequence["NestedIntList"]]
