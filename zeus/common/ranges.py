from typing import Any, Iterator, Sequence, TypeAlias, Union

import torch

from zeus.utils import nvtx

from .range import AttnRange, NaiveRange, RangeError

NaiveRanges: TypeAlias = Sequence[NaiveRange]

__all__ = [
    "is_valid_cu_seqlens",
    "check_valid_cu_seqlens",
    "AttnRanges",
]


def is_valid_cu_seqlens(cu_seqlens: list[int], seq_len: int) -> bool:
    if len(cu_seqlens) == 0:
        return True

    if not cu_seqlens[0] == 0:
        return False

    if not all(cu_seqlens[i - 1] < cu_seqlens[i] for i in range(1, len(cu_seqlens))):
        return False

    if not cu_seqlens[-1] == seq_len:
        return False

    return True


def check_valid_cu_seqlens(cu_seqlens: list[int], seq_len: int) -> None:
    if not is_valid_cu_seqlens(cu_seqlens, seq_len):
        raise ValueError(
            f"The cu_seqlens {cu_seqlens} is invalid against the rule: 'cu_seqlens[0] == 0', \
            and 'cu_seqlens[i-1] < cu_seqlens[i], for any i in [1, len(cu_seqlens))'"  # noqa
        )


def _calc_prefix_offset(merged_ranges: "AttnRanges") -> list[int]:
    """NOTE: the ranges passed in should be merged already,
    no check is applyed here
    """
    prefix_offset = [0]
    for item in merged_ranges:
        prefix_offset.append(prefix_offset[-1] + item.seqlen)
    prefix_offset.pop()

    return prefix_offset


def _binary_search(
    attn_ranges: Union[list[AttnRange], "AttnRanges"], target_range: AttnRange
) -> int:
    # left左侧都是小于等于target，right右侧都是大于target
    left, right = 0, len(attn_ranges) - 1
    while left <= right:
        mid = (left + right) // 2
        if attn_ranges[mid].start > target_range.start:
            right = mid - 1
        else:
            left = mid + 1
    return right


class AttnRanges:
    """
    A dataclass to manage a list of 'AttnRange' objects for attention computation

    高级方法(make_range_local, make_ranges_local, to_local_ranges, find_hole_ranges, find_overlap_ranges)
    NOTE: 这些高级方法都是为了能够更方便实现各种ranges的映射
    NOTE: 什么是AttnRanges的local_ranges?
        因为AttnRanges中的每个attn_range都是描述实际内存中的一片连续存储,
        所以AttnRanges可以选择映射到实际内存中的某一片连续存储, 这片连续的存储满足:
            1. 能够存储AttnRanges中的所有attn_range
            2. 所有的attn_range在其中有序存储(按照range.start和range.end)
        所以AttnRanges的local_ranges就是每个attn_range在这片内存中的实际位置
    Example::
        [[5, 10), [15, 20), [25, 30)]的local_ranges是[[0, 5), [5, 10), [10, 15)]
        [[5, 10), [25, 30), [15, 20)]的local_ranges是[[0, 5), [5, 10), [10, 15)]
        它们的映射关系都如下所示:
        .....[5, 10).....[15, 20).....[25, 30).....
                |           |             |
             [0, 5)      [5, 10)      [10, 15)
    NOTE: AttnRanges的sorted和merged解释如下:
        sorted要求AttnRanges中的每个attn_range都按照start由小到大排序
        merged要求AttnRanges中的每个attn_range都按照start由小到大排序, 并且相邻的attn_range不能有重叠
    """

    def __init__(self) -> None:
        self._ranges: list[AttnRange] = []

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
                f"Some of the {self._ranges=} is invalid against the rule: '0 <= start <= end'"
            )

    # NOTE: Inplace Operation (append, insert, extend, pop)
    def append(self, attn_range: AttnRange, check: bool = False) -> None:
        """Add the attn_range to the end"""
        if check:
            attn_range.check_valid()

        self._ranges.append(attn_range)

    def insert(self, idx: int, attn_range: AttnRange, check: bool = False) -> None:
        """Insert the attn_range to the 'idx'-th position,
        NOTE: if idx >= len(self._ranges), then use 'append' instead
        """
        if check:
            attn_range.check_valid()

        self._ranges.insert(idx, attn_range)

    def extend(self, attn_ranges: "AttnRanges", check: bool = False) -> None:
        if check:
            attn_ranges.check_valid()

        self._ranges.extend(attn_ranges._ranges)

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

        return self._ranges.pop(idx)

    def clear_empty(self) -> "AttnRanges":
        non_empty_ranges = AttnRanges()
        for attn_range in self._ranges:
            if not attn_range.is_empty():
                non_empty_ranges.append(attn_range)

        return non_empty_ranges

    @nvtx.instrument_nvtx
    def sort(self) -> "AttnRanges":
        """
        Sort the attn_ranges by 'attn_range.start' in ascending order

        NOTE: python的sort是稳定的, 因此当start相同时, 会保持原来的顺序
        """
        return AttnRanges.from_ranges(
            sorted(self._ranges, key=lambda attn_range: attn_range.start)
        )

    @nvtx.instrument_nvtx
    def merge(self) -> "AttnRanges":
        """Merge the attn_ranges for the overlapped parts
        in ascending order by 'attn_range.start'
        """

        _ranges = self.sort()._ranges  # required to be sorted first

        _merged_ranges = AttnRanges()

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

        return _merged_ranges

    @nvtx.instrument_nvtx
    def chunk(self, chunk_size: int) -> list["AttnRanges"]:
        _ranges = self.merge()._ranges  # required to be merged first

        chunked_ranges_list = []
        chunked_ranges = AttnRanges()
        cnt = 0
        for attn_range in _ranges:
            seqlen, start = attn_range.seqlen, attn_range.start
            new_cnt = cnt + seqlen
            while new_cnt >= chunk_size:
                seqlen_truc = chunk_size - cnt
                end = start + seqlen_truc
                chunked_ranges.append(AttnRange(start=start, end=end))
                chunked_ranges_list.append(chunked_ranges)

                chunked_ranges = AttnRanges()
                new_cnt -= chunk_size
                start = end
                cnt = 0
            cnt = new_cnt

            if cnt > 0:
                chunked_ranges.append(AttnRange(start=start, end=attn_range.end))

        if len(chunked_ranges) > 0:
            chunked_ranges_list.append(chunked_ranges)

        return chunked_ranges_list

    def truncate(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> "AttnRanges":
        trunc_ranges = AttnRanges()
        for attn_range in self._ranges:
            trunc_range = attn_range.truncate(start, end)
            if trunc_range.is_empty():
                # NOTE: skip the empty range, i.e. those beyond the truncate range
                continue
            trunc_ranges.append(trunc_range)

        return trunc_ranges

    def is_sorted(self) -> bool:
        if not all(
            self._ranges[i - 1].start <= self._ranges[i].start
            for i in range(1, len(self._ranges))
        ):
            return False
        return True

    def is_merged(self) -> bool:
        if self.is_sorted():
            if not all(
                self._ranges[i - 1].end < self._ranges[i].start
                for i in range(1, len(self._ranges))
            ):
                return False
            else:
                return True
        else:
            return False

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

    def to_cu_seqlens(self, seq_len: int) -> list[int]:
        assert self.is_cu_seqlens(
            seq_len
        ), "The ranges can not be converted to cu_seqlens"
        return [0] + [attn_range.end for attn_range in self._ranges]

    @nvtx.instrument_nvtx
    def make_range_local(
        self,
        other_attn_range: AttnRange,
        is_self_merged: bool = False,
        prefix_offset: list[int] | None = None,
    ) -> AttnRange:
        """
        将other_attn_range映射到self_ranges对应的local_ranges中,
        并返回other_attn_range在local_ranges中的位置 (允许通过可选的位置参数来截断)

        Args:
            other_attn_range(AttnRange): 需要被转换的other_attn_range
            is_self_merged(bool): 是否self已经merge
            prefix_offset(list[int] | None): 如果prefix_offset为None, 则计算prefix_offset

        Returns:
            local_range(AttnRange): other_attn_range在self的local_ranges中的位置（如果有截断操作，可能返回一个空的range）
        """

        merged_ranges = self if is_self_merged else self.merge()

        if prefix_offset is None:
            prefix_offset = _calc_prefix_offset(merged_ranges)
        else:
            assert len(prefix_offset) == len(merged_ranges)

        le_idx = _binary_search(merged_ranges, other_attn_range)
        target_range: AttnRange = merged_ranges[le_idx]

        if other_attn_range.is_subrange_of(target_range):
            start = prefix_offset[le_idx] + other_attn_range.start - target_range.start
            local_range = AttnRange(start=start, end=start + other_attn_range.seqlen)
            return local_range
        else:
            raise ValueError(
                f"The attn_range {other_attn_range} is not in the (even merged) attn_ranges {merged_ranges}"
            )

    @nvtx.instrument_nvtx
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
            local_ranges(AttnRanges): 每个attn_range在ref—local-ranges中的位置（如果有截断操作，则其中可能包含若干个空的range）

        Complexity:
            assume len(self) = m, len(other_attn_ranges) = n
            then the complexity is O(m + n * log(m))
        """
        local_ranges = AttnRanges()

        merged_ranges = self if is_self_merged else self.merge()

        prefix_offset = _calc_prefix_offset(merged_ranges)

        for attn_range in other_attn_ranges:
            local_range = merged_ranges.make_range_local(
                attn_range,
                is_self_merged=True,
                prefix_offset=prefix_offset,
            )
            local_ranges.append(local_range)

        return local_ranges

    @nvtx.instrument_nvtx
    def find_hole_ranges(
        self,
        other_attn_ranges: "AttnRanges",
        is_self_merged: bool = False,
        is_other_merged: bool = False,
    ) -> "AttnRanges":
        """
        返回self - other_attn_ranges的结果
        NOTE: 这里的-是集合的差集, 因此返回的hole_ranges中的range是self中的range,
            但是不包含other_attn_ranges中的range

        Args:
            other_attn_ranges(AttnRanges): 需要被减去的range

        Returns:
            NOTE: hole_ranges is merged
            hole_ranges(AttnRanges): self - other_attn_ranges的结果

        Example::
            self = [[0, 10), [15, 20), [20, 30)]
            other_attn_ranges = [[5, 10), [25, 30)]
            return [[0, 5), [15, 25)]
        """

        ranges1 = self if is_self_merged else self.merge()
        ranges2 = other_attn_ranges if is_other_merged else other_attn_ranges.merge()

        p1 = 0
        p2 = 0

        hole_ranges = AttnRanges()

        def get_hole_range(r1: AttnRange, r2: AttnRange) -> AttnRange:
            return AttnRange(start=r1.start, end=min(r1.end, r2.start))

        while p1 < len(ranges1) and p2 < len(ranges2):
            r1: AttnRange = ranges1[p1]
            r2: AttnRange = ranges2[p2]

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

        hole_ranges.extend(ranges1[p1:])

        return hole_ranges

    @nvtx.instrument_nvtx
    def find_overlap_ranges(
        self: "AttnRanges",
        other_attn_ranges: "AttnRanges",
        is_self_merged: bool = False,
        is_other_merged: bool = False,
    ) -> "AttnRanges":
        """
        返回self和other_attn_ranges的交集

        Args:
            other_attn_ranges(AttnRanges): 需要被求交集的range

        Returns:
            NOTE: overlap_ranges is guaranteed to be merged
            overlap_ranges(AttnRanges): self和other_attn_ranges的交集

        Example::
            self = [[0, 10), [15, 20), [25, 30)]
            other_attn_ranges = [[5, 10), [18, 30)]
            return [[5, 10), [18, 30)]
        """

        ranges1 = self if is_self_merged else self.merge()
        ranges2 = other_attn_ranges if is_other_merged else other_attn_ranges.merge()

        p1 = 0
        p2 = 0

        overlap_ranges = AttnRanges()

        while p1 < len(ranges1) and p2 < len(ranges2):
            r1: AttnRange = ranges1[p1]
            r2: AttnRange = ranges2[p2]

            if r1.end > r2.end:
                p2 += 1
            else:
                p1 += 1

            if r1.is_overlap_with(r2):
                overlap_ranges.append(r1.intersect(r2))

        return overlap_ranges

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        if self.is_empty():
            return torch.empty([0, 2], dtype=torch.int32, device=device)
        else:
            return torch.tensor(
                self.to_naive_ranges(), dtype=torch.int32, device=device
            )

    @staticmethod
    def from_cu_seqlens(
        cu_seqlens: list[int],
        seq_len: int,
    ) -> "AttnRanges":
        check_valid_cu_seqlens(cu_seqlens, seq_len)

        ranges = AttnRanges()

        for i in range(1, len(cu_seqlens)):
            ranges.append(AttnRange(cu_seqlens[i - 1], cu_seqlens[i]))

        return ranges

    @staticmethod
    def from_ranges(
        ranges: Union[NaiveRanges, list[AttnRange], "AttnRanges"],
        check: bool = False,
    ) -> "AttnRanges":
        if isinstance(ranges, AttnRanges):  # just copy
            attn_ranges = ranges
        else:
            attn_ranges = AttnRanges()
            _ranges = [AttnRange.from_range(attn_range) for attn_range in ranges]
            attn_ranges._ranges = _ranges

        if check:
            attn_ranges.check_valid()

        return attn_ranges

    def to_naive_ranges(self) -> NaiveRanges:
        return [attn_range.to_naive_range() for attn_range in self._ranges]

    @property
    def total_seqlen(self) -> int:
        """The total seqlen this ranges represent
        which is equal to the sum of the size of each range
        NOTE: distinguish with the property 'self.max_seqlen'
        """
        return sum(attn_range.seqlen for attn_range in self._ranges)

    @property
    def max_seqlen(self) -> int:
        """The maximum seqlen this ranges represent
        which is equal to the maximum size of each range
        NOTE: distinguish with the property 'self.total_seqlen'
        """
        if self.is_empty():
            return 0
        return max(attn_range.seqlen for attn_range in self._ranges)

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

    @property
    def points(self) -> list[int]:
        """The axis points covered by this ranges
        in ascending order and without duplicates
        """

        _points = set()
        for r in self._ranges:
            _points.add(r.start)
            _points.add(r.end)

        return sorted(list(_points))

    def is_empty(self) -> bool:
        return len(self._ranges) == 0

    def __len__(self) -> int:
        return len(self._ranges)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            sub_attn_ranges = AttnRanges()
            for attn_range in self._ranges[idx]:
                sub_attn_ranges.append(attn_range)
            return sub_attn_ranges

        return self._ranges[idx]

    def __setitem__(self, idx: int | slice, value: Union[AttnRange, "AttnRanges"]):
        if isinstance(idx, slice):
            assert isinstance(value, AttnRanges) and idx.stop - idx.start == len(value)
            self._ranges[idx] = value._ranges
        else:
            assert isinstance(value, AttnRange)
            self._ranges[idx] = value

    def __iter__(self) -> Iterator[AttnRange]:
        return iter(self._ranges)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRanges):
            return self._ranges == other._ranges
        return False

    def __hash__(self) -> int:
        return hash(tuple(self._ranges))

    def __repr__(self) -> str:
        if self.is_empty():  # to prevent repr as "[]" to mix up with empty list
            return "[[,)]"
        return f"{self._ranges}"


RangesType: TypeAlias = AttnRanges | NaiveRanges
