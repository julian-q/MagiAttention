from dataclasses import dataclass

from zeus.common.enum import AttnMaskType
from zeus.common.range import AttnRange
from zeus.common.ranges import AttnRanges
from zeus.utils import is_list_value_all


@dataclass(repr=False)
class AttnSlice:
    slice_id: int | None = None

    mask_type: AttnMaskType | None = None

    q_range: AttnRange | None = None
    k_range: AttnRange | None = None

    _area: int | None = None

    # HACK: 后面会将计算面积的逻辑封装在AttnSlice中并且area只读, 这里保留直接设置area的功能

    @property
    def area(self) -> int:
        if self._area is None:
            if self.mask_type is AttnMaskType.FULL:
                # just the area of a full rectangle mask
                self._area = self.q_range.seqlen * self.k_range.seqlen  # type: ignore
            elif self.mask_type is AttnMaskType.CAUSAL:
                raise NotImplementedError(
                    "TODO: Implement area auto-calculation for causal mask."
                )

        return self._area  # type: ignore

    @area.setter
    def area(self, area: int):
        self._area = area

    def __post_init__(self):
        pass

    def __repr__(self) -> str:
        return (
            f"AttnSlice(slice_id={self.slice_id}, "
            f"q_range={self.q_range}, k_range={self.k_range}, "
            f"mask_type={self.mask_type}, area={self.area})"
        )


@dataclass(repr=False)
class MultiKAttnSlice:
    q_range: AttnRange
    k_ranges: AttnRanges
    mask_types: list[AttnMaskType]

    slice_id: int | None = None
    _area: int | None = None

    @property
    def area(self) -> int:
        if self._area is None:
            assert is_list_value_all(
                self.mask_types,
                AttnMaskType.FULL,
            ), f"Only support all full mask for MultiKAttnSlice for now, but got {self.mask_types}"

            self._area = self.q_range.seqlen * self.k_ranges.total_seqlen

        return self._area

    @area.setter
    def area(self, area: int):
        self._area = area

    def __post_init__(self):
        assert len(self.mask_types) == len(self.k_ranges), (
            f"The length of mask_types and k_ranges should be the same, "
            f"but got {len(self.mask_types)} and {len(self.k_ranges)}"
        )

    def __repr__(self) -> str:
        return (
            f"MultiKAttnSlice(slice_id={self.slice_id}, "
            f"q_range={self.q_range}, k_ranges={self.k_ranges}, "
            f"mask_types={self.mask_types}, "
            f"area={self.area})"
        )
