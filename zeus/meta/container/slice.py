from dataclasses import dataclass

from zeus.common.enum import AttnMaskType
from zeus.common.range import AttnRange


@dataclass(repr=False)
class AttnSlice:
    slice_id: int | None = None

    mask_type: AttnMaskType | None = None

    q_range: AttnRange | None = None
    k_range: AttnRange | None = None

    area: int = 0

    def __post_init__(self):
        pass

    def __repr__(self) -> str:
        return (
            f"AttnSlice(slice_id={self.slice_id}, "
            f"q_range={self.q_range}, k_range={self.k_range}, "
            f"mask_type={self.mask_type}, area={self.area})"
        )
