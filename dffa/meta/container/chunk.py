from dataclasses import dataclass, field

from dffa.common.ranges import AttnRanges

from .slice import AttnSlice


@dataclass(repr=False)
class AttnChunk:
    chunk_id: int | None = None

    q_slices: list[AttnSlice] = field(default_factory=list)

    @property
    def q_ranges(self) -> AttnRanges:
        q_ranges = AttnRanges()
        for slice in self.q_slices:
            q_ranges.append(slice.q_range, check=False)  # type: ignore
        return q_ranges

    @property
    def k_ranges(self) -> AttnRanges:
        k_ranges = AttnRanges()
        for slice in self.q_slices:
            k_ranges.append(slice.k_range, check=False)  # type: ignore
        return k_ranges

    @property
    def attn_slices(self) -> list[AttnSlice]:
        return self.q_slices

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
