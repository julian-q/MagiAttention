import torch
import torch.distributed as dist

__all__ = ["scatter_v"]


def scatter_v(
    input: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int = 0,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    rank = dist.get_rank(group)

    if split_sizes is None:
        input_split = torch.chunk(input, chunks=dist.get_world_size(group), dim=dim)
    else:
        input_split = torch.split(input, split_sizes, dim=dim)

    return input_split[rank]
