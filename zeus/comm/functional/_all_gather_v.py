import torch
import torch.distributed as dist

from ..primitive._all_gather_v import all_gather_v


class AllGatherV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
        dim: int,
        split_sizes: list[int] | None = None,
    ):
        ctx.group = group
        ctx.dim = dim
        ctx.split_sizes = split_sizes
        return all_gather_v(x, group, dim, split_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group: dist.ProcessGroup = ctx.group
        dim: int = ctx.dim
        split_sizes: list[int] | None = ctx.split_sizes

        rank = dist.get_rank(group)

        if split_sizes is None:
            grad_output_split = torch.chunk(
                grad_output, chunks=dist.get_world_size(group), dim=dim
            )
        else:
            grad_output_split = torch.split(grad_output, split_sizes, dim=dim)

        return grad_output_split[rank], None, None, None


def all_gather_v_func(
    x_local: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    return AllGatherV.apply(x_local, group, dim, split_sizes)
