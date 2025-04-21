# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist

from .utils import _get_dims_as_trans_with_dim0, _trans_with_dim0

__all__ = ["all_gather_v"]


def all_gather_v(
    x_local: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int = 0,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """All-gather the local tensor 'x_local' along its dim,
    and return the gathered tensor 'x_gather',
    if not equally split along the dim, then gather indicated by the split sizes

    Args:
        x_local (torch.Tensor): the local tensor to be gathered
        group (dist.ProcessGroup): the process group to be used
        dim (int): the dim to be gathered along
        split_sizes (list[int] | None): the split sizes along the dim,
            where len(split_sizes) should equal to the world size of the group,
                and split_sizes[rank] is the dim size of this local tensor,
                and sum(split_sizes) should equal to the dim size of the global tensor,
            NOTE: if None, then all local tensors should share the same shape

    Returns:
        torch.Tensor: the gathered tensor 'x_gather'
    """

    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    x_local_shape = list(x_local.shape)
    this_dim, other_dims = _get_dims_as_trans_with_dim0(x_local_shape, dim)

    x_local = _trans_with_dim0(x_local, dim)

    if split_sizes is None:  # all local tensors share the same shape
        x_gather_shape = [this_dim * world_size] + other_dims
        x_gather = torch.empty(
            x_gather_shape,
            dtype=x_local.dtype,
            device=x_local.device,
        )
        dist.all_gather_into_tensor(x_gather, x_local, group=group)  # all-gather
    else:  # each local tensor may have a different shape along the dim
        assert (
            len(split_sizes) == world_size
        ), f"The length of {split_sizes=} should equal to {world_size=}"  # noqa
        assert split_sizes[rank] == this_dim, (
            f"The {rank}-th split size of {split_sizes=} should equal to "
            f"the {dim}-th dim size of {x_local_shape=}"  # noqa
        )

        x_gather_list = [
            torch.empty(
                [split_sizes[r]] + other_dims,
                dtype=x_local.dtype,
                device=x_local.device,
            )
            for r in range(world_size)
        ]

        dist.all_gather(x_gather_list, x_local, group=group)  # all-gather-v
        x_gather = torch.cat(x_gather_list, dim=0)

    x_gather = _trans_with_dim0(x_gather, dim)

    return x_gather
