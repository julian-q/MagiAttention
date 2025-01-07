from functools import partial
from itertools import chain
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

__all__ = ["group_cast_collective", "group_reduce_collective"]


def _to_a2a_v_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
):
    world_size = dist.get_world_size(group)

    ######################################
    # 计算a2a_input_split_size和a2a_input #
    ######################################
    input_split = torch.split(input, input_split_size_list, dim=0)
    input_repeat = []
    repeat_times = [len(dst_indices) for dst_indices in dst_indices_list]
    for i, repeat_time in enumerate(repeat_times):
        input_repeat.extend([input_split[i]] * repeat_time)
    flatten_dst_indices = list(chain(*dst_indices_list))
    tensor_with_rank = []
    a2a_input_split_size = [0 for _ in range(world_size)]
    for i, (tensor, dst_indice) in enumerate(zip(input_repeat, flatten_dst_indices)):
        a2a_input_split_size[dst_indice] += tensor.size(0)
        tensor_with_rank.append((tensor, dst_indice, i))
    tensor_with_rank.sort(key=lambda x: x[1])  # 排序是稳定的
    a2a_input_permute_index_list = [x[2] for x in tensor_with_rank]
    a2a_input_unpermute_index_list = sorted(
        range(len(a2a_input_permute_index_list)),
        key=lambda x: a2a_input_permute_index_list[x],
    )
    a2a_input_tensor_size_list = [x[0].size(0) for x in tensor_with_rank]
    a2a_input = torch.cat([x[0] for x in tensor_with_rank], dim=0)

    ########################################
    # 计算a2a_output_split_size和a2a_output #
    ########################################
    a2a_output_split_size_per_rank: list[list[int]] = [[] for _ in range(world_size)]
    a2a_output_permute_index_list_per_rank: list[list[int]] = [
        [] for _ in range(world_size)
    ]
    for i, src_index in enumerate(src_index_list):
        a2a_output_split_size_per_rank[src_index].append(output_split_size_list[i])
        a2a_output_permute_index_list_per_rank[src_index].append(i)
    a2a_output_split_size = [sum(x) for x in a2a_output_split_size_per_rank]
    a2a_output_tensor_size_list = list(chain(*a2a_output_split_size_per_rank))
    a2a_output_permute_index_list = list(chain(*a2a_output_permute_index_list_per_rank))
    a2a_output_unpermute_index_list = sorted(
        range(len(a2a_output_permute_index_list)),
        key=lambda x: a2a_output_permute_index_list[x],
    )
    a2a_output = output

    return (
        a2a_input,
        a2a_output,
        a2a_input_split_size,
        a2a_output_split_size,
        a2a_input_unpermute_index_list,
        a2a_input_tensor_size_list,
        a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list,
    )


def unpermute_tensor(
    tensor: torch.Tensor, unpermute_index_list: list[int], tensor_size_list: list[int]
) -> torch.Tensor:
    tensor_list = list(torch.split(tensor, tensor_size_list, dim=0))
    tensor_list_unpermute = [tensor_list[i] for i in unpermute_index_list]
    return torch.cat(tensor_list_unpermute, dim=0)


def reduce_to_tensor(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    a2a_output_unpermute_index_list: list[int],
    a2a_output_tensor_size_list: list[int],
    output_split_size_list: list[int],
    num_src_list: list[int],
) -> torch.Tensor:
    a2a_output_list = list(torch.split(a2a_output, a2a_output_tensor_size_list, dim=0))
    a2a_output_unpermute_list = [
        a2a_output_list[i] for i in a2a_output_unpermute_index_list
    ]
    output_split_list = list(torch.split(output, output_split_size_list, dim=0))
    output_reduce_list = []
    for i, output_split in enumerate(output_split_list):
        output_split_before_reduce = torch.stack(
            a2a_output_unpermute_list[i : i + num_src_list[i]] + [output_split], dim=0
        )
        output_split_reduced = torch.sum(output_split_before_reduce, dim=0)
        output_reduce_list.append(output_split_reduced)

    output_reduce = torch.cat(output_reduce_list, dim=0)
    output.data = output_reduce.data
    return output


@torch.no_grad()
def group_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> tuple[Work | None, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Args:
        input: [sum(input_split_size), ...]
        output: [sum(output_split_size), ...]
        input_split_size: 0 <= len(input_split_size_list) <= group_size
        output_split_size: [M]
        dst_indices: [N, ?]
        src_indices: [M, ?]

    Returns:
        work: Work | None
        post_process_fn: Callable[[torch.Tensor], torch.Tensor]

    NOTE(xiaowu): 使用a2a-v实现
    """

    assert len(input_split_size_list) == len(dst_indices_list)
    assert len(output_split_size_list) == len(src_index_list)

    (
        a2a_input,
        a2a_output,
        a2a_input_split_size,
        a2a_output_split_size,
        _,
        _,
        a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list,
    ) = _to_a2a_v_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        group=group,
    )

    work = dist.all_to_all_single(
        output=a2a_output,
        input=a2a_input,
        output_split_sizes=a2a_output_split_size,
        input_split_sizes=a2a_input_split_size,
        group=group,
        async_op=True,
    )

    post_process_fn = partial(
        unpermute_tensor,
        unpermute_index_list=a2a_output_unpermute_index_list,
        tensor_size_list=a2a_output_tensor_size_list,
    )

    if async_op:
        return work, post_process_fn
    else:
        work.wait()
        return None, post_process_fn


@torch.no_grad()
def group_reduce_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> tuple[Work | None, Callable[[torch.Tensor], torch.Tensor]]:
    assert len(input_split_size_list) == len(dst_index_list)
    assert len(output_split_size_list) == len(src_indices_list)

    output_gatherd = torch.empty_like(output)
    num_src_list = [len(src_indices) for src_indices in src_indices_list]

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list,
        _,
        _,
    ) = _to_a2a_v_args(
        input=output_gatherd,
        output=input,
        input_split_size_list=output_split_size_list,
        output_split_size_list=input_split_size_list,
        dst_indices_list=src_indices_list,
        src_index_list=dst_index_list,
        group=group,
    )

    work = dist.all_to_all_single(
        output=a2a_output,
        input=a2a_input,
        output_split_sizes=a2a_output_split_size,
        input_split_sizes=a2a_input_split_size,
        group=group,
        async_op=True,
    )

    post_process_fn = partial(
        reduce_to_tensor,
        a2a_output=a2a_output,
        a2a_output_unpermute_index_list=a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list=a2a_output_tensor_size_list,
        output_split_size_list=output_split_size_list,
        num_src_list=num_src_list,
    )

    if async_op:
        return work, post_process_fn
    else:
        work.wait()
        return None, post_process_fn
