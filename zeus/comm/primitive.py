from itertools import chain
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

__all__ = ["multi_cast_collective"]


def _to_a2a_v_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """ """
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
        tensor_with_rank.append((tensor, dst_indice))
    tensor_with_rank.sort(key=lambda x: x[1])
    a2a_input = torch.cat([x[0] for x in tensor_with_rank], dim=0)

    ########################################
    # 计算a2a_output_split_size和a2a_output #
    ########################################
    a2a_output_split_size = [0 for _ in range(world_size)]
    for i, src_index in enumerate(src_index_list):
        a2a_output_split_size[src_index] += output_split_size_list[i]
    a2a_output = output

    return a2a_input, a2a_output, a2a_input_split_size, a2a_output_split_size


def multi_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Work | None:
    """
    Args:
        input: [sum(input_split_size), ...]
        output: [sum(output_split_size), ...]
        input_split_size: 0 <= len(input_split_size_list) <= group_size
        output_split_size: [M]
        dst_indices: [N, ?]
        src_indices: [M, ?]

    NOTE(xiaowu): 使用a2a-v实现
    """

    assert len(input_split_size_list) == len(dst_indices_list)
    assert len(output_split_size_list) == len(src_index_list)

    a2a_input, a2a_output, a2a_input_split_size, a2a_output_split_size = _to_a2a_v_args(
        input,
        output,
        input_split_size_list,
        output_split_size_list,
        dst_indices_list,
        src_index_list,
        group,
    )

    work = dist.all_to_all_single(
        output=a2a_output,
        input=a2a_input,
        output_split_sizes=a2a_output_split_size,
        input_split_sizes=a2a_input_split_size,
        group=group,
        async_op=True,
    )

    if async_op:
        return work
    else:
        work.wait()
        return None
