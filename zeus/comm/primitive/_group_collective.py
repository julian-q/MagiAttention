from functools import partial
from itertools import chain
from logging import getLogger
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

logger = getLogger("zeus")


__all__ = ["group_cast_collective", "group_reduce_collective"]


def safe_cat(tensor_list: list[torch.Tensor], ref_tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor_list) > 0:
        return torch.cat(tensor_list, dim=0)
    else:
        return torch.empty(
            [0, *ref_tensor.shape[1:]], dtype=ref_tensor.dtype, device=ref_tensor.device
        )


def unpermute_tensor(
    tensor: torch.Tensor, unpermute_index_list: list[int], tensor_size_list: list[int]
) -> torch.Tensor:
    tensor_list = list(torch.split(tensor, tensor_size_list, dim=0))
    tensor_list_unpermute = [tensor_list[i] for i in unpermute_index_list]
    return safe_cat(tensor_list=tensor_list_unpermute, ref_tensor=tensor)


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
    start = 0
    for i, output_split in enumerate(output_split_list):
        output_split_before_reduce = torch.stack(
            a2a_output_unpermute_list[start : start + num_src_list[i]] + [output_split],
            dim=0,
        )
        output_split_reduced = torch.sum(output_split_before_reduce, dim=0)
        output_reduce_list.append(output_split_reduced)
        start += num_src_list[i]

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

    NOTE(xiaowu):
        * 可以通过input_split_size_list把input变成list[splited_input], 其中
        每一个splited_input都可以发给0个或多个rank, 然后通过dst_indices_list
        来决定发给哪个rank.
        * 可以通过output_split_size_list把output变成list[splited_output], 其中
        每一个splited_output都必须从1个src_rank那里收到, 然后通过src_index_list
        来决定从哪个rank那里收到.

    REVIEW(xiaowu):
        * 是否splited_output必须从1个src_rank那里收到? 能否是0个?
    """

    assert len(input_split_size_list) == len(dst_indices_list)
    assert len(output_split_size_list) == len(src_index_list)

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
        for i, (tensor, dst_indice) in enumerate(
            zip(input_repeat, flatten_dst_indices)
        ):
            a2a_input_split_size[dst_indice] += tensor.size(0)
            tensor_with_rank.append((tensor, dst_indice, i))
        tensor_with_rank.sort(key=lambda x: x[1])  # 排序是稳定的
        a2a_input = safe_cat(
            tensor_list=[x[0] for x in tensor_with_rank], ref_tensor=input
        )

        ########################################
        # 计算a2a_output_split_size和a2a_output #
        ########################################
        a2a_output_split_size_per_rank: list[list[int]] = [
            [] for _ in range(world_size)
        ]
        a2a_output_permute_index_list_per_rank: list[list[int]] = [
            [] for _ in range(world_size)
        ]
        for i, src_index in enumerate(src_index_list):
            a2a_output_split_size_per_rank[src_index].append(output_split_size_list[i])
            a2a_output_permute_index_list_per_rank[src_index].append(i)
        a2a_output_split_size = [sum(x) for x in a2a_output_split_size_per_rank]
        a2a_output_tensor_size_list = list(chain(*a2a_output_split_size_per_rank))
        a2a_output_permute_index_list = list(
            chain(*a2a_output_permute_index_list_per_rank)
        )
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
            a2a_output_unpermute_index_list,
            a2a_output_tensor_size_list,
        )

    (
        a2a_input,
        a2a_output,
        a2a_input_split_size,
        a2a_output_split_size,
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

    logger.debug(
        f"RANK {dist.get_rank()}:: args for group_cast_collective: {input.shape=}, {output.shape=}, "
        f"{input_split_size_list=}, {output_split_size_list=}, {dst_indices_list=}, {src_index_list=}. "
        f"args: {a2a_input.shape=}, {a2a_output.shape=}, {a2a_output_split_size=}, {a2a_input_split_size=}."
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
    """
    NOTE(xiaowu):
        * 可以通过input_split_size_list把input变成list[splited_input], 其中
        每一个splited_input都必须发给一个rank, 然后通过dst_index_list来决定发给哪个rank.
        * 可以通过output_split_size_list把output变成list[splited_output], 其中
        每一个splited_output都可以从0个或多个src_rank那里reduce得到, 然后通过src_indices_list
        来决定从哪个rank那里reduce得到.
    """
    assert len(input_split_size_list) == len(dst_index_list)
    assert len(output_split_size_list) == len(src_indices_list)

    num_src_list = [len(src_indices) for src_indices in src_indices_list]

    def _to_a2a_v_args(
        input: torch.Tensor,
        output: torch.Tensor,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_index_list: list[int],
        src_indices_list: list[list[int]],
        group: Optional[dist.ProcessGroup] = None,
    ):
        world_size = dist.get_world_size(group)

        input_split_list = torch.split(input, input_split_size_list, dim=0)
        input_split_with_rank_list = []
        a2a_input_split_size = [0 for _ in range(world_size)]
        for input_split, dst_index in zip(input_split_list, dst_index_list):
            input_split_with_rank_list.append((input_split, dst_index))
            a2a_input_split_size[dst_index] += input_split.size(0)
        input_split_with_rank_list.sort(key=lambda x: x[1])
        a2a_input = safe_cat(
            tensor_list=[x[0] for x in input_split_with_rank_list], ref_tensor=input
        )

        a2a_output_split_size = [0 for _ in range(world_size)]
        size_src_index_i_list = []
        idx = 0
        for output_split_size, src_indices in zip(
            output_split_size_list, src_indices_list
        ):
            for src_index in src_indices:
                a2a_output_split_size[src_index] += output_split_size
                size_src_index_i_list.append((output_split_size, src_index, idx))
                idx += 1
        size_src_index_i_list.sort(key=lambda x: x[1])
        a2a_output_permute_index_list = [x[2] for x in size_src_index_i_list]
        a2a_output_unpermute_index_list = sorted(
            range(len(a2a_output_permute_index_list)),
            key=lambda x: a2a_output_permute_index_list[x],
        )
        a2a_output_tensor_size_list = [x[0] for x in size_src_index_i_list]
        a2a_output = torch.empty(
            [sum(a2a_output_split_size), *output.shape[1:]],
            device=output.device,
            dtype=output.dtype,
        )

        return (
            a2a_input,
            a2a_output,
            a2a_input_split_size,
            a2a_output_split_size,
            a2a_output_tensor_size_list,
            a2a_output_unpermute_index_list,
        )

    (
        a2a_input,
        a2a_output,
        a2a_input_split_size,
        a2a_output_split_size,
        a2a_output_tensor_size_list,
        a2a_output_unpermute_index_list,
    ) = _to_a2a_v_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_index_list=dst_index_list,
        src_indices_list=src_indices_list,
        group=group,
    )

    logger.debug(
        f"RANK {dist.get_rank()}:: args for group_reduce_collective: {input.shape=}, {output.shape=}, "
        f"{input_split_size_list=}, {output_split_size_list=}, {dst_index_list=}, {src_indices_list=}. "
        f"args: {a2a_input.shape=}, {a2a_output.shape=}, {a2a_output_split_size=}, {a2a_input_split_size=}, "
        f"{a2a_output_unpermute_index_list=}, {a2a_output_tensor_size_list=}, "
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
