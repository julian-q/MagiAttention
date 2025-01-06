from itertools import chain
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import Work

__all__ = ["multi_cast_collective", "AsyncStreamWork"]


class AsyncStreamWork:
    def __init__(self, side_stream: torch.cuda.Stream) -> None:
        super().__init__()

        self.side_stream = side_stream

    def wait(self, cur_stream: torch.cuda.Stream | None = None) -> bool:
        cur_stream = (
            cur_stream if cur_stream is not None else torch.cuda.current_stream()
        )
        cur_stream.wait_stream(self.side_stream)
        return True


def _perm_idxs2unperm_idxs(perm_idxs: list[int]) -> list[int]:
    unperm_idxs = [0] * len(perm_idxs)

    for i in range(len(perm_idxs)):
        unperm_idxs[perm_idxs[i]] = i

    return unperm_idxs


def _unpermute_a2a_output(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    a2a_output_split_size: list[int],
    a2a_output_unperm_idxs: list[int],
    rank: int,
) -> None:
    a2a_output_split_size = [
        split_size for r, split_size in enumerate(a2a_output_split_size) if r != rank
    ]
    a2a_output_split = torch.split(a2a_output, a2a_output_split_size, dim=0)
    output.copy_(
        torch.cat([a2a_output_split[i] for i in a2a_output_unperm_idxs], dim=0)
    )


def _to_a2a_v_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[int]]:
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
    a2a_output = torch.empty_like(output)
    a2a_output_split_sizes_per_rank: list[list[tuple[int, int]]] = [
        [] for _ in range(world_size)
    ]

    for i, src_index in enumerate(src_index_list):
        a2a_output_split_sizes_per_rank[src_index].append(
            (output_split_size_list[i], i)
        )

    a2a_output_perm_idxs = [
        split_size_with_idx[1]
        for a2a_output_split_sizes_for_this_rank in a2a_output_split_sizes_per_rank
        for split_size_with_idx in a2a_output_split_sizes_for_this_rank
    ]
    a2a_output_unperm_idxs = _perm_idxs2unperm_idxs(a2a_output_perm_idxs)

    a2a_output_split_size = [0 for _ in range(world_size)]
    for rank, a2a_output_split_sizes_for_this_rank in enumerate(
        a2a_output_split_sizes_per_rank
    ):
        for split_size_with_idx in a2a_output_split_sizes_for_this_rank:
            split_size, _ = split_size_with_idx
            a2a_output_split_size[rank] += split_size

    return (
        a2a_input,
        a2a_output,
        a2a_input_split_size,
        a2a_output_split_size,
        a2a_output_unperm_idxs,
    )


def multi_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Work | AsyncStreamWork | None:
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

    if async_op:
        multi_cast_stream = torch.cuda.Stream()
    else:
        multi_cast_stream = torch.cuda.current_stream()

    (
        a2a_input,
        a2a_output,
        a2a_input_split_size,
        a2a_output_split_size,
        a2a_output_unperm_idxs,
    ) = _to_a2a_v_args(
        input,
        output,
        input_split_size_list,
        output_split_size_list,
        dst_indices_list,
        src_index_list,
        group,
    )

    multi_cast_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(multi_cast_stream):
        a2a_work = dist.all_to_all_single(
            output=a2a_output,
            input=a2a_input,
            output_split_sizes=a2a_output_split_size,
            input_split_sizes=a2a_input_split_size,
            group=group,
            async_op=True,
        )
        a2a_work.wait()  # NOTE: we need to unpermute the a2a output immediately
        _unpermute_a2a_output(
            output=output,
            a2a_output=a2a_output,
            a2a_output_split_size=a2a_output_split_size,
            a2a_output_unperm_idxs=a2a_output_unperm_idxs,
            rank=dist.get_rank(group),
        )
    a2a_output.record_stream(multi_cast_stream)

    if async_op:
        return AsyncStreamWork(multi_cast_stream)
    else:
        return None
