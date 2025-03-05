from typing import Optional

import torch
import torch.distributed as dist

from zeus.comm.work import WorkWithPostProcessFn
from zeus.utils import nvtx

from .utils import _calc_group_cast_a2a_args, _calc_group_reduce_a2a_args

__all__ = [
    "group_cast_collective",
    "group_reduce_collective",
]


@torch.no_grad()
@nvtx.instrument_nvtx
def group_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """
    Args:
        input: [sum(input_split_size), ...]
        output: [sum(output_split_size), ...]
        input_split_size: 0 <= len(input_split_size_list) <= group_size
        output_split_size: [M]
        dst_indices: [N, ?]
        src_indices: [M, ?]

        HACK:
        **kwargs: additional keyword arguments,
        this kernel is for now based on all2all-v,
        thus introducing pre-/post-processing overhead
        on both tensor and meta info to be compatible with all2all-v input/output.
        Therefore, we add `kwargs` since the processing of meta info
        can be processed in advance, and just passed in through `kwargs` to reduce runtime overhead

    Returns:
        work + with_post_process_fn:
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

    world_size = dist.get_world_size(group)

    # ---------    calc group cast a2a args     --------- #

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    ) = _calc_group_cast_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    work = dist.all_to_all_single(
        output=a2a_output,
        input=a2a_input,
        output_split_sizes=a2a_output_split_size,
        input_split_sizes=a2a_input_split_size,
        group=group,
        async_op=True,
    )

    return WorkWithPostProcessFn(
        work=work,
        post_process_fn=post_process_fn,
        sync=not async_op,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def group_reduce_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """

    Args:
        input: [sum(input_split_size),...]
        output: [sum(output_split_size),...]
        input_split_size: 0 <= len(input_split_size_list) <= group_size
        output_split_size: [M]
        dst_index: [N,?]
        src_indices: [M,?]

        HACK:
        **kwargs: additional keyword arguments,
        this kernel is for now based on all2all-v,
        thus introducing pre-/post-processing overhead
        on both tensor and meta info to be compatible with all2all-v input/output.
        Therefore, we add `kwargs` since the processing of meta info
        can be processed in advance, and just passed in through `kwargs` to reduce runtime overhead

    Returns:
        work + with_post_process_fn:
            work: Work | None
            post_process_fn: Callable[[torch.Tensor], torch.Tensor]


    NOTE(xiaowu):
        * 可以通过input_split_size_list把input变成list[splited_input], 其中
        每一个splited_input都必须发给一个rank, 然后通过dst_index_list来决定发给哪个rank.
        * 可以通过output_split_size_list把output变成list[splited_output], 其中
        每一个splited_output都可以从0个或多个src_rank那里reduce得到, 然后通过src_indices_list
        来决定从哪个rank那里reduce得到.
    """
    assert len(input_split_size_list) == len(dst_index_list)
    assert len(output_split_size_list) == len(src_indices_list)

    world_size = dist.get_world_size(group)

    # ---------    calc group reduce a2a args     --------- #

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    ) = _calc_group_reduce_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_index_list=dst_index_list,
        src_indices_list=src_indices_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    work = dist.all_to_all_single(
        output=a2a_output,
        input=a2a_input,
        output_split_sizes=a2a_output_split_size,
        input_split_sizes=a2a_input_split_size,
        group=group,
        async_op=True,
    )

    return WorkWithPostProcessFn(
        work=work,
        post_process_fn=post_process_fn,
        sync=not async_op,
    )
