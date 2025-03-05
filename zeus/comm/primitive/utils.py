from collections import defaultdict
from functools import partial
from itertools import accumulate, chain, pairwise
from logging import getLogger
from typing import Callable, TypeAlias

import torch
import torch.distributed as dist

from zeus.common.range import NaiveRange
from zeus.common.ranges import NaiveRanges
from zeus.utils import nvtx

logger = getLogger("zeus")

__all__ = [
    "_calc_group_cast_a2a_args",
    "_calc_group_reduce_a2a_args",
    "_calc_group_cast_a2a_input_meta_args",
    "_calc_group_cast_a2a_output_meta_args",
    "_calc_group_reduce_a2a_input_meta_args",
    "_calc_group_reduce_a2a_output_meta_args",
    "_trans_with_dim0",
    "_get_dims_as_trans_with_dim0",
]


def _seqlens2curanges(
    seqlens: list[int],
) -> NaiveRanges:
    return list(pairwise(accumulate([0] + seqlens)))


RangesWithRank: TypeAlias = list[tuple[NaiveRange, int]]


# ------------------        utils for group cast collective       ------------------ #


@nvtx.instrument_nvtx
def _unpermute_tensor(
    tensor: torch.Tensor,
    unpermute_index_list: list[int],
    tensor_size_list: list[int],
) -> torch.Tensor:
    """unpermute a2a output to output
    as a post-processing func for group_cast_collective
    """

    unperm_tensor = torch.empty(
        [sum(tensor_size_list[i] for i in unpermute_index_list), *tensor.shape[1:]],
        dtype=tensor.dtype,
        device=tensor.device,
    )

    if unperm_tensor.numel() > 0:
        tensor_list = torch.split(tensor, tensor_size_list, dim=0)
        unperm_tensor = torch.cat(
            [tensor_list[i] for i in unpermute_index_list],
            dim=0,
            out=unperm_tensor,
        )

    return unperm_tensor


@nvtx.instrument_nvtx
def _calc_group_cast_a2a_input_meta_args(
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    world_size: int,
) -> tuple[RangesWithRank, list[int]]:
    input_size_ranges = _seqlens2curanges(input_split_size_list)

    a2a_input_size_ranges_with_rank: RangesWithRank = sorted(
        list(
            chain(
                *[
                    [(input_size_ranges[i], dst_rank) for dst_rank in dst_indices]
                    for i, dst_indices in enumerate(dst_indices_list)
                ]
            )
        ),
        key=lambda x: x[1],
    )

    a2a_input_split_size_dict: dict[int, int] = defaultdict(int)
    for (start, end), rank in a2a_input_size_ranges_with_rank:
        a2a_input_split_size_dict[rank] += end - start
    a2a_input_split_size = [
        a2a_input_split_size_dict[rank] for rank in range(world_size)
    ]

    return a2a_input_size_ranges_with_rank, a2a_input_split_size


def _calc_group_cast_a2a_input_args(
    input: torch.Tensor,
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int]]:
    # -----     group_cast_a2a_input meta args     ----- #

    # check if pre-calculated
    a2a_input_size_ranges_with_rank = kwargs.get(
        "a2a_input_size_ranges_with_rank", None
    )
    a2a_input_split_size = kwargs.get("a2a_input_split_size", None)

    if a2a_input_size_ranges_with_rank is None or a2a_input_split_size is None:
        (
            a2a_input_size_ranges_with_rank,
            a2a_input_split_size,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            world_size=world_size,
        )

    # -----     group_cast_a2a_input tensor sargs     ----- #

    a2a_input = torch.empty(
        [sum(a2a_input_split_size), *input.shape[1:]],
        dtype=input.dtype,
        device=input.device,
    )

    # NOTE: a2a_input.numel() == 0 is a corner case when
    # dst_indices_list is all-empty like [[], [], [], [], []]
    # then a2a_input_split_size is all-zero like [0, 0, 0, 0]
    if a2a_input.numel() > 0:
        a2a_input = torch.cat(
            [
                input[start:end]
                for ((start, end), rank) in a2a_input_size_ranges_with_rank
            ],
            dim=0,
            out=a2a_input,
        )

    return a2a_input, a2a_input_split_size


@nvtx.instrument_nvtx
def _calc_group_cast_a2a_output_meta_args(
    output_split_size_list: list[int],
    src_index_list: list[int],
    world_size: int,
) -> tuple[list[int], list[int], list[int]]:
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

    return (
        a2a_output_split_size,
        a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list,
    )


def _calc_group_cast_a2a_output_args(
    output: torch.Tensor,
    output_split_size_list: list[int],
    src_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
    # -----     group_cast_a2a_output meta args     ----- #

    # check if pre-calculated
    a2a_output_split_size = kwargs.get("a2a_output_split_size", None)
    a2a_output_unpermute_index_list = kwargs.get(
        "a2a_output_unpermute_index_list", None
    )
    a2a_output_tensor_size_list = kwargs.get("a2a_output_tensor_size_list", None)

    if (
        a2a_output_split_size is None
        or a2a_output_unpermute_index_list is None
        or a2a_output_tensor_size_list is None
    ):
        (
            a2a_output_split_size,
            a2a_output_unpermute_index_list,
            a2a_output_tensor_size_list,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            world_size=world_size,
        )

    # -----     group_cast_a2a_output tensor sargs     ----- #

    a2a_output = output

    return (
        a2a_output,
        a2a_output_split_size,
        a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def _calc_group_cast_a2a_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    Callable[[torch.Tensor], torch.Tensor],
]:
    # ---------    calc a2a_input_split_size and a2a input     --------- #

    a2a_input, a2a_input_split_size = _calc_group_cast_a2a_input_args(
        input=input,
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    calc a2a_output_split_size and a2a output     --------- #

    (
        a2a_output,
        a2a_output_split_size,
        a2a_output_unpermute_index_list,
        a2a_output_tensor_size_list,
    ) = _calc_group_cast_a2a_output_args(
        output=output,
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    prepare post-process fn    --------- #

    post_process_fn = partial(
        _unpermute_tensor,
        unpermute_index_list=a2a_output_unpermute_index_list,
        tensor_size_list=a2a_output_tensor_size_list,
    )

    # DE-BUG
    logger.debug(
        f"RANK {dist.get_rank()}:: args for group_cast_collective: {input.shape=}, {output.shape=}, "
        f"{input_split_size_list=}, {output_split_size_list=}, {dst_indices_list=}, {src_index_list=}, "
        f"args: {a2a_input.shape=}, {a2a_output.shape=}, {a2a_output_split_size=}, {a2a_input_split_size=}, "
    )

    return (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    )


# ------------------        utils for group reduce collective       ------------------ #


# TODO: fuse this kernel in the future
# FIXME: if using torch.compile, it's fused incompletely w/o performance gain
# what's worse, the re-compilation in online exps would hang the comm
@nvtx.instrument_nvtx
def _reduce_to_tensor(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    a2a_output_reduce_ranges_list: list[NaiveRanges],
    output_size_ranges: NaiveRanges,
) -> torch.Tensor:
    """sum-reduce a2a output to output
    as a post-processing func for group_reduce_collective
    """

    for (out_start, out_end), reduce_ranges in zip(
        output_size_ranges, a2a_output_reduce_ranges_list
    ):
        out_slice = output[out_start:out_end]
        for reduce_start, reduce_end in reduce_ranges:
            out_slice.add_(a2a_output[reduce_start:reduce_end])

    return output


@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_input_meta_args(
    input_split_size_list: list[int],
    dst_index_list: list[int],
    world_size: int,
) -> tuple[RangesWithRank, list[int]]:
    input_size_ranges = _seqlens2curanges(input_split_size_list)
    a2a_input_size_ranges_with_rank: RangesWithRank = sorted(
        [(input_size_ranges[i], dst_rank) for i, dst_rank in enumerate(dst_index_list)],
        key=lambda x: x[1],
    )

    a2a_input_split_size_dict: dict[int, int] = defaultdict(int)
    for (start, end), rank in a2a_input_size_ranges_with_rank:
        a2a_input_split_size_dict[rank] += end - start
    a2a_input_split_size = [
        a2a_input_split_size_dict[rank] for rank in range(world_size)
    ]

    return a2a_input_size_ranges_with_rank, a2a_input_split_size


def _calc_group_reduce_a2a_input_args(
    input: torch.Tensor,
    input_split_size_list: list[int],
    dst_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int]]:
    # -----     group_reduce_a2a_input meta args     ----- #

    # check if pre-calculated
    a2a_input_size_ranges_with_rank = kwargs.get(
        "a2a_input_size_ranges_with_rank", None
    )
    a2a_input_split_size = kwargs.get("a2a_input_split_size", None)

    if a2a_input_size_ranges_with_rank is None or a2a_input_split_size is None:
        (
            a2a_input_size_ranges_with_rank,
            a2a_input_split_size,
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=input_split_size_list,
            dst_index_list=dst_index_list,
            world_size=world_size,
        )

    # -----     group_reduce_a2a_input tensor args     ----- #

    a2a_input = torch.empty(
        [sum(a2a_input_split_size), *input.shape[1:]],
        dtype=input.dtype,
        device=input.device,
    )

    # NOTE: a2a_input.numel() == 0 is a corner case when
    # dst_index_list is empty like []
    # then a2a_input_split_size is all-zero like [0, 0, 0, 0]
    if a2a_input.numel() > 0:
        a2a_input = torch.cat(
            [
                input[start:end]
                for ((start, end), rank) in a2a_input_size_ranges_with_rank
            ],
            dim=0,
            out=a2a_input,
        )

    return a2a_input, a2a_input_split_size


@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_output_meta_args(
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
) -> tuple[list[int], list[NaiveRanges], NaiveRanges]:
    # phase1 meta
    a2a_output_split_size = [0 for _ in range(world_size)]
    size_src_index_i_list = []
    idx = 0
    for output_split_size, src_indices in zip(output_split_size_list, src_indices_list):
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
    num_src_list = [len(src_indices) for src_indices in src_indices_list]

    # phase2 meta
    a2a_output_size_ranges = _seqlens2curanges(a2a_output_tensor_size_list)
    output_size_ranges = _seqlens2curanges(output_split_size_list)
    cum_src_ranges = _seqlens2curanges(num_src_list)
    a2a_output_reduce_ranges_list: list[NaiveRanges] = []
    for start, end in cum_src_ranges:
        a2a_output_reduce_ranges_list.append(
            [
                a2a_output_size_ranges[index]
                for index in a2a_output_unpermute_index_list[start:end]
            ]
        )

    return (
        a2a_output_split_size,
        a2a_output_reduce_ranges_list,
        output_size_ranges,
    )


def _calc_group_reduce_a2a_output_args(
    output: torch.Tensor,
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int], list[NaiveRanges], NaiveRanges]:
    # -----     group_reduce_a2a_output meta args     ----- #

    # check if pre-calculated
    a2a_output_split_size = kwargs.get("a2a_output_split_size", None)
    a2a_output_reduce_ranges_list = kwargs.get("a2a_output_reduce_ranges_list", None)
    output_size_ranges = kwargs.get("output_size_ranges", None)

    if (
        a2a_output_split_size is None
        or a2a_output_reduce_ranges_list is None
        or output_size_ranges is None
    ):
        (
            a2a_output_split_size,
            a2a_output_reduce_ranges_list,
            output_size_ranges,
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_indices_list=src_indices_list,
            world_size=world_size,
        )

    # -----     group_reduce_a2a_output tensor args     ----- #

    a2a_output = torch.empty(
        [sum(a2a_output_split_size), *output.shape[1:]],
        device=output.device,
        dtype=output.dtype,
    )

    return (
        a2a_output,
        a2a_output_split_size,
        a2a_output_reduce_ranges_list,
        output_size_ranges,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    **kwargs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    Callable[[torch.Tensor], torch.Tensor],
]:
    # ---------    calc a2a_input_split_size and a2a input     --------- #

    a2a_input, a2a_input_split_size = _calc_group_reduce_a2a_input_args(
        input=input,
        input_split_size_list=input_split_size_list,
        dst_index_list=dst_index_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    calc a2a_output_split_size and a2a output     --------- #

    (
        a2a_output,
        a2a_output_split_size,
        a2a_output_reduce_ranges_list,
        output_size_ranges,
    ) = _calc_group_reduce_a2a_output_args(
        output=output,
        output_split_size_list=output_split_size_list,
        src_indices_list=src_indices_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    prepare post process fn     --------- #

    post_process_fn = partial(
        _reduce_to_tensor,
        a2a_output=a2a_output,
        a2a_output_reduce_ranges_list=a2a_output_reduce_ranges_list,
        output_size_ranges=output_size_ranges,
    )

    # DE-BUG
    logger.debug(
        f"RANK {dist.get_rank()}:: args for group_reduce_collective: {input.shape=}, {output.shape=}, "
        f"{input_split_size_list=}, {output_split_size_list=}, {dst_index_list=}, {src_indices_list=}. "
        f"args: {a2a_input.shape=}, {a2a_output.shape=}, {a2a_output_split_size=}, {a2a_input_split_size=}, "
    )

    return (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    )


# ------------------        utils for all-gather-v       ------------------ #


def _trans_with_dim0(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    is_first_dim = dim == 0 or (dim == -1 and len(x.shape) == 1)

    if not is_first_dim:
        x = x.transpose(0, dim)
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _get_dims_as_trans_with_dim0(
    x_shape: list[int],
    dim: int = 0,
) -> tuple[int, list[int]]:
    shape_len = len(x_shape)
    assert dim == -1 or 0 <= dim < len(
        x_shape
    ), f"dim should be in [0, {shape_len - 1}) or -1"

    this_dim = x_shape[dim]

    other_dims = x_shape.copy()
    other_dims[0] = this_dim
    other_dims[dim] = x_shape[0]
    other_dims = other_dims[1:]

    return this_dim, other_dims


# ------------------        utils for scatter-v       ------------------ #
