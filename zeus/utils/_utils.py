import os
from typing import Any, Callable, List, Sequence, Tuple, TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from rich import print as rprint

from . import nvtx


def setup_dist_env(
    base_seed: int | None = None, seed_bias: Callable = lambda rank: 0
) -> Tuple[int, int, dist.ProcessGroup, str, int | None]:
    """set up distributed environment with NCCL backend,
    NOTE: the test script using this func to set up should be executed through torchrun
    """
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    manual_seed = None
    if base_seed is not None:
        manual_seed = base_seed + seed_bias(rank)
        torch.manual_seed(manual_seed)

    return rank, world_size, dist.group.WORLD, f"cuda:{rank}", manual_seed  # noqa: E231


def clearup_dist_env() -> None:
    dist.destroy_process_group()


def rprint_rank(msg: str, rank: int | None = None, width: int = 50):
    if rank is None or dist.get_rank() == rank:
        rank = dist.get_rank()
        rprint(
            f"\n{'-' * width}{' ' * 5}{rank=}{' ' * 5}{'-' * width}\n\n" + msg,
            flush=True,
        )


NestedIntList: TypeAlias = Union[List[int], Tuple[int, ...], Sequence["NestedIntList"]]


@nvtx.instrument_nvtx
def flatten_nested_list(nested_list: NestedIntList) -> List[int]:
    # Initialize a stack with the reversed nested list to process elements from left to right
    stack = list(nested_list[::-1])

    # Initialize an empty list to store the flattened elements
    flat_list: List[int] = []

    # Process the stack until all elements are handled
    while stack:
        item = stack.pop()  # Pop the last element from the stack
        if isinstance(item, (list, tuple)):
            # If the element is a list, reverse it and extend the stack with its elements
            stack.extend(item[::-1])  # type: ignore
        else:
            # If the element is not a list, add it to the flat list
            flat_list.append(item)  # type: ignore

    return flat_list  # Return the fully flattened list


def perm_idxs2unperm_idxs(perm_idxs: List[int]) -> List[int]:
    if not perm_idxs:
        return []

    unperm_idxs = [0] * len(perm_idxs)

    for i in range(len(perm_idxs)):
        unperm_idxs[perm_idxs[i]] = i

    return unperm_idxs


def wrap_to_list(x: Any, broadcast_to_length: int = 1) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x] * broadcast_to_length


def is_list_value_all(
    _list: list[Any], val: Any = None, just_same: bool = False
) -> bool:
    if not _list:
        return False

    if just_same:
        assert val is None, "val should be None when just_same is True"
        val = _list[0]

    return all(x == val for x in _list)


def is_list_type_all(
    _list: list[Any], _type: Any = None, just_same: bool = False
) -> bool:
    if not _list:
        return False

    if just_same:
        assert _type is None, "_type should be None when just_same is True"
        _type = type(_list[0])

    return all(isinstance(x, _type) for x in _list)


def repr_matrix(matrix: np.ndarray) -> str:
    repr_str = ""
    sep = "    "

    nrows, ncols = matrix.shape[0], matrix.shape[1]
    row_idx_width = len(str(nrows))
    col_idx_width = len(str(ncols))
    to_str = lambda x: f"{x: <{col_idx_width}}"  # noqa

    repr_str += " " * (row_idx_width + 3) + sep.join(map(to_str, range(ncols))) + "\n"
    col_width = len(repr_str)
    repr_str += " " * 3 + "+" + "-" * (col_width - 3) + ">" + "\n"

    for row_idx, row in enumerate(matrix):
        repr_str += (
            f"{row_idx: <{row_idx_width}}" + " | " + sep.join(map(to_str, row)) + "\n"
        )

    return repr_str


def vis_matrix(
    matrix: np.ndarray,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    val_ticks: List[float] | None = None,
    format_ticks: Callable | None = None,
    save_path: str | None = None,
) -> None:
    cmap = plt.cm.gray
    nrows, ncols = matrix.shape[0], matrix.shape[1]

    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    ax.set_xticks(np.arange(ncols), np.arange(ncols))
    ax.set_yticks(np.arange(nrows), np.arange(nrows))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = plt.colorbar(cax)

    if val_ticks is not None:
        cbar.set_ticks(val_ticks)
    if format_ticks is not None:
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

    plt.show()

    if save_path is not None:
        plt.savefig(save_path)
