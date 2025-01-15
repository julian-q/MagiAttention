import torch

from zeus.comm.functional import all_gather_v_func
from zeus.common.enum import AttnType
from zeus.meta.collection import DispatchMeta
from zeus.utils import nvtx


@nvtx.instrument_nvtx
def dispatch_func(
    x_global: torch.Tensor,
    meta: DispatchMeta,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Dispatch the global tensor `x_global` along its sequence dim following the meta info,
    and return the dispatched local tensor `x_local`

    Args:
        x_global (torch.Tensor): the global tensor to be dispatched
        meta (DispatchMeta): the meta info of the dispatch
        seq_dim (int): the sequence dimension of the tensor

    Returns:
        torch.Tensor: the dispatched local tensor `x_local`
    """

    assert (
        meta.attn_type is AttnType.SELF_ATTN
    ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

    x_split = torch.split(x_global, split_size_or_sections=meta.seqlens, dim=seq_dim)
    x_perm = torch.concat(
        [x_split[i] for i in meta.seqlens_perm_idxs],
        dim=seq_dim,
    )
    x_chunked = torch.chunk(
        x_perm,
        chunks=meta.num_chunks,
        dim=seq_dim,
    )
    x_local = torch.concat(
        [x_chunked[i] for i in meta.partitions_permed[meta.cp_rank]],
        dim=seq_dim,
    )

    return x_local


@nvtx.instrument_nvtx
def undispatch_func(
    x_local: torch.Tensor,
    meta: DispatchMeta,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Undispatch the local tensor `x_local` along its sequence dim following the meta info,
    and return the undispatched global tensor `x_global`

    Args:
        x_local (torch.Tensor): the local tensor to be undispatched
        meta (DispatchMeta): the meta info of the undispatch
        seq_dim (int): the sequence dimension of the tensor

    Returns:
        torch.Tensor: the undispatched global tensor `x_global`
    """

    assert (
        meta.attn_type is AttnType.SELF_ATTN
    ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

    x_gather = all_gather_v_func(x_local, group=meta.cp_group_nccl, dim=0)
    x_chunked = torch.chunk(
        x_gather,
        chunks=meta.num_chunks,
        dim=seq_dim,
    )
    x_perm = torch.concat(
        [x_chunked[i] for i in meta.partitions_unperm_idxs],
        dim=seq_dim,
    )
    x_split = torch.split(
        x_perm,
        split_size_or_sections=meta.seqlens_permed,
        dim=seq_dim,
    )
    x_global = torch.concat(
        [x_split[i] for i in meta.seqlens_unperm_idxs],
        dim=seq_dim,
    )

    return x_global
