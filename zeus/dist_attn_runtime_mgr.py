import torch
import torch.distributed as dist

from zeus.common import AttnRanges
from zeus.common.enum import AttnMaskType
from zeus.config import DistAttnConfig
from zeus.functional.dispatch import dispatch_func, undispatch_func
from zeus.functional.dist_attn import DistFlashAttnRuntime, dist_attn_func
from zeus.meta import (
    calc_attn_meta_from_dispatch_meta,
    calc_dispatch_meta_from_qk_ranges,
)
from zeus.meta.collection import DispatchMeta
from zeus.meta.container import AttnBucket
from zeus.meta.solver.dist_attn_solver import DistAttnSolver


class DistAttnRuntimeMgr:
    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        q_dispatch_meta: DispatchMeta,
        k_dispatch_meta: DispatchMeta,
        dist_attn_config: DistAttnConfig,
        attn_solver: DistAttnSolver,
        dist_attn_runtime: DistFlashAttnRuntime,
    ):
        self.cp_group = cp_group
        self.q_dispatch_meta = q_dispatch_meta
        self.k_dispatch_meta = k_dispatch_meta
        self.dist_attn_config = dist_attn_config
        self.attn_solver = attn_solver
        self.dist_attn_runtime = dist_attn_runtime

    def dispatch_qo(self, q_or_o: torch.Tensor) -> torch.Tensor:
        q_or_o = dispatch_func(
            x_global=q_or_o,
            group=self.cp_group,
            meta=self.q_dispatch_meta,
        )
        return q_or_o

    def dispatch_kv(self, k_or_v: torch.Tensor) -> torch.Tensor:
        k_or_v = dispatch_func(
            x_global=k_or_v,
            group=self.cp_group,
            meta=self.k_dispatch_meta,
        )
        return k_or_v

    def undispatch_qo(self, q_or_o: torch.Tensor) -> torch.Tensor:
        q_or_o = undispatch_func(
            x_local=q_or_o,
            group=self.cp_group,
            meta=self.q_dispatch_meta,
        )
        return q_or_o

    def undispatch_kv(self, k_or_v: torch.Tensor) -> torch.Tensor:
        k_or_v = undispatch_func(
            x_local=k_or_v,
            group=self.cp_group,
            meta=self.k_dispatch_meta,
        )
        return k_or_v

    def calc_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return dist_attn_func(q, k, v, self.dist_attn_runtime)

    @property
    def bucket(self) -> AttnBucket:
        return self.attn_solver.bucket

    @property
    def host_q_ranges_global(self) -> AttnRanges:
        return self.attn_solver.host_rank_entry_this_rank.host_q_ranges_global

    @property
    def host_k_ranges_global(self) -> AttnRanges:
        return self.attn_solver.host_rank_entry_this_rank.host_k_ranges_global

    @property
    def remote_k_ranges_global(self) -> AttnRanges:
        return self.attn_solver.host_rank_entry_this_rank.remote_k_ranges_global


def init_dist_attn_runtime_mgr(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: AttnMaskType | list[AttnMaskType],
    total_seqlen_q: int,
    total_seqlen_k: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup,
    is_same_source: bool,
    is_q_permutable: bool,
    is_k_permutable: bool,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
) -> DistAttnRuntimeMgr:
    """

    Args:
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (AttnMaskType | list[AttnMaskType]): attn mask type (list)

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

        chunk_size (int): chunk size to chunk the permutable tensor

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now

        is_same_source (bool): is query tensor and key tensor share the same source
        is_q_permutable (bool): is query tensor permutable
        is_k_permutable (bool): is key tensor permutable
        NOTE: e.g.
                1. for decoder-only transformer like gpt, it applies 'self-attn' as follows:
                    a) is_same_source is True
                    b) both q and k are permutable, as long as they are permuted in the same way.
                2. for encoder-decoder transformer like t5, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) q is permutable but k is not
                3. for multi-modal transformer with external encoders, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) q is unpermutable cuz of self-attn, but k is permutable even in a different way

        dist_attn_config (DistAttnConfig): dist attn config

    Returns:
        DistAttnRuntimeMgr: dist attn runtime mgr

    Example::
        >>> dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type=AttnMaskType.FULL,
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     chunk_size=512,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
        ...     is_same_source=True,
        ...     is_q_permutable=True,
        ...     is_k_permutable=True,
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=OverlapAlgType.UNIFORM,
        ...         ),
        ...     ),
        ... )
        >>> # Dispatch global query tensor to local query tensor
        >>> local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        >>> # Dispatch global key tensor to local key tensor
        >>> local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        >>> # Dispatch global value tensor to local value tensor
        >>> local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)
        >>> # Calculate local attention result
        >>> local_out = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)
        >>> # Gather local attention results to global result
        >>> total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
    """

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    q_dispatch_meta, k_dispatch_meta, attn_buckets = calc_dispatch_meta_from_qk_ranges(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        chunk_size=chunk_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        dispatch_config=dist_attn_config.dispatch_config,
        is_same_source=is_same_source,
        is_q_permutable=is_q_permutable,
        is_k_permutable=is_k_permutable,
    )

    comm_meta, attn_calc_meta, attn_solver = calc_attn_meta_from_dispatch_meta(
        dispatch_meta_q=q_dispatch_meta,
        dispatch_meta_k=k_dispatch_meta,
        bucket_per_rank=attn_buckets,
        cp_group=cp_group,
        overlap_config=dist_attn_config.overlap_config,
    )

    dist_attn_runtime = DistFlashAttnRuntime(
        comm_meta=comm_meta,
        calc_meta=attn_calc_meta,
        cp_group_kv=cp_group,
        cp_group_dkv=cp_group,
        deterministic=dist_attn_config.deterministic,
    )

    return DistAttnRuntimeMgr(
        cp_group,
        q_dispatch_meta,
        k_dispatch_meta,
        dist_attn_config,
        attn_solver,
        dist_attn_runtime,
    )
