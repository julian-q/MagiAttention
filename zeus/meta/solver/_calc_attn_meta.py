import torch.distributed as dist

from zeus.meta.collection.calc_meta import AttnCalcMeta
from zeus.meta.collection.comm_meta import CommMeta
from zeus.meta.collection.dispatch_meta import DispatchMeta
from zeus.meta.container.bucket import AttnBucket

from .dist_attn_solver import AttnSolver


def calc_attn_meta_from_dispatch_meta(
    dispatch_meta_q: DispatchMeta,
    dispatch_meta_k: DispatchMeta,
    bucket_per_rank: list[AttnBucket],
    cp_group_nccl: dist.ProcessGroup,
    cp_group_gloo: dist.ProcessGroup,
    overlap_degree: int,
) -> tuple[CommMeta, AttnCalcMeta]:
    attn_solver = AttnSolver(
        bucket_per_rank=bucket_per_rank,
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_kv=dispatch_meta_k,
        cp_group_nccl=cp_group_nccl,
        cp_group_gloo=cp_group_gloo,
        overlap_degree=overlap_degree,
    )

    comm_meta = attn_solver.calc_comm_meta()
    calc_meta = attn_solver.calc_calc_meta()

    return comm_meta, calc_meta
