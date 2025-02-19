import torch.distributed as dist

from zeus.common.config import OverlapConfig
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
    overlap_config: OverlapConfig,
) -> tuple[CommMeta, AttnCalcMeta]:
    """Calculate the communication and calculation meta from the dispatch meta

    Args:
        dispatch_meta_q (DispatchMeta): The dispatch meta for query
        dispatch_meta_k (DispatchMeta): The dispatch meta for key
        bucket_per_rank (list[AttnBucket]): The bucket per rank
        cp_group_nccl (dist.ProcessGroup): The NCCL process group
        cp_group_gloo (dist.ProcessGroup): The GLOO process group
        overlap_config (OverlapConfig): The overlap config, including the overlap mode, overlap degree, overlap chunk size, etc

    Returns:
        tuple[CommMeta, AttnCalcMeta]: The communication and calculation meta
    """

    attn_solver = AttnSolver(
        bucket_per_rank=bucket_per_rank,
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        cp_group_nccl=cp_group_nccl,
        cp_group_gloo=cp_group_gloo,
        overlap_config=overlap_config,
    )

    comm_meta = attn_solver.calc_comm_meta()
    calc_meta = attn_solver.calc_attn_calc_meta()

    assert comm_meta.overlap_degree == calc_meta.overlap_degree, (
        "The overlap degree is inconsistent between "
        f"comm meta ({comm_meta.overlap_degree}) and calc meta ({calc_meta.overlap_degree})."
    )

    # DE-BUG: log attn solver, comm meta and calc meta
    # from zeus.utils import write_rank
    # write_rank(repr(attn_solver), "attn_solver.log")
    # write_rank(repr(comm_meta), "comm_meta.log")
    # write_rank(repr(calc_meta), "calc_meta.log")

    return comm_meta, calc_meta
