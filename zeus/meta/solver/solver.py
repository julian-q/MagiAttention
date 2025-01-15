import torch.distributed as dist

from zeus.common.ranges import AttnRanges


def calc_dispatch_meta_from_qk_ranges(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    is_same_source: bool,
    is_q_permutable: bool,
    is_k_permutable: bool,
    chunk_size: int,
    cp_size: int,
    cp_rank: int,
    overlap_degree: int,
):
    """
    计算出dispatch meta
    """
    pass


def clac_attn_meta_from_dispatch_meta(
    dispatch_meta,
    cp_group_nccl: dist.ProcessGroup,
    cp_group_gloo: dist.ProcessGroup,
):
    """
    从dispatch meta中计算出attn meta
    需要使用到dispatch meta中的bucket_per_rank
    每个bucket需要有以下attr:
        - host_ranges_global
        - host_ranges_local
        - remote_ranges_global
        - remote_ranges_local
    """
    pass
