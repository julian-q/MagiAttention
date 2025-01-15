from zeus.common.enum import AttnMaskType
from zeus.common.ranges import AttnRanges
from zeus.meta.collection import DispatchMeta
from zeus.meta.container import AttnBucket


def calc_dispatch_meta_from_qk_ranges(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: AttnMaskType | list[AttnMaskType],
    is_same_source: bool,
    is_q_permutable: bool,
    is_k_permutable: bool,
    chunk_size: int,
    cp_size: int,
    cp_rank: int,
    overlap_degree: int,
) -> tuple[DispatchMeta, DispatchMeta, list[AttnBucket]]:
    """Calculate dispatch meta from query and key ranges

    Args:
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (AttnMaskType | list[AttnMaskType]): attn mask type

        is_same_source (bool): is query tensor and key tensor share the same source
        is_q_permutable (bool): is query tensor permutable
        is_k_permutable (bool): is key tensor permutable
        NOTE: e.g.
                1. for decoder-only transformer like gpt, it applies 'self-attn' as follows:
                    a) is_same_source is True
                    b) both q and k are permutable, as long as they are permuted in the same way.
                2. for encoder-decoder transformer like t5, it applies 'cross-attn' as follows:
                    a) is_same_source is True
                    b) q is permutable but k is not
                3. for multi-modal transformer with external encoders, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) both q and k are permutable, even if they are permuted in different ways

        chunk_size (int): chunk size to chunk the permutable tensor
        cp_size (int): context-parallel world size
        cp_rank (int): context-parallel local rank, ranging in [0,  cp_size)
        overlap_degree (int): the degree to shard the permutable tensor further
            into multiple stages for pipeline-style overlapping

    Returns:
        tuple[DispatchMeta, DispatchMeta]: dispatch_meta_q and dispatch_meta_k
        NOTE: When is_same_source is True, dispatch_meta_k should contain attributes
                that are mostly the same as those in dispatch_meta_q.
    """
    raise NotImplementedError
