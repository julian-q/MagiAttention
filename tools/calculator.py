from dffa.common import AttnRanges
from dffa.common.enum import AttnMaskType
from dffa.meta._calc_dispatch_meta import _calc_self_attn_areas


def calculate_attn_flops(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    num_heads_q: int,
    head_dim: int,
) -> dict[str, float]:
    attn_area = _calc_self_attn_areas(
        q_ranges,
        k_ranges,
        attn_mask_type,
        num_chunks=1,
        chunk_size=total_seqlen_q,
    ).area

    flops_fwd = 4 * attn_area * num_heads_q * head_dim
    flops_bwd = flops_fwd * 2.5  # 2.0(bwd) + 0.5(recompute)
    flops_1f1b = flops_fwd + flops_bwd

    return {
        "fwd": flops_fwd,
        "bwd": flops_bwd,
        "1f1b": flops_1f1b,
    }
