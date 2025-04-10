"""
comm
common
    range -> ranges -> bucket
-------------------------------------
solver
    meta_solver
        from_ranges

    dispatch_meta
        slice: q_range, k_range
            1. qk来源是不是一个（2种）
            2. q能不能动、k能不能动 （4种）

-------------------------------------
        1. is_same_source
            a. (Done) q is permutable, k is permutable -> self-attn
            b. q is permutable, k is not permutable -> invalid
            c. q is not permutable, k is permutable -> invalid
            d. q is not permutable, k is not permutable -> output meta

        2. is_not_same_source
            a. q is permutable, k is permutable -> pure cross attn
            b. q is permutable, k is not permutable -> t5
            c. q is not permutable, k is permutable -> multi-modal
            d. (TODO) q is not permutable, k is not permutable -> output meta

    comm_meta
        local, stage0, ..., stageN

    attn_meta
-------------------------------------
functional
    dispatch, undispatch
    dist_attn: dist_attn_runtime
        dist_attn_runtime
TODO:
    1. causal
    2. multi-stage-overlap
    3. cross-attn
    4. load-balance
        * token-balance
        * minimize comm size
        * minimize compute budget

    4. comm-kernel
    5. abitrary-attn-mask(V2)
"""

import os

from . import config
from .dist_attn_runtime_mgr import init_dist_attn_runtime_mgr

__all__ = [
    "init_dist_attn_runtime_mgr",
    "is_sanity_check_enable",
    "config",
]


def is_sanity_check_enable() -> bool:
    """
    Toggling this env variable to 1 can enable many sanity check codes inside dffa
    which is only supposed to be used for testing or debugging,
    since these codes involve performance overhead
    """
    return os.environ.get("DFFA_SANITY_CHECK", "0") == "1"


def is_sdpa_backend_enable() -> bool:
    """
    Toggling this env variable to 1 can switch the attn kernel backend
    from ffa to sdpa-math, to support higher precision like fp32, fp64,
    which is only supposed to be used for testing or debugging,
    since the performance is not acceptable
    """
    return os.environ.get("DFFA_SDPA_BACKEND", "0") == "1"


def is_causal_mask_enable() -> bool:
    """
    Toggle this env variable to 1 to allow causal mask
    NOTE: This flag is only used during experimental stage
    needed to be removed when causal mask is fully supported,
    both in functionality and performance
    """
    return os.environ.get("DFFA_SUPPORT_CAUSAL_MASK", "0") == "1"
