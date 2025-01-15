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
        1. qk来源是1个
            a. q能动, k能动 -> self-attn
            b. q能动, k不能动 -> invalid
            c. q不能动, k能动 -> invalid
            d. q不能动, k不能动 -> 输出meta

        2. qk来源是2个
            a. q能动, k能动 -> 纯cross attn
            b. q能动, k不能动 -> t5
            c. q不能动, k能动 -> multi-modal
            d. q不能动, k不能动 -> 输出meta

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
