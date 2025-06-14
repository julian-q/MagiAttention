# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor

# isort: off
# We need to import the CUDA kernels after importing torch
import flexible_flash_attention._C

# isort: on

flexible_flash_attention_cuda = torch.ops.flexible_flash_attention

def maybe_contiguous(x):
    if x is None:
        return x
    if isinstance(x, FakeTensor):
        return x.contiguous()      # don’t peek at .stride()
    return x if x.stride(-1) == 1 else x.contiguous()


def _flex_flash_attn_forward(
    q,
    k,
    v,
    q_ranges,
    k_ranges,
    max_seqlen_q,
    max_seqlen_k,
    attn_type_map,
    softmax_scale,
    softcap,
    deterministic,
    sm_margin,
    return_dtype,
    disable_fwd_atomic_reduction,
):
    q, k, v, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (q, k, v, q_ranges, k_ranges)
    ]

    if q_ranges.shape[0] == 0:
        # FIXME: This logic should be written in the cuda kernel, this is a temporary workaround
        ttk, nh, hd = q.shape
        out = torch.zeros_like(q)
        out_accum = torch.zeros_like(q, dtype=torch.float32)
        softmax_lse = torch.empty(nh, ttk, dtype=torch.float32)
        softmax_lse.fill_(-float("inf"))
    else:
        out, out_accum, softmax_lse = flexible_flash_attention_cuda.fwd(
            q,
            k,
            v,
            None,  # k_new, v_new
            None,
            None,  # qv
            # NOTE(julian-q): out should be None because we removed
            # the (out!) alias from the return signature to support 
            # fullgraph.
            # https://github.com/Dao-AILab/flash-attention/blob/db4baba2cae7be5a9155304636ba50a571c680a6/hopper/flash_api.cpp#L1628
            None,  # out
            q_ranges,
            k_ranges,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos, rotary_sin, seqlens_rotary
            None,
            None,
            None,  # q_descale, k_descale, v_descale
            None,
            None,
            softmax_scale,
            False,  # causal
            -1,  # window_size[0]
            -1,  # window_size[1]
            softcap,
            True,  # rotary_interleaved
            None,  # scheduler_metadata
            1,  # num_splits
            None,  # pack_gqa
            sm_margin,
            disable_fwd_atomic_reduction,
        )
    
    if disable_fwd_atomic_reduction:
        out = out
    else:
        out = out_accum

    if return_dtype is None:
        out = out.to(q.dtype)
    else:
        out = out.to(return_dtype)

    return out, softmax_lse


def _flex_flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    q_ranges,
    k_ranges,
    max_seqlen_q,
    max_seqlen_k,
    attn_type_map,
    softmax_scale,
    softcap,
    deterministic,
    sm_margin,
):
    dout, q, k, v, out, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (dout, q, k, v, out, q_ranges, k_ranges)
    ]

    if q_ranges.shape[0] == 0:
        # FIXME: This logic should be written in the cuda kernel, this is a temporary workaround
        ttk, nh, hd = q.shape
        dq_accum = torch.zeros_like(q, dtype=torch.float32)
        dk_accum = torch.zeros_like(k, dtype=torch.float32)
        dv_accum = torch.zeros_like(v, dtype=torch.float32)
        softmax_d = torch.zeros(nh, ttk, dtype=torch.float32)
    else:
        (
            _,
            _,
            _,
            softmax_d,
            _,
            dq_accum,
            dk_accum,
            dv_accum,
        ) = flexible_flash_attention_cuda.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            # NOTE(julian-q): dq, dk, dv should be None because we removed
            # the (dq!), (dk!), (dv!) aliases from the return signature
            # to support fullgraph.
            # https://github.com/Dao-AILab/flash-attention/blob/db4baba2cae7be5a9155304636ba50a571c680a6/hopper/flash_api.cpp#L1651
            None,
            None,
            None,  # dq, dk, dv
            q_ranges,
            k_ranges,
            None,  # cu_seqlens_q, cu_seqlens_k
            None,
            None,  # seqused_q, seqused_k
            None,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map,
            softmax_scale,
            False,  # causal
            -1,  # window_size[0]
            -1,  # window_size[1]
            softcap,
            deterministic,
            sm_margin,
        )

    return dq_accum.to(q.dtype), dk_accum.to(q.dtype), dv_accum.to(q.dtype), softmax_d


class FlexFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map,
        softmax_scale,
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
        return_dtype=None,
        disable_fwd_atomic_reduction=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert isinstance(
            max_seqlen_q, int
        ), "max_seqlen_q must be an int, otherwise would lead to performance degradation"
        assert isinstance(
            max_seqlen_k, int
        ), "max_seqlen_k must be an int, otherwise would lead to performance degradation"

        out, softmax_lse = _flex_flash_attn_forward(
            q,
            k,
            v,
            q_ranges,
            k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map,
            softmax_scale,
            softcap,
            deterministic,
            sm_margin,
            return_dtype,
            disable_fwd_atomic_reduction,
        )

        ctx.save_for_backward(
            q, k, v, out, softmax_lse, q_ranges, k_ranges, attn_type_map
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            q_ranges,
            k_ranges,
            attn_type_map,
        ) = ctx.saved_tensors
        dq, dk, dv, _ = _flex_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            q_ranges,
            k_ranges,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            attn_type_map,
            softmax_scale=ctx.softmax_scale,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
            sm_margin=ctx.sm_margin,
        )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    attn_type_map: Optional[torch.Tensor] = None,
    softmax_scale=None,
    softcap=0.0,
    deterministic=False,
    sm_margin=0,
    return_dtype=None,
    disable_fwd_atomic_reduction=False,
):
    """
    An interface similar to flash attention that doesn't require distributed environment, dispatch or undispatch.
    Directly call magi_attn_kernel to get attention output and lse. This is faster when you don't need context parallel.
    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        q_ranges (torch.Tensor): query ranges in the ref attn mask.
        k_ranges (torch.Tensor): key ranges in the ref attn mask.
        max_seqlen_q (int): Maximum sequence length of q_ranges.
        max_seqlen_k (int): Maximum sequence length of k_ranges.
        attn_type_map (torch.Tensor): Attention type map with dtype=torch.int32.
            0: full attention
            1: causal attention
            2: inverse causal attention
            3: bidirectional causal attention
        softmax_scale (float): Softmax scale.
        softcap (float): Softcap.
        deterministic (bool): Whether to use deterministic attention.
        sm_margin (int): the amount of SMs(streaming multiprocessors) reserved for communication.
        return_dtype (torch.dtype): Return dtype.
        disable_fwd_atomic_reduction (bool): Whether to disable forward atomic reduction.
            If you can ensure q_ranges has no overlap, you can set this to True for better performance.
            Overlap in q_ranges is defined as: if any two q_ranges have non-empty intersection, then there is overlap.
            For example, q_ranges = [[0, 15], [10, 20], [20, 30]] has overlap because [0, 15] and [10, 20] intersect.
            While q_ranges = [[0, 15], [15, 20], [20, 30]] has no overlap.
    Returns:
        out (torch.Tensor): Attention output tensor
        lse (torch.Tensor): Log-sum-exp values with dtype=torch.float32.
    Shape:
        q: (num_tokens_q, num_heads, head_dim)
        k: (num_tokens_kv, num_heads, head_dim)
        v: (num_tokens_kv, num_heads, head_dim)
        q_ranges: (num_ranges, 2)
        k_ranges: (num_ranges, 2)
        attn_type_map: (num_ranges, )
        out: (num_tokens_q, num_heads, head_dim)
        lse: (num_heads, num_tokens_q)
    NOTE: attn_type_map explanation:
        (In addition to the textual explanations provided below, feel free to check out our blog for a visual interpretation:
        https://sandai-org.github.io/MagiAttention/#flex-flash-attn)
        1. full attention
            If seqlen_q = 5 and seqlen_k = 2, the full mask is:
                1 1
                1 1
                1 1
                1 1
                1 1
            If seqlen_q = 2 and seqlen_k = 5, the full mask is:
                1 1 1 1 1
                1 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the full mask is:
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
        2: causal attention (bottom-right aligned)
            If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
                0 0
                0 0
                0 0
                1 0
                1 1
            if seqlen_q = 2 and seqlen_k = 5, the causal mask is:
                1 1 1 1 0
                1 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the causal mask is:
                1 0 0 0 0
                1 1 0 0 0
                1 1 1 0 0
                1 1 1 1 0
                1 1 1 1 1
        3: inverse causal attention (top-left aligned)
            if seqlen_q = 5 and seqlen_k = 2, the inverse causal mask is:
                1 1
                0 1
                0 0
                0 0
                0 0
            if seqlen_q = 2 and seqlen_k = 5, the inverse causal mask is:
                1 1 1 1 1
                0 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the inverse causal mask is:
                1 1 1 1 1
                0 1 1 1 1
                0 0 1 1 1
                0 0 0 1 1
                0 0 0 0 1
        4. bidirectional causal attention (top-left & bottom-right intersect-aligned)
            bidirectional causal attention mask is an 'and mask' of casual and inverse causal attention.
            if seqlen_q = 5 and seqlen_k = 2, the bidirectional causal mask is:
                0 0
                0 0
                0 0
                0 0
                0 0
            if seqlen_q = 2 and seqlen_k = 5, the bidirectional causal mask is:
                1 1 1 1 0
                0 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the bidirectional causal mask is:
                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 1 0
                0 0 0 0 1
    """
    assert not deterministic, "deterministic is not supported yet."
    return FlexFlashAttnFunc.apply(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map,
        softmax_scale,
        softcap,
        deterministic,
        sm_margin,
        return_dtype,
        disable_fwd_atomic_reduction,
    )

@torch.library.register_fake("flexible_flash_attention::fwd")
def _fwd(
    q,
    k,
    v,
    k_new = None,
    v_new = None,
    qv = None,
    out = None,
    q_ranges = None,
    k_ranges = None,
    cu_seqlens_q = None,
    cu_seqlens_k = None,
    cu_seqlens_k_new = None,
    seqused_q = None,
    seqused_k = None,
    max_seqlen_q = None,
    max_seqlen_k = None,
    attn_type_map = None,
    page_table = None,
    kv_batch_idx = None,
    leftpad_k = None,
    rotary_cos = None,
    rotary_sin = None,
    seqlens_rotary = None,
    q_descale = None,
    k_descale = None,
    v_descale = None,
    softmax_scale = None,
    causal = False,
    window_size_left = -1,
    window_size_right = -1,
    softcap = 0.0,
    rotary_interleaved = False,
    scheduler_metadata = None,
    num_splits = 0,
    pack_gqa = None,
    sm_margin = 0,
    disable_fwd_atomic_reduction = False,
):
    torch._check(q.shape[1:] == k.shape[1:])
    torch._check(q.shape[1:] == v.shape[1:])
    torch._check(q.dtype == k.dtype)
    torch._check(q.dtype == v.dtype)

    out = torch.empty_like(q)
    out_accum = torch.empty_like(q)
    softmax_lse = torch.empty(q.shape[1], q.shape[0])

    return out, out_accum, softmax_lse

def round_up_headdim(head_size: int) -> int:
    if head_size <= 64:
        return 64
    if head_size <= 96:
        return 96
    if head_size <= 128:
        return 128
    if head_size <= 192:
        return 192
    return 256

def get_kBlockM(arch: int, head_size: int, is_causal: bool, softcap: float, is_local: bool) -> int:
    if arch >= 90:
        if head_size <= 64:
            if is_causal and softcap > 0.0:
                return 96
            else:
                return 128
        elif head_size <= 96:
            return 64
        elif head_size <= 128:
            if is_causal or is_local or softcap > 0.0:
                return 64
            else:
                return 80
        else:
            return 64
    elif arch == 86 or arch == 89:
        if head_size <= 192:
            return 64
        else:
            return 32
    else:
        if head_size <= 64:
            return 128
        else:
            return 64

def round_multiple(x: int, m: int) -> int:
    return (x + m - 1) // m * m

@torch.library.register_fake("flexible_flash_attention::bwd")
def _bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq = None,
    dk = None,
    dv = None,
    q_ranges = None,
    k_ranges = None,
    cu_seqlens_q = None,
    cu_seqlens_k = None,
    seqused_q = None,
    seqused_k = None,
    max_seqlen_q = None,
    max_seqlen_k = None,
    attn_type_map = None,
    softmax_scale = None,
    is_causal = False,
    window_size_left = -1,
    window_size_right = -1,
    softcap = 0.0,
    deterministic = False,
    sm_margin = 0,
):
    # NOTE(julian-q): This logic from flash_api.cpp:
    # https://github.com/Dao-AILab/flash-attention/blob/db4baba2cae7be5a9155304636ba50a571c680a6/hopper/flash_api.cpp#L1300
    head_size = q.size(-1)
    head_size = round_up_headdim(head_size)
    # props = torch.cuda.get_device_properties()
    # arch = props.major * 10 + props.minor
    arch = 90
    is_local = (window_size_left >= 0 or window_size_right >= 0) and not is_causal
    kBlockM = get_kBlockM(arch, head_size, is_causal, softcap, is_local)
    seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    batch_size = q_ranges.size(0)
    num_heads = q.size(-2)
    softmax_d = torch.empty(batch_size, num_heads, seqlen_q_rounded, dtype=torch.float32)
    softmax_lse_log2 = torch.empty(batch_size, num_heads, seqlen_q_rounded, dtype=torch.float32)

    dq_accum = torch.empty_like(q)
    dk_accum = torch.empty_like(k)
    dv_accum = torch.empty_like(v)

    return dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum
