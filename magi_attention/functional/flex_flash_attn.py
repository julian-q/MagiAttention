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

# isort: off
# We need to import the CUDA kernels after importing torch
import flexible_flash_attention_cuda

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


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

    out, out_accum, softmax_lse = flexible_flash_attention_cuda.fwd(
        q,
        k,
        v,
        None,  # k_new, v_new
        None,
        None,  # qv
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
