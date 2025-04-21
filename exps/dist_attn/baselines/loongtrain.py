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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import transformer_engine  # noqa
import transformer_engine_torch as tex
from einops import rearrange, repeat
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward as flash_attn_varlen_bwd,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward as flash_attn_varlen_fwd,
)
from transformer_engine.pytorch.attention import get_cu_seqlens_on_cp_rank

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

from .interface import AttnBaselineInterface
from .teusp_utils import (
    RingComm,
    _get_chunk_indices_on_cp_rank,
    _get_zigzag2d_varlen_unpad_data,
    _SeqAllToAll,
    fa_varlen_lse_pad,
    fa_varlen_lse_unpad,
    fa_varlen_thd_pad,
    fa_varlen_thd_unpad,
    flash_attn_fwd_softmax_lse_correction,
    get_distributed_rank,
    get_distributed_world_size,
)

softmax_lse_in_packed_format = True


@dataclass
class PackedSeqParams:
    """
    parameters to ZigZagTEUSPAttnVarlenFunc and dispatch for the
    `thd` (packed) sequence format
    """

    indices: torch.Tensor = None
    indices_in_thd_padded: torch.Tensor = None
    cu_seqlens: torch.Tensor = None
    cu_seqlens_padded: torch.Tensor = None
    max_seqlen_in_batch: int = 0
    max_seqlen_in_padded: int = 0
    first_axis_dim: int = 0


class ParallelMode(Enum):
    HEAD = "head"
    CONTEXT = "context"
    INTER_WINDOW = "inter_window"
    INTRA_WINDOW = "intra_window"
    DKV_INTER_WINDOW = "dkv_inter_window"
    DKV_INTRA_WINDOW = "dkv_intra_window"


def zigzag_loongtrain_dispatch(
    input,
    packed_seq_params,
    cp_size_a2a,
    rank_a2a,
    cp_size_p2p,
    rank_p2p,
    *args,
    **kwargs,
):
    assert input.ndim >= 2
    total_padded_len = packed_seq_params.cu_seqlens_padded[-1].item()
    packed_seq_params.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
    second_dim = other_shape.numel()
    # remove thd origin pad
    unpad_input = torch.gather(
        rearrange(input, "b ... -> b (...)"),
        0,
        repeat(packed_seq_params.indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)
    # pad to thd
    thd_input = torch.zeros(
        total_padded_len, *input.shape[1:], device=input.device, dtype=input.dtype
    )
    thd_input[packed_seq_params.indices_in_thd_padded] = unpad_input
    # load balance chunk
    chunk_indices = _get_chunk_indices_on_cp_rank(
        packed_seq_params.cu_seqlens_padded,
        cp_size_a2a,
        rank_a2a,
        cp_size_p2p,
        rank_p2p,
    )
    return (
        torch.gather(
            rearrange(thd_input, "b ... -> b (...)"),
            0,
            repeat(chunk_indices, "z -> z d", d=second_dim),
        )
        .reshape(-1, *other_shape)
        .contiguous()
    )


# reorder after all gather
def zigzag_reorder_undispatch_thd(
    local_r_group, cp_size_p2p, cu_seqlens_padded, *args, **kwargs
):
    seq_chunks = []
    batch = len(cu_seqlens_padded) - 1
    cu_seqlens_padded_local = cu_seqlens_padded // cp_size_p2p
    seqlens_padded_local = cu_seqlens_padded_local[1:] - cu_seqlens_padded_local[:-1]
    offset = 0
    for i in range(batch):
        seq = []
        seqlen = seqlens_padded_local[i] // 2
        offset = cu_seqlens_padded_local[i]
        for j in range(cp_size_p2p):
            seq.append(local_r_group[j][offset : offset + seqlen])
        for j in range(cp_size_p2p - 1, -1, -1):
            seq.append(local_r_group[j][offset + seqlen : offset + 2 * seqlen])
        seq_chunks.append(torch.cat(seq, dim=0))
    return torch.cat(seq_chunks, dim=0)


def zigzag_loongtrain_undispatch(
    x_local,
    packed_seq_params,
    world_size,
    cp_group,
    *args,
    **kwargs,
):
    input_dim = x_local.ndim
    assert input_dim >= 2
    total_seqlen, *other_shape = x_local.shape

    # Get the sizes of the two-level CP groups
    cp_group_a2a = cp_group[ParallelMode.HEAD]
    cp_group_p2p = cp_group[ParallelMode.CONTEXT]

    ud = dist.get_world_size(cp_group_a2a)
    rd = dist.get_world_size(cp_group_p2p)

    assert (
        ud * rd == world_size
    ), "Current two-level CP groups need ud*rd == world_size!"

    # ulysess all gather
    local_u_group = [torch.empty_like(x_local) for _ in range(ud)]
    dist.all_gather(local_u_group, x_local, group=cp_group_a2a)
    x_local_ring = torch.cat(local_u_group, dim=0)

    # ring all gather
    local_r_group = [torch.empty_like(x_local_ring) for _ in range(rd)]
    dist.all_gather(local_r_group, x_local_ring, group=cp_group_p2p)
    thd_input = zigzag_reorder_undispatch_thd(
        local_r_group, rd, packed_seq_params.cu_seqlens_padded
    )

    # unpad
    unpad_input = thd_input[packed_seq_params.indices_in_thd_padded]
    unpad_input = rearrange(unpad_input, "b ... -> b (...)")

    # pad
    x_global = torch.zeros(
        packed_seq_params.first_axis_dim,
        unpad_input.shape[1],
        device=unpad_input.device,
        dtype=unpad_input.dtype,
    )
    x_global.scatter_(
        0,
        repeat(packed_seq_params.indices, "z -> z d", d=unpad_input.shape[1]),
        unpad_input,
    )
    x_global = x_global.reshape(
        packed_seq_params.first_axis_dim, *other_shape
    ).contiguous()

    return x_global


# transformer_engine_torch update
def update_lse(softmax_lse, block_lse, cu_seqlens_q_padded, is_half):
    # update lse
    if softmax_lse is None:
        softmax_lse = torch.clone(block_lse).to(torch.double)
    elif not is_half:
        flash_attn_fwd_softmax_lse_correction(softmax_lse, block_lse)
    else:
        tex.thd_second_half_lse_correction(
            softmax_lse,
            block_lse,
            cu_seqlens_q_padded,
            softmax_lse_in_packed_format,
        )
    torch.cuda.synchronize()
    return softmax_lse


def update_out(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, is_half):
    # update out
    if not is_half:
        tex.thd_out_correction(
            out,
            block_out,
            softmax_lse,
            block_lse,
            cu_seqlens_q_padded,
            False,  # maybe not is_half
            softmax_lse_in_packed_format,  # softmax_lse_in_packed_format
        )
    else:
        tex.thd_out_correction(
            out,
            block_out,
            softmax_lse,
            block_lse,
            cu_seqlens_q_padded,
            True,  # maybe is_half
            softmax_lse_in_packed_format,  # softmax_lse_in_packed_format
        )
    torch.cuda.synchronize()


# transformer_engine_torch update
def update_grad_dq(grad, grad_, step, is_half, cu_seqlens_padded):
    if step == 0:
        if is_half:
            tex.thd_grad_correction(grad, grad_, cu_seqlens_padded, "none", "copy")
        else:
            grad.copy_(grad_)
    else:
        if is_half:
            tex.thd_grad_correction(grad, grad_, cu_seqlens_padded, "none", "add")
        else:
            grad.add_(grad_)


def update_grad_dkv(grad, grad_, step, is_half, cu_seqlens_padded):
    if step == 0:
        if is_half:
            tex.thd_grad_correction(grad, grad_, cu_seqlens_padded, "copy", "none")
        else:
            grad.copy_(grad_)
    else:
        if is_half:
            tex.thd_grad_correction(grad, grad_, cu_seqlens_padded, "add", "none")
        else:
            grad.add_(grad_)


# unpad -> fa varlen -> pad
def forward_varlen(
    q_inputs,
    k_inputs,
    v_inputs,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    max_seqlen_q,
    causal,
    fa_forward_kwargs,
):
    # unpad
    unpad_q_inputs, unpad_q_indices, max_seqlen_per_step_q = fa_varlen_thd_unpad(
        q_inputs,
        cu_seqlens_q_per_step,
        cu_seqlens_q_padded,
    )
    kv_inputs = torch.stack([k_inputs, v_inputs], dim=0)
    unpad_kv_inputs, unpad_kv_indices, max_seqlen_per_step_kv = fa_varlen_thd_unpad(
        kv_inputs,
        cu_seqlens_kv_per_step,
        cu_seqlens_kv_padded,
        packed=True,
    )
    fa_outputs = flash_attn_varlen_fwd(
        unpad_q_inputs,
        unpad_kv_inputs[0],
        unpad_kv_inputs[1],
        cu_seqlens_q_per_step,
        cu_seqlens_kv_per_step,
        max_seqlen_per_step_q,
        max_seqlen_per_step_kv,
        causal=causal,
        **fa_forward_kwargs,
    )
    out_per_step = fa_outputs[0]
    softmax_lse_per_step = fa_outputs[1]
    rng_states = fa_outputs[3]
    # pad
    out_per_step = fa_varlen_thd_pad(out_per_step, cu_seqlens_q_padded, unpad_q_indices)
    # softmax_lse_per_step = fa_varlen_lse_repad(softmax_lse_per_step, max_seqlen_q)
    softmax_lse_per_step = fa_varlen_lse_pad(
        softmax_lse_per_step, cu_seqlens_q_padded[-1], unpad_q_indices
    )

    return out_per_step, softmax_lse_per_step, rng_states


def _first_window_forward(
    q,
    k,
    v,
    cp_size,  # global p2p world size
    window_offset,  # global p2p rank offset
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    causal,
    local_p2p_comm,
    fa_forward_kwargs,
):
    local_rank = local_p2p_comm.rank
    local_cp_size = local_p2p_comm.world_size

    cu_seqlens_q_per_step = [None for _ in range(local_cp_size)]
    cu_seqlens_kv_per_step = [None for _ in range(local_cp_size)]
    rng_states = [None for _ in range(local_cp_size)]
    out_per_step = [None for _ in range(local_cp_size)]
    lse_per_step = [None for _ in range(local_cp_size)]

    # out = torch.zeros_like(q)
    softmax_lse = None
    for step in range(local_cp_size):
        if step + 1 != local_cp_size:
            next_k: torch.Tensor = local_p2p_comm.send_recv(k)
            next_v: torch.Tensor = local_p2p_comm.send_recv(v)
            local_p2p_comm.commit()

        flatten_p2p_rank_q = window_offset + local_rank
        flatten_p2p_rank_kv = window_offset + (local_rank - step) % local_cp_size
        if causal:
            if step == 0:  # q, k, v
                cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_q,
                    cu_seqlens_q_padded,
                    cp_size,
                    flatten_p2p_rank_q,
                    True,
                    True,
                )
                cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_kv,
                    cu_seqlens_kv_padded,
                    cp_size,
                    flatten_p2p_rank_kv,
                    True,
                    True,
                )
                block_out, block_lse, rng_state = forward_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens_q_per_step[step],
                    cu_seqlens_kv_per_step[step],
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    max_seqlen_q,
                    causal=True,
                    fa_forward_kwargs=fa_forward_kwargs,
                )
                rng_states[step] = rng_state
                out_per_step[step] = block_out
                lse_per_step[step] = block_lse
                softmax_lse = update_lse(
                    softmax_lse, block_lse, cu_seqlens_q_padded, False
                )
                # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, False)
            elif step <= local_rank:  # q, k0, v0
                cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_q,
                    cu_seqlens_q_padded,
                    cp_size,
                    flatten_p2p_rank_q,
                    True,
                    True,
                )
                cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_kv,
                    cu_seqlens_kv_padded,
                    cp_size,
                    flatten_p2p_rank_kv,
                    True,
                    False,
                )
                k0 = tex.thd_read_half_tensor(k, cu_seqlens_kv_padded, 0)
                v0 = tex.thd_read_half_tensor(v, cu_seqlens_kv_padded, 0)
                block_out, block_lse, rng_state = forward_varlen(
                    q,
                    k0,
                    v0,
                    cu_seqlens_q_per_step[step],
                    cu_seqlens_kv_per_step[step],
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded // 2,
                    max_seqlen_q,
                    causal=False,
                    fa_forward_kwargs=fa_forward_kwargs,
                )
                rng_states[step] = rng_state
                out_per_step[step] = block_out
                lse_per_step[step] = block_lse
                softmax_lse = update_lse(
                    softmax_lse, block_lse, cu_seqlens_q_padded, False
                )
                # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, False)
            else:  # q1, k, v
                cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_q,
                    cu_seqlens_q_padded,
                    cp_size,
                    flatten_p2p_rank_q,
                    False,
                    True,
                )
                cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_kv,
                    cu_seqlens_kv_padded,
                    cp_size,
                    flatten_p2p_rank_kv,
                    True,
                    True,
                )
                q1 = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                block_out, block_lse, rng_state = forward_varlen(
                    q1,
                    k,
                    v,
                    cu_seqlens_q_per_step[step],
                    cu_seqlens_kv_per_step[step],
                    cu_seqlens_q_padded // 2,
                    cu_seqlens_kv_padded,
                    max_seqlen_q // 2,
                    causal=False,
                    fa_forward_kwargs=fa_forward_kwargs,
                )
                rng_states[step] = rng_state
                out_per_step[step] = block_out
                lse_per_step[step] = block_lse
                softmax_lse = update_lse(
                    softmax_lse, block_lse, cu_seqlens_q_padded, True
                )
                # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, True)
        else:
            cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q,
                cu_seqlens_q_padded,
                cp_size,
                flatten_p2p_rank_q,
                True,
                True,
            )
            cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv,
                cu_seqlens_kv_padded,
                cp_size,
                flatten_p2p_rank_kv,
                True,
                True,
            )
            block_out, block_lse, rng_state = forward_varlen(
                q,
                k,
                v,
                cu_seqlens_q_per_step[step],
                cu_seqlens_kv_per_step[step],
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                max_seqlen_q,
                False,
                fa_forward_kwargs=fa_forward_kwargs,
            )
            rng_states[step] = rng_state
            out_per_step[step] = block_out
            lse_per_step[step] = block_lse
            # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, False)
            softmax_lse = update_lse(softmax_lse, block_lse, cu_seqlens_q_padded, False)

        if step + 1 != local_p2p_comm.world_size:
            local_p2p_comm.wait()
            k = next_k
            v = next_v

    # for step in range(local_cp_size):
    #     tex.thd_out_correction(
    #         out,
    #         out_per_step[step],
    #         softmax_lse,
    #         lse_per_step[step],
    #         cu_seqlens_q_padded,
    #         False,  # maybe not is_half
    #         False,  # softmax_lse_in_packed_format
    #     )
    return (
        softmax_lse,
        cu_seqlens_q_per_step,
        cu_seqlens_kv_per_step,
        rng_states,
        out_per_step,
        lse_per_step,
    )


def _other_window_forward(
    q,
    k,
    v,
    cp_size,  # global p2p world size
    window_offset,  # global p2p rank offset
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    causal,
    softmax_lse,
    window_num_idx,
    p2p_comm,
    local_p2p_comm,
    fa_forward_kwargs,
):
    local_rank = local_p2p_comm.rank
    local_cp_size = local_p2p_comm.world_size

    cu_seqlens_q_per_step = [None for _ in range(local_cp_size)]
    cu_seqlens_kv_per_step = [None for _ in range(local_cp_size)]
    rng_states = [None for _ in range(local_cp_size)]
    out_per_step = [None for _ in range(local_cp_size)]
    lse_per_step = [None for _ in range(local_cp_size)]

    for step in range(local_cp_size):
        flatten_p2p_rank_q = window_offset + local_rank
        flatten_p2p_rank_kv = window_offset + (local_rank - step) % local_cp_size

        if step + 1 != local_cp_size:
            next_k: torch.Tensor = local_p2p_comm.send_recv(k)
            next_v: torch.Tensor = local_p2p_comm.send_recv(v)
            local_p2p_comm.commit()

        if causal:
            if window_num_idx > p2p_comm.rank:  # q1, k ,v
                cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_q,
                    cu_seqlens_q_padded,
                    cp_size,
                    flatten_p2p_rank_q,
                    False,
                    True,
                )
                cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_kv,
                    cu_seqlens_kv_padded,
                    cp_size,
                    flatten_p2p_rank_kv,
                    True,
                    True,
                )
                q1 = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                block_out, block_lse, rng_state = forward_varlen(
                    q1,
                    k,
                    v,
                    cu_seqlens_q_per_step[step],
                    cu_seqlens_kv_per_step[step],
                    cu_seqlens_q_padded // 2,
                    cu_seqlens_kv_padded,
                    max_seqlen_q // 2,
                    causal=False,
                    fa_forward_kwargs=fa_forward_kwargs,
                )
                rng_states[step] = rng_state
                out_per_step[step] = block_out
                lse_per_step[step] = block_lse
                softmax_lse = update_lse(
                    softmax_lse, block_lse, cu_seqlens_q_padded, True
                )
                # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, True)
            else:  # q, k0, v0
                cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_q,
                    cu_seqlens_q_padded,
                    cp_size,
                    flatten_p2p_rank_q,
                    True,
                    True,
                )
                cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                    cu_seqlens_kv,
                    cu_seqlens_kv_padded,
                    cp_size,
                    flatten_p2p_rank_kv,
                    True,
                    False,
                )
                k0 = tex.thd_read_half_tensor(k, cu_seqlens_kv_padded, 0)
                v0 = tex.thd_read_half_tensor(v, cu_seqlens_kv_padded, 0)
                block_out, block_lse, rng_state = forward_varlen(
                    q,
                    k0,
                    v0,
                    cu_seqlens_q_per_step[step],
                    cu_seqlens_kv_per_step[step],
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded // 2,
                    max_seqlen_q,
                    causal=False,
                    fa_forward_kwargs=fa_forward_kwargs,
                )
                rng_states[step] = rng_state
                out_per_step[step] = block_out
                lse_per_step[step] = block_lse
                softmax_lse = update_lse(
                    softmax_lse, block_lse, cu_seqlens_q_padded, False
                )
                # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, False)
        else:
            cu_seqlens_q_per_step[step] = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q,
                cu_seqlens_q_padded,
                cp_size,
                flatten_p2p_rank_q,
                True,
                True,
            )
            cu_seqlens_kv_per_step[step] = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv,
                cu_seqlens_kv_padded,
                cp_size,
                flatten_p2p_rank_kv,
                True,
                True,
            )
            block_out, block_lse, rng_state = forward_varlen(
                q,
                k,
                v,
                cu_seqlens_q_per_step[step],
                cu_seqlens_kv_per_step[step],
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                max_seqlen_q,
                causal=False,
                fa_forward_kwargs=fa_forward_kwargs,
            )
            rng_states[step] = rng_state
            out_per_step[step] = block_out
            lse_per_step[step] = block_lse
            softmax_lse = update_lse(softmax_lse, block_lse, cu_seqlens_q_padded, False)
            # out, softmax_lse = update_out_and_lse(out, block_out, softmax_lse, block_lse, cu_seqlens_q_padded, False)

        if step + 1 != local_cp_size:
            local_p2p_comm.wait()
            k = next_k
            v = next_v

    return (
        softmax_lse,
        cu_seqlens_q_per_step,
        cu_seqlens_kv_per_step,
        rng_states,
        out_per_step,
        lse_per_step,
    )


def backward_varlen(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    causal,
    rng_state,
    fa_backward_kwargs,
):
    fa_backward_kwargs["rng_state"] = rng_state
    # fa_backward_kwargs["window_size"] = (-1, -1)
    # fa_backward_kwargs["window_size_left"] = -1
    # fa_backward_kwargs["window_size_right"] = -1
    unpad_q_, unpad_q_indices, max_seqlen_per_step_q = fa_varlen_thd_unpad(
        q, cu_seqlens_q_per_step, cu_seqlens_q_padded
    )
    kv = torch.stack([k, v], dim=0)
    unpad_kv_, unpad_kv_indices, max_seqlen_per_step_kv = fa_varlen_thd_unpad(
        kv,
        cu_seqlens_kv_per_step,
        cu_seqlens_kv_padded,
        packed=True,
    )
    dq_ = torch.zeros_like(unpad_q_)
    dkv_ = torch.empty_like(unpad_kv_)
    unpad_out_, _, _ = fa_varlen_thd_unpad(
        out,
        cu_seqlens_q_per_step,
        cu_seqlens_q_padded,
    )
    unpad_dout_, _, _ = fa_varlen_thd_unpad(
        dout,
        cu_seqlens_q_per_step,
        cu_seqlens_q_padded,
    )
    unpad_softmax_lse = fa_varlen_lse_unpad(softmax_lse, unpad_q_indices)
    # unpad_softmax_lse = fa_varlen_lse_repad(softmax_lse, max_seqlen_per_step_q)
    flash_attn_varlen_bwd(
        unpad_dout_,
        unpad_q_,
        unpad_kv_[0],
        unpad_kv_[1],
        unpad_out_,
        unpad_softmax_lse,
        dq_,
        dkv_[0],
        dkv_[1],
        cu_seqlens_q_per_step,
        cu_seqlens_kv_per_step,
        max_seqlen_per_step_q,
        max_seqlen_per_step_kv,
        causal=causal,
        **fa_backward_kwargs,
    )
    dq_ = fa_varlen_thd_pad(dq_, cu_seqlens_q_padded, unpad_q_indices)
    dkv_ = fa_varlen_thd_pad(dkv_, cu_seqlens_kv_padded, unpad_kv_indices, packed=True)

    return dq_, dkv_[0], dkv_[1]


def _first_window_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_lse_,
    cu_seqlens_q_inner_steps,
    cu_seqlens_kv_inner_steps,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    causal,
    rng_states,
    local_kv_comm,
    local_dkv_comm,
    fa_backward_kwargs,
):
    dk_comm_buffer, dv_comm_buffer = None, None
    # dq, dk, dv = None, None, None
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    local_cp_size = local_kv_comm.world_size
    local_cp_rank = local_kv_comm.rank

    for step in range(local_cp_size):
        if step + 1 != local_cp_size:
            next_k = local_kv_comm.send_recv(k)
            next_v = local_kv_comm.send_recv(v)
            local_kv_comm.commit()

        cu_seqlens_q_per_step = cu_seqlens_q_inner_steps[step]
        cu_seqlens_kv_per_step = cu_seqlens_kv_inner_steps[step]
        rng_state = rng_states[step]

        if causal:
            if step == 0:  # q, k, v
                dq_, dk_, dv_ = backward_varlen(
                    dout,
                    q,
                    k,
                    v,
                    out,
                    softmax_lse,
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    True,
                    rng_state,
                    fa_backward_kwargs,
                )
                update_grad_dq(
                    dq,
                    dq_.to(torch.float32),
                    step=0,
                    is_half=False,
                    cu_seqlens_padded=cu_seqlens_q_padded,
                )
                update_grad_dkv(
                    dk,
                    dk_.to(torch.float32),
                    step=0,
                    is_half=False,
                    cu_seqlens_padded=cu_seqlens_kv_padded,
                )
                update_grad_dkv(
                    dv,
                    dv_.to(torch.float32),
                    step=0,
                    is_half=False,
                    cu_seqlens_padded=cu_seqlens_kv_padded,
                )
            else:
                if step <= local_cp_rank:  # q, k0, v0
                    k0 = tex.thd_read_half_tensor(k, cu_seqlens_kv_padded, 0)
                    v0 = tex.thd_read_half_tensor(v, cu_seqlens_kv_padded, 0)
                    dq_, dk_, dv_ = backward_varlen(
                        dout,
                        q,
                        k0,
                        v0,
                        out,
                        softmax_lse,
                        cu_seqlens_q_per_step,
                        cu_seqlens_kv_per_step,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded // 2,
                        False,
                        rng_state,
                        fa_backward_kwargs,
                    )
                    update_grad_dq(
                        dq,
                        dq_.to(torch.float32),
                        step=step,
                        is_half=False,
                        cu_seqlens_padded=cu_seqlens_q_padded,
                    )
                else:  # q1, k, v
                    q1 = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                    out1 = tex.thd_read_half_tensor(out, cu_seqlens_q_padded, 1)
                    dout1 = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, 1)

                    dq_, dk_, dv_ = backward_varlen(
                        dout1,
                        q1,
                        k,
                        v,
                        out1,
                        softmax_lse_,
                        cu_seqlens_q_per_step,
                        cu_seqlens_kv_per_step,
                        cu_seqlens_q_padded // 2,
                        cu_seqlens_kv_padded,
                        False,
                        rng_state,
                        fa_backward_kwargs,
                    )
                    # always use the first half in dq_buffer.
                    update_grad_dq(
                        dq,
                        dq_,
                        step=step,
                        is_half=True,
                        cu_seqlens_padded=cu_seqlens_q_padded,
                    )

                local_dkv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv  # noqa: F821

                if step <= local_cp_rank:
                    update_grad_dkv(
                        dk,
                        dk_,
                        step=step,
                        is_half=True,
                        cu_seqlens_padded=cu_seqlens_kv_padded,
                    )
                    update_grad_dkv(
                        dv,
                        dv_,
                        step=step,
                        is_half=True,
                        cu_seqlens_padded=cu_seqlens_kv_padded,
                    )
                else:
                    update_grad_dkv(
                        dk,
                        dk_.to(torch.float32),
                        step=step,
                        is_half=False,
                        cu_seqlens_padded=cu_seqlens_kv_padded,
                    )
                    update_grad_dkv(
                        dv,
                        dv_.to(torch.float32),
                        step=step,
                        is_half=False,
                        cu_seqlens_padded=cu_seqlens_kv_padded,
                    )
        else:
            dq_, dk_, dv_ = backward_varlen(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q_per_step,
                cu_seqlens_kv_per_step,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                False,
                rng_state,
                fa_backward_kwargs,
            )
            update_grad_dq(
                dq,
                dq_.to(torch.float32),
                step=step,
                is_half=False,
                cu_seqlens_padded=cu_seqlens_q_padded,
            )

            if step > 0:
                local_dkv_comm.wait()
                dk_comm_buffer, dv_comm_buffer = dk, dv
                dk, dv = next_dk, next_dv  # noqa: F821

            update_grad_dkv(
                dk,
                dk_.to(torch.float32),
                step=step,
                is_half=False,
                cu_seqlens_padded=cu_seqlens_kv_padded,
            )
            update_grad_dkv(
                dv,
                dv_.to(torch.float32),
                step=step,
                is_half=False,
                cu_seqlens_padded=cu_seqlens_kv_padded,
            )

        if step + 1 != local_cp_size:
            local_kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
        local_dkv_comm.commit()

    local_dkv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


def _other_window_backward(
    dout,
    q,
    k,
    v,
    dq,
    dk,
    dv,
    out,
    softmax_lse,
    softmax_lse_,
    cu_seqlens_q_inner_steps,
    cu_seqlens_kv_inner_steps,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    causal,
    rng_states,
    window_num_idx,
    kv_comm,
    dkv_comm,
    local_kv_comm,
    local_dkv_comm,
    fa_backward_kwargs,
):
    dk_comm_buffer, dv_comm_buffer = None, None

    local_cp_size = local_kv_comm.world_size
    # local_cp_rank = local_kv_comm.rank

    for step in range(local_cp_size):
        if step + 1 != local_cp_size:
            next_k = local_kv_comm.send_recv(k)
            next_v = local_kv_comm.send_recv(v)
            local_kv_comm.commit()

        cu_seqlens_q_per_step = cu_seqlens_q_inner_steps[step]
        cu_seqlens_kv_per_step = cu_seqlens_kv_inner_steps[step]
        rng_state = rng_states[step]

        if causal:
            if window_num_idx > kv_comm.rank:  # q1, k, v
                q1 = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                out1 = tex.thd_read_half_tensor(out, cu_seqlens_q_padded, 1)
                dout1 = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, 1)
                dq_, dk_, dv_ = backward_varlen(
                    dout1,
                    q1,
                    k,
                    v,
                    out1,
                    softmax_lse_,
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    cu_seqlens_q_padded // 2,
                    cu_seqlens_kv_padded,
                    False,
                    rng_state,
                    fa_backward_kwargs,
                )
                # always use the first half in dq_buffer.
                update_grad_dq(
                    dq, dq_, step=1, is_half=True, cu_seqlens_padded=cu_seqlens_q_padded
                )
            else:  # q, k0, v0
                k0 = tex.thd_read_half_tensor(k, cu_seqlens_kv_padded, 0)
                v0 = tex.thd_read_half_tensor(v, cu_seqlens_kv_padded, 0)
                dq_, dk_, dv_ = backward_varlen(
                    dout,
                    q,
                    k0,
                    v0,
                    out,
                    softmax_lse,
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded // 2,
                    False,
                    rng_state,
                    fa_backward_kwargs,
                )
                update_grad_dq(
                    dq,
                    dq_.to(torch.float32),
                    step=1,
                    is_half=False,
                    cu_seqlens_padded=cu_seqlens_q_padded,
                )
        else:
            dq_, dk_, dv_ = backward_varlen(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q_per_step,
                cu_seqlens_kv_per_step,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                False,
                rng_state,
                fa_backward_kwargs,
            )
            update_grad_dq(
                dq,
                dq_.to(torch.float32),
                step=1,
                is_half=False,
                cu_seqlens_padded=cu_seqlens_q_padded,
            )

        if step > 0:
            local_dkv_comm.wait()
            dk_comm_buffer, dv_comm_buffer = dk, dv
            dk, dv = next_dk, next_dv  # noqa: F821

        if step == 0:
            dkv_comm.wait()

        # update dk,dv
        if window_num_idx > kv_comm.rank or not causal:  # q1, k, v
            update_grad_dkv(
                dk,
                dk_.to(torch.float32),
                step=1,
                is_half=False,
                cu_seqlens_padded=cu_seqlens_kv_padded,
            )
            update_grad_dkv(
                dv,
                dv_.to(torch.float32),
                step=1,
                is_half=False,
                cu_seqlens_padded=cu_seqlens_kv_padded,
            )
        else:  # q, k0, v0
            update_grad_dkv(
                dk, dk_, step=1, is_half=True, cu_seqlens_padded=cu_seqlens_kv_padded
            )
            update_grad_dkv(
                dv, dv_, step=1, is_half=True, cu_seqlens_padded=cu_seqlens_kv_padded
            )

        if step + 1 != local_cp_size:
            local_kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = local_dkv_comm.send_recv(dk, dk_comm_buffer)
        next_dv = local_dkv_comm.send_recv(dv, dv_comm_buffer)
        local_dkv_comm.commit()

    local_dkv_comm.wait()

    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class ZigZagDoubleRingAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        attn_mask_type,
        deterministic,
        cp_group,
        slide_window_size,
    ):
        context_group = cp_group[ParallelMode.CONTEXT]
        inter_window_group = cp_group[ParallelMode.INTER_WINDOW]
        intra_window_group = cp_group[ParallelMode.INTRA_WINDOW]
        dkv_inter_window_group = cp_group[ParallelMode.DKV_INTER_WINDOW]
        dkv_intra_window_group = cp_group[ParallelMode.DKV_INTRA_WINDOW]

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        causal = "causal" in attn_mask_type

        k = k.contiguous()
        v = v.contiguous()

        ring_comm = RingComm(context_group)
        p2p_comm = RingComm(inter_window_group)
        local_p2p_comm = RingComm(intra_window_group)

        softcap = 0.0
        window_size = (-1, -1)
        fa_forward_kwargs = {"softmax_scale": softmax_scale}
        fa_forward_kwargs["dropout_p"] = dropout_p
        fa_forward_kwargs["return_softmax"] = False
        # fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
        # fa_forward_kwargs["window_size"] = (-1, -1)
        fa_forward_kwargs["window_size_left"] = window_size[0]
        fa_forward_kwargs["window_size_right"] = window_size[1]
        fa_forward_kwargs["alibi_slopes"] = None
        fa_forward_kwargs["softcap"] = softcap

        cp_size = ring_comm.world_size

        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
        cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size
        qkv_dtype = q.dtype
        # q_f16 = q

        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        window_num = ring_comm.world_size // slide_window_size
        rng_states = [None for _ in range(cp_size)]
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]
        out_per_step = [None for _ in range(cp_size)]
        lse_per_step = [None for _ in range(cp_size)]

        local_k = k
        local_v = v

        out = torch.zeros_like(q)
        for j in range(window_num):
            if j > 0:
                p2p_comm.wait()
                local_k = next_k  # noqa: F821
                local_v = next_v  # noqa: F821

            if j + 1 != window_num:
                next_k: torch.Tensor = p2p_comm.send_recv(  # noqa: F841
                    local_k.contiguous()
                )  # noqa: F841
                next_v: torch.Tensor = p2p_comm.send_recv(  # noqa: F841
                    local_v.contiguous()
                )  # noqa: F841
                p2p_comm.commit()

            window_rank = p2p_comm.rank
            window_offset = ((window_rank - j) % window_num) * slide_window_size
            if j == 0:
                (
                    lse,
                    cu_seqlens_q_inner_steps,
                    cu_seqlens_kv_inner_steps,
                    rng_states_inner,
                    out_inner_steps,
                    lse_inner_steps,
                ) = _first_window_forward(
                    q,
                    local_k,
                    local_v,
                    cp_size,
                    window_offset,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    causal,
                    local_p2p_comm,
                    fa_forward_kwargs,
                )
            else:
                (
                    lse,
                    cu_seqlens_q_inner_steps,
                    cu_seqlens_kv_inner_steps,
                    rng_states_inner,
                    out_inner_steps,
                    lse_inner_steps,
                ) = _other_window_forward(
                    q,
                    local_k,
                    local_v,
                    cp_size,  # global p2p world size
                    window_offset,  # global p2p rank offset
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    causal,
                    lse,
                    j,
                    p2p_comm,
                    local_p2p_comm,
                    fa_forward_kwargs,
                )
            # save for backward
            cu_seqlens_q_per_step[
                window_offset : window_offset + slide_window_size
            ] = cu_seqlens_q_inner_steps
            cu_seqlens_kv_per_step[
                window_offset : window_offset + slide_window_size
            ] = cu_seqlens_kv_inner_steps
            rng_states[
                window_offset : window_offset + slide_window_size
            ] = rng_states_inner
            out_per_step[
                window_offset : window_offset + slide_window_size
            ] = out_inner_steps
            lse_per_step[
                window_offset : window_offset + slide_window_size
            ] = lse_inner_steps

        # update out
        softmax_lse = lse.to(torch.float)
        for step in range(cp_size):
            if lse_per_step[step].shape[-1] == cu_seqlens_q_padded[-1]:
                update_out(
                    out,
                    out_per_step[step],
                    softmax_lse,
                    lse_per_step[step],
                    cu_seqlens_q_padded,
                    False,
                )
            else:
                update_out(
                    out,
                    out_per_step[step],
                    softmax_lse,
                    lse_per_step[step],
                    cu_seqlens_q_padded,
                    True,
                )

        second_half_lse_seqlen = cu_seqlens_q_padded[-1] // 2

        # lse = lse.squeeze(dim=-1).transpose(1, 2)
        out = out.to(qkv_dtype)

        out_f16 = out.to(qkv_dtype)
        out_ret = out_f16
        out_save = out_f16

        ctx.save_for_backward(
            q,
            k,
            v,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.slide_window_size = slide_window_size
        ctx.alibi_slopes = None
        ctx.deterministic = deterministic
        ctx.attn_mask_type = attn_mask_type
        # ctx.cp_group = cp_group
        ctx.context_group = context_group
        ctx.inter_window_group = inter_window_group
        ctx.intra_window_group = intra_window_group
        ctx.dkv_inter_window_group = dkv_inter_window_group
        ctx.dkv_intra_window_group = dkv_intra_window_group

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        # TODO read ctx data
        (*saved_tensors,) = ctx.saved_tensors
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        ) = saved_tensors[:7]

        # cp_group = ctx.cp_group
        slide_window_size = ctx.slide_window_size

        context_group = ctx.context_group
        inter_window_group = ctx.inter_window_group
        intra_window_group = ctx.intra_window_group
        dkv_inter_window_group = ctx.dkv_inter_window_group
        dkv_intra_window_group = ctx.dkv_intra_window_group

        context_comm = RingComm(context_group)
        kv_comm = RingComm(inter_window_group)
        local_kv_comm = RingComm(intra_window_group)
        dkv_comm = RingComm(dkv_inter_window_group)
        local_dkv_comm = RingComm(dkv_intra_window_group)

        cp_size = context_comm.world_size

        cu_seqlens_q_per_steps = saved_tensors[7 : 7 + cp_size]
        cu_seqlens_kv_per_steps = saved_tensors[7 + cp_size : 7 + cp_size * 2]
        rng_states = saved_tensors[7 + cp_size * 2 : 7 + cp_size * 3]

        causal = "causal" in ctx.attn_mask_type
        # softmax_lse_ = tex.thd_read_second_half_lse(
        #     softmax_lse, cu_seqlens_q_padded, False
        # )

        # if causal and ctx.second_half_lse_seqlen is not None:
        softmax_lse_ = tex.thd_read_second_half_lse(
            softmax_lse,
            cu_seqlens_q_padded,
            softmax_lse_in_packed_format,
            ctx.second_half_lse_seqlen,
        )

        fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
        fa_backward_kwargs["dropout_p"] = ctx.dropout_p
        fa_backward_kwargs["alibi_slopes"] = None
        fa_backward_kwargs["deterministic"] = ctx.deterministic
        fa_backward_kwargs["window_size_left"] = ctx.window_size[0]
        fa_backward_kwargs["window_size_right"] = ctx.window_size[1]
        fa_backward_kwargs["softcap"] = ctx.softcap

        window_num = context_comm.world_size // slide_window_size

        local_k = k
        local_v = v

        window_rank = kv_comm.rank
        for j in range(window_num):
            if j > 0:
                kv_comm.wait()
                local_k = next_k  # noqa: F821
                local_v = next_v  # noqa: F821

            if j + 1 != window_num:
                next_k: torch.Tensor = kv_comm.send_recv(  # noqa: F841
                    local_k.contiguous()
                )  # noqa: F841
                next_v: torch.Tensor = kv_comm.send_recv(  # noqa: F841
                    local_v.contiguous()
                )  # noqa: F841
                kv_comm.commit()

            if j > 0:
                # dkv_comm.wait()
                dk = next_dk  # noqa: F821
                dv = next_dv  # noqa: F821

            window_offset = ((window_rank - j) % window_num) * slide_window_size
            # TODO
            cu_seqlens_q_inner_steps = cu_seqlens_q_per_steps[
                window_offset : window_offset + slide_window_size
            ]
            cu_seqlens_kv_inner_steps = cu_seqlens_kv_per_steps[
                window_offset : window_offset + slide_window_size
            ]
            rng_state_inner_steps = rng_states[
                window_offset : window_offset + slide_window_size
            ]

            if j == 0:
                dq, dk, dv = _first_window_backward(
                    dout,
                    q,
                    local_k,
                    local_v,
                    out,
                    softmax_lse,
                    softmax_lse_,
                    cu_seqlens_q_inner_steps,
                    cu_seqlens_kv_inner_steps,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    causal,
                    rng_state_inner_steps,
                    local_kv_comm,
                    local_dkv_comm,
                    fa_backward_kwargs,
                )
            else:
                dq, dk, dv = _other_window_backward(
                    dout,
                    q,
                    local_k,
                    local_v,
                    dq,
                    dk,
                    dv,
                    out,
                    softmax_lse,
                    softmax_lse_,
                    cu_seqlens_q_inner_steps,
                    cu_seqlens_kv_inner_steps,
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                    causal,
                    rng_state_inner_steps,
                    j,
                    kv_comm,
                    dkv_comm,
                    local_kv_comm,
                    local_dkv_comm,
                    fa_backward_kwargs,
                )

            next_dk: torch.Tensor = dkv_comm.send_recv(dk.contiguous())
            next_dv: torch.Tensor = dkv_comm.send_recv(dv.contiguous())
            dkv_comm.commit()

        dkv_comm.wait()

        dq = dq.to(q.dtype)
        next_dk = next_dk.to(q.dtype)
        next_dv = next_dv.to(q.dtype)

        return (
            dq,
            next_dk,
            next_dv,
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
            None,
        )


class LoongTrain(AttnBaselineInterface):
    def __init__(self):
        super().__init__()
        self.packed_seq_params = {}

    def dispatch(
        self,
        x_global: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        cp_group: dist.ProcessGroup,
        **kwargs,
    ) -> torch.Tensor:
        print(f"Extra kwargs received: {kwargs}")
        ranges = kwargs.get("ranges", None)
        attention_mask_thd = kwargs.get("attention_mask_thd", None)
        qkv_ = kwargs.get("qkv_", "q")

        cp_group_a2a = cp_group[ParallelMode.HEAD]
        cp_group_p2p = cp_group[ParallelMode.CONTEXT]

        cp_size_a2a = get_distributed_world_size(cp_group_a2a)
        rank_a2a = get_distributed_rank(cp_group_a2a)
        cp_size_p2p = get_distributed_world_size(cp_group_p2p)
        rank_p2p = get_distributed_rank(cp_group_p2p)

        assert (
            cp_size_a2a * cp_size_p2p == cp_size
        ), "Current two-level CP groups need cp_size_a2a*cp_size_p2p == cp_size!"

        # 2*cp*up
        self.padding_factor_p2p = 2 * cp_size_p2p
        self.padding_factor_a2a = 2 * cp_size

        total_seqlen = attention_mask_thd.sum(dim=0, dtype=torch.int32).item()
        cu_seqlens = torch.tensor(
            ranges.to_cu_seqlens(seq_len=total_seqlen),
            device=x_global.device,
            dtype=torch.int32,
        )
        assert total_seqlen == cu_seqlens[-1], "total_seqlen != cu_seqlens[-1]"

        (
            indices,
            indices_in_thd_padded,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        ) = _get_zigzag2d_varlen_unpad_data(
            attention_mask_thd=attention_mask_thd,
            cu_seqlens=cu_seqlens,
            padding_factor_p2p=self.padding_factor_p2p,
            padding_factor_a2a=self.padding_factor_a2a,
        )
        self.packed_seq_params[qkv_] = PackedSeqParams(
            indices,
            indices_in_thd_padded,
            cu_seqlens,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        )

        x_local = zigzag_loongtrain_dispatch(
            x_global,
            self.packed_seq_params[qkv_],
            cp_size_a2a,
            rank_a2a,
            cp_size_p2p,
            rank_p2p,
        )

        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,  # t,h,d
        cp_rank: int,
        cp_size: int,
        cp_group: Optional[Union[dist.ProcessGroup, Dict[Any, dist.ProcessGroup]]],
        **kwargs,
    ) -> torch.Tensor:
        print(f"Extra kwargs received: {kwargs}")
        qkv_ = kwargs.get("qkv_", "q")

        x_global = zigzag_loongtrain_undispatch(
            x_local, self.packed_seq_params[qkv_], cp_size, cp_group
        )
        return x_global

    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: AttnMaskType | list[AttnMaskType],
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        deterministic: bool,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        print(f"Extra kwargs received: {kwargs}")
        cp_group = kwargs.get("cp_group", None)
        dropout_p = kwargs.get("dropout_p", 0.0)
        slide_window_size = kwargs.get("slide_window_size", 1)

        cp_group_a2a = cp_group[ParallelMode.HEAD]

        q = _SeqAllToAll.apply(cp_group_a2a, 1, 0, q)
        k = _SeqAllToAll.apply(cp_group_a2a, 1, 0, k)
        v = _SeqAllToAll.apply(cp_group_a2a, 1, 0, v)

        assert isinstance(
            attn_mask_type, AttnMaskType
        ), "attn_mask_type must be an AttnMaskType!"
        if attn_mask_type == AttnMaskType.CAUSAL:
            loongtrain_attn_mask = "padding_causal"
        else:
            loongtrain_attn_mask = "padding"

        # context = self.local_attn(q, k, v, *args, **kwargs)
        context, softmax_lse = ZigZagDoubleRingAttnVarlenFunc.apply(
            q,
            k,
            v,
            self.packed_seq_params["q"].cu_seqlens,
            self.packed_seq_params["k"].cu_seqlens,
            self.packed_seq_params["q"].max_seqlen_in_padded,
            self.packed_seq_params["k"].max_seqlen_in_padded,
            self.packed_seq_params["q"].cu_seqlens_padded,
            self.packed_seq_params["k"].cu_seqlens_padded,
            dropout_p,
            softmax_scale,
            loongtrain_attn_mask,
            deterministic,
            cp_group,
            slide_window_size,
        )

        context = _SeqAllToAll.apply(cp_group_a2a, 0, 1, context)

        return context, softmax_lse
