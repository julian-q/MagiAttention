import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_pkg_version
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import transformer_engine  # noqa
import transformer_engine_torch as tex
from einops import rearrange, repeat
from packaging.version import Version as PkgVersion
from transformer_engine.pytorch.attention import (  # flash_attn_fwd_softmax_lse_correction,
    _get_supported_versions,
    fa_logger,
    flash_attn_p2p_communicate,
    get_cu_seqlens_on_cp_rank,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# full causal
from dffa.common.enum import AttnMaskType
from dffa.common.ranges import AttnRanges

from .interface import AttnBaselineInterface
from .teusp_utils import (
    SeqAllToAll3D,
    _get_chunk_indices_on_cp_rank,
    _get_varlen_unpad_data,
    fa_varlen_lse_pad,
    fa_varlen_lse_unpad,
    fa_varlen_thd_pad,
    fa_varlen_thd_unpad,
    flash_attn_fwd_softmax_lse_correction,
    get_distributed_rank,
    get_distributed_world_size,
)

# from transformer_engine.pytorch.distributed import (
#     get_distributed_world_size,
#     get_distributed_rank
# )


# def flash_attn_fwd_softmax_lse_correction(
#     softmax_lse: torch.Tensor,
#     softmax_lse_per_step: torch.Tensor,
# ):
#     """Merge softmax stats of each step in Attention with context parallelism"""
#     max_scale = torch.max(softmax_lse, softmax_lse_per_step)
#     min_scale = torch.min(softmax_lse, softmax_lse_per_step)
#     new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
#     softmax_lse.copy_(new_scale)


# TODO
_NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))
_NVTE_FUSED_ATTN = int(os.getenv("NVTE_FUSED_ATTN", "1"))
_NVTE_UNFUSED_ATTN = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))

# Detect flash-attn v2 in the environment
_flash_attn_version = PkgVersion("0")
_flash_attn_version_required = PkgVersion("2.1.1")
_flash_attn_max_version = PkgVersion("2.7.6")
_flash_attn_2_3_plus = False
_flash_attn_2_4_plus = False
_flash_attn_2_4_1_plus = False
_flash_attn_2_5_7_plus = False
_flash_attn_2_6_0_plus = False
_flash_attn_2_7_0_plus = False

# flash_attn_varlen_fwd = None
# flash_attn_varlen_bwd = None

try:
    _flash_attn_version = PkgVersion(get_pkg_version("flash-attn"))
except PackageNotFoundError:
    if torch.cuda.is_available() and _NVTE_FLASH_ATTN:
        fa_logger.debug(
            "flash-attn v2 is not installed. To use, please install it by"
            """ "pip install flash-attn".""",
        )
else:
    if _flash_attn_version_required <= _flash_attn_version <= _flash_attn_max_version:
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_backward as flash_attn_varlen_bwd,
        )
        from flash_attn.flash_attn_interface import (
            _flash_attn_varlen_forward as flash_attn_varlen_fwd,
        )

        _flash_attn_2_3_plus = _flash_attn_version >= PkgVersion("2.3")
        _flash_attn_2_4_plus = _flash_attn_version >= PkgVersion("2.4")
        _flash_attn_2_4_1_plus = _flash_attn_version >= PkgVersion("2.4.1")
        _flash_attn_2_5_7_plus = _flash_attn_version >= PkgVersion("2.5.7")
        _flash_attn_2_6_0_plus = _flash_attn_version >= PkgVersion("2.6.0")
        _flash_attn_2_7_0_plus = _flash_attn_version >= PkgVersion("2.7.0")
    elif torch.cuda.is_available() and _NVTE_FLASH_ATTN:
        fa_logger.warning(
            "Supported flash-attn versions are %s. Found flash-attn %s.",
            _get_supported_versions(
                _flash_attn_version_required,
                _flash_attn_max_version,
            ),
            _flash_attn_version,
        )

TE_DType = {
    torch.uint8: tex.DType.kByte,
    torch.int32: tex.DType.kInt32,
    torch.float32: tex.DType.kFloat32,
    torch.half: tex.DType.kFloat16,
    torch.bfloat16: tex.DType.kBFloat16,
}


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


def basic_dispatch(x_global, rank, world_size, cp_group, dim=0, *args, **kwargs):
    return x_global.chunk(world_size, dim=dim)[rank].detach().clone()


# cp_group[0] ulysess
# cp_group[1] ring
def zigzag_usp_dispatch(x_global, rank, world_size, cp_group, dim=0, *args, **kwargs):
    """
    x_global is a tensor of shape (s, b, ...)
    """
    input_dim = x_global.dim()
    assert input_dim >= 2
    seqlen, batch_size, *rest = x_global.shape

    ud = dist.get_world_size(group=cp_group[0])
    rd = dist.get_world_size(group=cp_group[1])

    assert (
        ud * rd == world_size
    ), "Current two-level CP groups need ud*rd == world_size!"

    x_chunks = x_global.chunk(2 * rd, dim=dim)
    u_rank = dist.get_rank(group=cp_group[0])
    r_rank = dist.get_rank(group=cp_group[1])

    x_local = torch.cat(
        [x_chunks[r_rank], x_chunks[2 * rd - r_rank - 1]], dim=dim
    ).chunk(ud, dim=dim)[u_rank]

    new_shape = [seqlen // world_size, batch_size] + rest
    return x_local.reshape(new_shape).contiguous()


def zigzag_teusp_dispatch(
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


def basic_undispatch(x_local, rank, world_size, cp_group, dim=0, *args, **kwargs):
    chunk_shape = list(x_local.shape)
    chunk_shape[0] *= world_size
    x_global = torch.empty(chunk_shape, dtype=x_local.dtype, device=x_local.device)
    chunks = list(torch.chunk(x_global, world_size, dim=dim))
    dist.all_gather(chunks, x_local, group=cp_group)
    return torch.cat(chunks, dim=dim)


def zigzag_usp_undispatch(x_local, rank, world_size, cp_group, dim=0, *args, **kwargs):
    input_dim = x_local.ndim()
    assert input_dim >= 2
    seqlen, batch_size, *rest = x_local.shape

    # Get the sizes of the two-level CP groups
    ud = dist.get_world_size(group=cp_group[0])
    rd = dist.get_world_size(group=cp_group[1])

    assert (
        ud * rd == world_size
    ), "Current two-level CP groups need ud*rd == world_size!"

    # Get the ranks for the current process in the two-level CP groups
    # u_rank = dist.get_rank(group=cp_group[0])
    r_rank = dist.get_rank(group=cp_group[1])

    # ulysess all gather
    local_u_group = [torch.empty_like(x_local) for _ in range(ud)]
    dist.all_gather(local_u_group, x_local, group=cp_group[0])
    x_local_ring = torch.cat(local_u_group, dim=dim)

    # ring all gather
    local_r_group = [torch.empty_like(x_local_ring) for _ in range(rd)]
    dist.all_gather(local_r_group, x_local_ring, group=cp_group[1])

    split_chunks = []
    for chunk in local_r_group:
        split_chunks.extend(chunk.chunk(2, dim=dim))

    reassembled_chunks = []
    for r_rank in range(rd):
        reassembled_chunks.append(split_chunks[r_rank])
        reassembled_chunks.append(split_chunks[2 * rd - r_rank - 1])

    x_global = torch.cat(reassembled_chunks, dim=dim)

    new_shape = [seqlen * world_size, batch_size] + rest
    return x_global.reshape(new_shape).contiguous()


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


# thd
def zigzag_teusp_undispatch(
    x_local,
    packed_seq_params,
    world_size,
    cp_group,
    use_ulysess_low=True,
    *args,
    **kwargs,
):
    input_dim = x_local.ndim
    assert input_dim >= 2
    total_seqlen, *other_shape = x_local.shape

    # Get the sizes of the two-level CP groups
    if use_ulysess_low:
        cp_group_a2a = cp_group[0]
        cp_group_p2p = cp_group[1]
    else:
        cp_group_a2a = cp_group[1]
        cp_group_p2p = cp_group[0]

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


# load balanced 的 te usp varlen 实现
class ZigZagTEUSPAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        is_training,
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
        use_fused_attention,
        cp_group,
        cp_global_ranks,
        cp_stream,
        use_ulysses_low=True,
    ):
        attn_bias_type = "no_bias"
        attn_bias = None
        qkv_format = "thd"
        # pylint: disable=missing-function-docstring
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert isinstance(
            cp_group, list
        ), "Hierarchical CP implementation needs multi-level CP groups!"

        if use_ulysses_low:
            cp_group_a2a = cp_group[0]
            cp_size_a2a = get_distributed_world_size(cp_group_a2a)
            rank_a2a = get_distributed_rank(cp_group_a2a)
            cp_group = cp_group[1]
        else:
            cp_group_a2a = cp_group[1]
            cp_size_a2a = get_distributed_world_size(cp_group_a2a)
            rank_a2a = get_distributed_rank(cp_group_a2a)
            cp_group = cp_group[0]

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        if use_ulysses_low:
            send_dst = cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
            recv_src = cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        else:  # TODO
            send_dst = cp_global_ranks[(rank + 1) % cp_size + rank_a2a * cp_size_a2a]
            recv_src = cp_global_ranks[(rank - 1) % cp_size + rank_a2a * cp_size_a2a]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            cp_size == 2
        )

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        seq_dim = 0
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        pad_between_seqs_q = not torch.equal(cu_seqlens_q_padded, cu_seqlens_q)
        pad_between_seqs_kv = not torch.equal(cu_seqlens_kv_padded, cu_seqlens_kv)
        # TODO ?
        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
        cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        # TODO
        fused_attn_qkv_dtype = None
        fused_attn_backend = None
        qkv_dtype = q.dtype

        q_f16 = q
        if use_fused_attention:
            fp8_meta_kwargs = {}
            fused_attn_qkv_dtype = TE_DType[q.dtype]
            fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        assert (
            qkv_format == "thd"
            and q.shape[seq_dim] % 2 == 0
            and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        total_tokens_kv = k.shape[0]
        # remove padded tokens at the end
        k, v = [x[: cu_seqlens_kv_padded[-1]] for x in [k, v]]

        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        # TODO
        softmax_lse_in_packed_format = (
            not use_fused_attention and _flash_attn_2_6_0_plus
        )
        flash_attn_fwd = None
        softcap = 0.0
        window_size = (-1, -1)
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            flash_attn_fwd = flash_attn_varlen_fwd
            fa_forward_kwargs["dropout_p"] = dropout_p
            fa_forward_kwargs["return_softmax"] = False
            # if _flash_attn_2_3_plus:
            #     fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)
            if _flash_attn_2_3_plus:
                if _flash_attn_2_7_0_plus:
                    fa_forward_kwargs["window_size_left"] = window_size[0]
                    fa_forward_kwargs["window_size_right"] = window_size[1]
                    fa_forward_kwargs["softcap"] = softcap
                else:
                    fa_forward_kwargs["window_size"] = window_size
            if _flash_attn_2_4_plus:
                fa_forward_kwargs["alibi_slopes"] = None
            if _flash_attn_2_5_7_plus:
                fa_forward_kwargs["block_table"] = None

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        # lse_indices_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        # useless
        # attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        # kv packed
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        # softmax_lse_ = None
        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    # 最后一个step不用
                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        # i 当前计算并发送
                        # i+1 下一次计算并接收
                        send_recv_reqs[i % 2] = flash_attn_p2p_communicate(
                            rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )

                    kv_inputs[i % 2] = p2p_comm_buffers[i]

                    if causal:
                        if i == 0:  # q k v
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q,
                                    cu_seqlens_q_padded,
                                    cp_size,
                                    rank,
                                    True,
                                    True,
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    rank,
                                    True,
                                    True,
                                )
                            else:
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size

                            if use_fused_attention:
                                q_inputs[i % 2] = q
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    fused_attn_qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type=attn_mask_type,
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias,
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                (
                                    softmax_lse_per_step[i],
                                    rng_states[i],
                                    *rest,
                                ) = aux_ctx_tensors
                            else:
                                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    2, -1, *k.shape[-2:]
                                )
                                # unpad
                                (
                                    unpad_q_inputs,
                                    unpad_q_indices,
                                    max_seqlen_per_step_q,
                                ) = fa_varlen_thd_unpad(
                                    q_inputs[i % 2],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_q_padded,
                                )
                                (
                                    unpad_kv_inputs,
                                    unpad_kv_indices,
                                    max_seqlen_per_step_kv,
                                ) = fa_varlen_thd_unpad(
                                    kv_inputs[i % 2],
                                    cu_seqlens_kv_per_step[i],
                                    cu_seqlens_kv_padded,
                                    packed=True,
                                )
                                fa_outputs = flash_attn_fwd(
                                    unpad_q_inputs,
                                    unpad_kv_inputs[0],
                                    unpad_kv_inputs[1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_per_step_q,
                                    max_seqlen_per_step_kv,
                                    causal=True,
                                    **fa_forward_kwargs,
                                )
                                out_per_step[i] = fa_outputs[0]
                                softmax_lse_per_step[i] = fa_outputs[1]
                                rng_states[i] = fa_outputs[3]
                                # pad
                                out_per_step[i] = fa_varlen_thd_pad(
                                    out_per_step[i],
                                    cu_seqlens_q_padded,
                                    unpad_q_indices,
                                )
                                # softmax_lse_per_step[i] = fa_varlen_lse_repad(
                                #     softmax_lse_per_step[i], max_seqlen_q
                                # )
                                softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                    softmax_lse_per_step[i],
                                    cu_seqlens_q_padded[-1],
                                    unpad_q_indices,
                                )
                                # softmax_lse_per_step[i], lse_indices_per_step[i] = fa_varlen_lse_pad2
                                # (softmax_lse_per_step[i], cu_seqlens_q_per_step[i], cu_seqlens_q_padded)
                        elif i <= rank:  # q k0 v0
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q,
                                    cu_seqlens_q_padded,
                                    cp_size,
                                    rank,
                                    True,
                                    True,
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    False,
                                )
                            else:
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // (
                                    cp_size * 2
                                )

                            if use_fused_attention:
                                q_inputs[i % 2] = q
                                # [2, t, np, hn] -> [2, t/2, np, hn]
                                kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                    kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                                )
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q,
                                    max_seqlen_kv // 2,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    fused_attn_qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias,
                                    cu_seqlens_q_padded=cu_seqlens_q_padded,
                                    cu_seqlens_kv_padded=(
                                        None
                                        if cu_seqlens_kv_padded is None
                                        else cu_seqlens_kv_padded // 2
                                    ),
                                    **fp8_meta_kwargs,
                                )
                                (
                                    softmax_lse_per_step[i],
                                    rng_states[i],
                                    *rest,
                                ) = aux_ctx_tensors
                            else:
                                # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                                q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                                # [2, t, np, hn] -> [2, t/2, np, hn]
                                kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                    kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                                )
                                # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                                # kv_inputs[i % 2] = kv_inputs[i % 2].view(2, -1, *k.shape[-2:])
                                # if _flash_attn_2_3_plus:
                                #     fa_forward_kwargs["window_size"] = (-1, -1)
                                # unpad
                                (
                                    unpad_q_inputs,
                                    unpad_q_indices,
                                    max_seqlen_per_step_q,
                                ) = fa_varlen_thd_unpad(
                                    q_inputs[i % 2],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_q_padded,
                                )
                                (
                                    unpad_kv_inputs,
                                    unpad_kv_indices,
                                    max_seqlen_per_step_kv,
                                ) = fa_varlen_thd_unpad(
                                    kv_inputs[i % 2],
                                    cu_seqlens_kv_per_step[i],
                                    cu_seqlens_kv_padded // 2,
                                    packed=True,
                                )
                                fa_outputs = flash_attn_fwd(
                                    unpad_q_inputs,
                                    unpad_kv_inputs[0],
                                    unpad_kv_inputs[1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_per_step_q,
                                    max_seqlen_per_step_kv,
                                    causal=False,
                                    **fa_forward_kwargs,
                                )
                                out_per_step[i] = fa_outputs[0]
                                softmax_lse_per_step[i] = fa_outputs[1]
                                rng_states[i] = fa_outputs[3]
                                # pad
                                out_per_step[i] = fa_varlen_thd_pad(
                                    out_per_step[i],
                                    cu_seqlens_q_padded,
                                    unpad_q_indices,
                                )
                                # softmax_lse_per_step[i] = fa_varlen_lse_repad(
                                #     softmax_lse_per_step[i], max_seqlen_q
                                # )
                                softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                    softmax_lse_per_step[i],
                                    cu_seqlens_q_padded[-1],
                                    unpad_q_indices,
                                )
                                # softmax_lse_per_step[i], lse_indices_per_step[i] = fa_varlen_lse_pad2
                                # (softmax_lse_per_step[i], cu_seqlens_q_per_step[i], cu_seqlens_q_padded)
                        else:  # q1 k v
                            if pad_between_seqs_q:
                                cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_q,
                                    cu_seqlens_q_padded,
                                    cp_size,
                                    rank,
                                    False,
                                    True,
                                )
                            else:
                                cu_seqlens_q_per_step[i] = cu_seqlens_q // (cp_size * 2)
                            if pad_between_seqs_kv:
                                cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                    cu_seqlens_kv,
                                    cu_seqlens_kv_padded,
                                    cp_size,
                                    (rank - i) % cp_size,
                                    True,
                                    True,
                                )
                            else:
                                cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size

                            if use_fused_attention:
                                # [t, np, hn] -> [t/2, np, hn]
                                q_inputs[i % 2] = tex.thd_read_half_tensor(
                                    q, cu_seqlens_q_padded, 1
                                )
                                out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                    is_training,
                                    max_seqlen_q // 2,
                                    max_seqlen_kv,
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    fused_attn_qkv_dtype,
                                    fused_attn_backend,
                                    attn_scale=softmax_scale,
                                    dropout=dropout_p,
                                    qkv_layout=qkv_layout,
                                    attn_mask_type="padding" if padding else "no_mask",
                                    attn_bias_type=attn_bias_type,
                                    attn_bias=attn_bias,
                                    cu_seqlens_q_padded=(
                                        None
                                        if cu_seqlens_q_padded is None
                                        else cu_seqlens_q_padded // 2
                                    ),
                                    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                    **fp8_meta_kwargs,
                                )
                                (
                                    softmax_lse_per_step[i],
                                    rng_states[i],
                                    *rest,
                                ) = aux_ctx_tensors
                            else:
                                # [t, np, hn] -> [t/2, np, hn]
                                q_inputs[i % 2] = tex.thd_read_half_tensor(
                                    q, cu_seqlens_q_padded, 1
                                )
                                # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    2, -1, *k.shape[-2:]
                                )
                                # if _flash_attn_2_3_plus:
                                #     fa_forward_kwargs["window_size"] = (-1, -1)
                                # unpad
                                (
                                    unpad_q_inputs,
                                    unpad_q_indices,
                                    max_seqlen_per_step_q,
                                ) = fa_varlen_thd_unpad(
                                    q_inputs[i % 2],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_q_padded // 2,
                                )
                                (
                                    unpad_kv_inputs,
                                    unpad_kv_indices,
                                    max_seqlen_per_step_kv,
                                ) = fa_varlen_thd_unpad(
                                    kv_inputs[i % 2],
                                    cu_seqlens_kv_per_step[i],
                                    cu_seqlens_kv_padded,
                                    packed=True,
                                )
                                fa_outputs = flash_attn_fwd(
                                    unpad_q_inputs,
                                    unpad_kv_inputs[0],
                                    unpad_kv_inputs[1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_per_step_q,
                                    max_seqlen_per_step_kv,
                                    causal=False,
                                    **fa_forward_kwargs,
                                )
                                out_per_step[i] = fa_outputs[0]
                                softmax_lse_per_step[i] = fa_outputs[1]
                                rng_states[i] = fa_outputs[3]
                                # pad
                                out_per_step[i] = fa_varlen_thd_pad(
                                    out_per_step[i],
                                    cu_seqlens_q_padded // 2,
                                    unpad_q_indices,
                                )
                                # softmax_lse_per_step[i] = fa_varlen_lse_repad(
                                #     softmax_lse_per_step[i], max_seqlen_q // 2
                                # )
                                # softmax_lse_per_step[i], lse_indices_per_step[i] = fa_varlen_lse_pad2
                                # (softmax_lse_per_step[i], cu_seqlens_q_per_step[i], cu_seqlens_q_padded // 2)
                                softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                    softmax_lse_per_step[i],
                                    cu_seqlens_q_padded[-1] // 2,
                                    unpad_q_indices,
                                )
                    else:
                        if pad_between_seqs_q:
                            cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_q,
                                cu_seqlens_q_padded,
                                cp_size,
                                rank,
                                True,
                                True,
                            )
                        else:
                            cu_seqlens_q_per_step[i] = cu_seqlens_q // cp_size
                        if pad_between_seqs_kv:
                            cu_seqlens_kv_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_kv,
                                cu_seqlens_kv_padded,
                                cp_size,
                                (rank - i) % cp_size,
                                True,
                                True,
                            )
                        else:
                            cu_seqlens_kv_per_step[i] = cu_seqlens_kv // cp_size

                        if use_fused_attention:
                            out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                                is_training,
                                max_seqlen_q,
                                max_seqlen_kv,
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                q,
                                kv_inputs[i % 2][0],
                                kv_inputs[i % 2][1],
                                fused_attn_qkv_dtype,
                                fused_attn_backend,
                                attn_scale=softmax_scale,
                                dropout=dropout_p,
                                qkv_layout=qkv_layout,
                                attn_mask_type=attn_mask_type,
                                attn_bias_type=attn_bias_type,
                                attn_bias=attn_bias,
                                cu_seqlens_q_padded=cu_seqlens_q_padded,
                                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                                **fp8_meta_kwargs,
                            )
                            (
                                softmax_lse_per_step[i],
                                rng_states[i],
                                *rest,
                            ) = aux_ctx_tensors
                        else:
                            # t,h,d
                            # [b, sq, np, hn] -> [b*sq, np, hn]
                            q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                            # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                            kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                2, -1, *k.shape[-2:]
                            )
                            # unpad
                            # ctx.indices
                            (
                                unpad_q_inputs,
                                unpad_q_indices,
                                max_seqlen_per_step_q,
                            ) = fa_varlen_thd_unpad(
                                q_inputs[i % 2],
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_q_padded,
                            )
                            (
                                unpad_kv_inputs,
                                unpad_kv_indices,
                                max_seqlen_per_step_kv,
                            ) = fa_varlen_thd_unpad(
                                kv_inputs[i % 2],
                                cu_seqlens_kv_per_step[i],
                                cu_seqlens_kv_padded,
                                packed=True,
                            )
                            fa_outputs = flash_attn_fwd(
                                unpad_q_inputs,
                                unpad_kv_inputs[0],  # k
                                unpad_kv_inputs[1],  # v
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                max_seqlen_per_step_q,
                                max_seqlen_per_step_kv,
                                causal=False,
                                **fa_forward_kwargs,
                            )
                            out_per_step[i] = fa_outputs[0]
                            softmax_lse_per_step[i] = fa_outputs[1]
                            rng_states[i] = fa_outputs[3]
                            # pad
                            out_per_step[i] = fa_varlen_thd_pad(
                                out_per_step[i], cu_seqlens_q_padded, unpad_q_indices
                            )
                            # softmax_lse_per_step[i] = fa_varlen_lse_repad(
                            #     softmax_lse_per_step[i], max_seqlen_q
                            # )
                            softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                softmax_lse_per_step[i],
                                cu_seqlens_q_padded[-1],
                                unpad_q_indices,
                            )
                            # softmax_lse_per_step[i], lse_indices_per_step[i] = fa_varlen_lse_pad2
                            # (softmax_lse_per_step[i], cu_seqlens_q_per_step[i], cu_seqlens_q_padded)

            if i > 0:
                # wait until fwd restuls correction of last step is done
                #
                # 等待修正结束
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(
                        fwd_results_correction_done
                    )

                if use_fused_attention:
                    # [b, np, sq, 1] -> [b, np, sq]
                    softmax_lse_per_step[i - 1].squeeze_(-1)
                # TODO ?
                # if qkv_format != "thd" and softmax_lse_in_packed_format:
                #     # [np, t] -> [np, b, sq]
                #     softmax_lse_per_step[i - 1] = softmax_lse_per_step[i - 1].view(
                #         q.shape[-2], q.shape[0], -1
                #     )
                # 处理lse结果修正
                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if i == 1:
                        out = torch.zeros_like(q)
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(
                            torch.double
                        )
                    elif (i - 1) <= rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        tex.thd_second_half_lse_correction(
                            softmax_lse,
                            softmax_lse_per_step[i - 1],
                            cu_seqlens_q_padded,
                            softmax_lse_in_packed_format,
                        )
                # 等待计算和通信
                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(
                        fwd_results_correction_done
                    )

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        second_half_lse_seqlen = None
        if causal and rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]

        softmax_lse = softmax_lse.to(torch.float)

        for i in range(cp_size):
            if i <= rank or not causal:
                tex.thd_out_correction(
                    out,
                    out_per_step[i],
                    softmax_lse,
                    softmax_lse_per_step[i],
                    cu_seqlens_q_padded,
                    False,
                    softmax_lse_in_packed_format,
                )
                torch.cuda.synchronize()
            else:
                tex.thd_out_correction(
                    out,
                    out_per_step[i],
                    softmax_lse,
                    softmax_lse_per_step[i],
                    cu_seqlens_q_padded,
                    True,
                    softmax_lse_in_packed_format,
                )

        # TODO ？
        # if qkv_format != "thd" and softmax_lse_in_packed_format:
        #     # [np, b, sq] -> [np, t]
        #     softmax_lse = softmax_lse.view(softmax_lse.shape[0], -1)
        kv = p2p_comm_buffers[-1]

        out_f16 = out.to(qkv_dtype)
        out_ret = out_f16

        q_f16 = q_f16.view(q.shape)
        q_save, kv_save, out_save = q_f16, kv, out_f16
        fp8_fwd_scales, fp8_fwd_scale_invs = None, None

        ctx.save_for_backward(
            q_save,
            kv_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            fp8_fwd_scales,
            fp8_fwd_scale_invs,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
            # *lse_indices_per_step,
            # *attn_biases,
        )
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.use_ulysess_low = use_ulysses_low
        ctx.cp_group_a2a = cp_group_a2a
        ctx.cp_size_a2a = cp_size_a2a
        ctx.rank_a2a = rank_a2a
        ctx.cp_group = cp_group
        ctx.cp_global_ranks = cp_global_ranks
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.total_tokens_kv = total_tokens_kv
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.second_half_lse_seqlen = second_half_lse_seqlen

        # TODO
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.fp8 = False
        ctx.fp8_meta = None
        ctx.is_input_fp8 = False
        ctx.is_output_fp8 = False
        print("end forward")

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        # pylint: disable=missing-function-docstring
        cp_size_a2a = ctx.cp_size_a2a
        rank_a2a = ctx.rank_a2a

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        if ctx.use_ulysess_low:
            send_dst = ctx.cp_global_ranks[
                (rank - 1) % cp_size * cp_size_a2a + rank_a2a
            ]
            recv_src = ctx.cp_global_ranks[
                (rank + 1) % cp_size * cp_size_a2a + rank_a2a
            ]
        else:
            send_dst = ctx.cp_global_ranks[
                (rank - 1) % cp_size + rank_a2a * cp_size_a2a
            ]
            recv_src = ctx.cp_global_ranks[
                (rank + 1) % cp_size + rank_a2a * cp_size_a2a
            ]

        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            cp_size == 2
        )

        (*saved_tensors,) = ctx.saved_tensors
        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        ) = saved_tensors[:6]
        (fp8_fwd_scales, fp8_fwd_scale_invs) = saved_tensors[6:8]
        cu_seqlens_q_per_step = saved_tensors[8 : 8 + cp_size]
        cu_seqlens_kv_per_step = saved_tensors[8 + cp_size : 8 + cp_size * 2]
        rng_states = saved_tensors[8 + cp_size * 2 : 8 + cp_size * 3]
        # lse_indices_per_step = saved_tensors[8 + cp_size * 3 : 8 + cp_size * 4]
        # attn_biases = saved_tensors[8 + cp_size * 3 : 8 + cp_size * 4]

        causal = "causal" in ctx.attn_mask_type
        padding = "padding" in ctx.attn_mask_type

        # seq_dim = 0
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
        # attn_dbias = None
        # attn_dbias_ = None

        softmax_lse_in_packed_format = (
            not ctx.use_fused_attention and _flash_attn_2_6_0_plus
        )

        # if causal:
        #     if ctx.qkv_format == "thd" or softmax_lse_in_packed_format:
        #         softmax_lse_ = tex.thd_read_second_half_lse(
        #             softmax_lse, cu_seqlens_q_padded, softmax_lse_in_packed_format
        #         )

        if causal and ctx.second_half_lse_seqlen is not None:
            softmax_lse_ = tex.thd_read_second_half_lse(
                softmax_lse,
                cu_seqlens_q_padded,
                softmax_lse_in_packed_format,
                ctx.second_half_lse_seqlen,
            )
        if ctx.use_fused_attention:
            # [b, np, sq] -> [b, np, sq, 1]
            softmax_lse.unsqueeze_(-1)

        dout_dtype = dout.dtype
        fused_attn_backend = None
        fused_attn_qkv_dtype = None
        fused_attn_dqkv_dtype = None
        # TODO
        # dout_fp8_dtype = None

        dq = torch.empty_like(q)
        if ctx.qkv_format == "thd" and causal:
            dq[cu_seqlens_q_padded[-1] :].fill_(0)
        p2p_comm_buffers = [
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
        ]
        p2p_comm_buffers[0][0].copy_(kv)
        if ctx.use_fused_attention:
            fp8_meta_kwargs = {}
            fused_attn_qkv_dtype = TE_DType[q.dtype]
            fused_attn_dqkv_dtype = TE_DType[dout_dtype]
            fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        send_recv_reqs = []

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            flash_attn_bwd = flash_attn_varlen_bwd
            fa_backward_kwargs["dropout_p"] = ctx.dropout_p
            if _flash_attn_2_3_plus:
                if _flash_attn_2_7_0_plus:
                    fa_backward_kwargs["window_size_left"] = ctx.window_size[0]
                    fa_backward_kwargs["window_size_right"] = ctx.window_size[1]
                    fa_backward_kwargs["softcap"] = ctx.softcap
                else:
                    fa_backward_kwargs["window_size"] = ctx.window_size
            if _flash_attn_2_4_plus:
                fa_backward_kwargs["alibi_slopes"] = None
            if _flash_attn_2_4_1_plus:
                fa_backward_kwargs["deterministic"] = ctx.deterministic

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]
            if i == 0:
                send_tensor = send_tensor[0]
                recv_tensor = recv_tensor[0]
            if i == (cp_size - 1):
                send_tensor = send_tensor[1]
                recv_tensor = recv_tensor[1]
            send_recv_reqs = flash_attn_p2p_communicate(
                rank,
                send_tensor,
                send_dst,
                recv_tensor,
                recv_src,
                ctx.cp_group,
                batch_p2p_comm,
            )

            kv = p2p_comm_buffers[i % 2][0]
            dk_, dv_ = None, None

            if causal:
                if i == (cp_size - 1):  # q,k,v
                    if ctx.use_fused_attention:
                        q_, kv_, out_, dout_ = q, kv, out, dout
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_,
                            kv_[0],
                            kv_[1],
                            out_,
                            dout_,
                            fused_attn_qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                    else:
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        q_ = q.view(-1, *q.shape[-2:])
                        # dq_ = torch.zeros_like(q_)
                        # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                        kv_ = kv.view(2, -1, *kv.shape[-2:])
                        # dkv_ = torch.empty_like(kv_)
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        out_ = out.view(-1, *out.shape[-2:])
                        dout_ = dout.view(-1, *dout.shape[-2:])
                        # if _flash_attn_2_3_plus:
                        #     fa_backward_kwargs["window_size"] = (-1, 0)
                        fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                        # unpad
                        (
                            unpad_q_,
                            unpad_q_indices,
                            max_seqlen_per_step_q,
                        ) = fa_varlen_thd_unpad(
                            q_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded,
                        )
                        (
                            unpad_kv_,
                            unpad_kv_indices,
                            max_seqlen_per_step_kv,
                        ) = fa_varlen_thd_unpad(
                            kv_,
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            cu_seqlens_kv_padded,
                            packed=True,
                        )
                        dq_ = torch.zeros_like(unpad_q_)
                        dkv_ = torch.empty_like(unpad_kv_)
                        unpad_out_, _, _ = fa_varlen_thd_unpad(
                            out_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded,
                        )
                        unpad_dout_, _, _ = fa_varlen_thd_unpad(
                            dout_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded,
                        )
                        # unpad_softmax_lse = fa_varlen_lse_repad(
                        #     softmax_lse, max_seqlen_per_step_q
                        # )
                        unpad_softmax_lse = fa_varlen_lse_unpad(
                            softmax_lse, unpad_q_indices
                        )
                        flash_attn_bwd(
                            unpad_dout_,
                            unpad_q_,
                            unpad_kv_[0],
                            unpad_kv_[1],
                            unpad_out_,
                            unpad_softmax_lse,
                            dq_,
                            dkv_[0],
                            dkv_[1],
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            max_seqlen_per_step_q,
                            max_seqlen_per_step_kv,
                            causal=True,
                            **fa_backward_kwargs,
                        )
                        # pad
                        dq_ = fa_varlen_thd_pad(
                            dq_, cu_seqlens_q_padded, unpad_q_indices
                        )
                        dkv_ = fa_varlen_thd_pad(
                            dkv_, cu_seqlens_kv_padded, unpad_kv_indices, packed=True
                        )
                elif i >= (cp_size - rank - 1):  # q,k0,v0
                    if ctx.use_fused_attention:
                        q_, out_, dout_ = q, out, dout
                        # [2, t, np, hn] -> [2, t/2, np, hn]
                        kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
                        aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q,
                            ctx.max_seqlen_kv // 2,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_,
                            kv_[0],
                            kv_[1],
                            out_,
                            dout_,
                            fused_attn_qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=(
                                None
                                if cu_seqlens_kv_padded is None
                                else cu_seqlens_kv_padded // 2
                            ),
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                    else:
                        q_ = q.view(-1, *q.shape[-2:])
                        # dq_ = torch.zeros_like(q_)
                        kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
                        # dkv_ = torch.empty_like(kv_)
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        out_ = out.view(-1, *out.shape[-2:])
                        dout_ = dout.view(-1, *dout.shape[-2:])
                        # if _flash_attn_2_3_plus:
                        #     fa_backward_kwargs["window_size"] = (-1, -1)
                        fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                        # unpad
                        (
                            unpad_q_,
                            unpad_q_indices,
                            max_seqlen_per_step_q,
                        ) = fa_varlen_thd_unpad(
                            q_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded,
                        )
                        (
                            unpad_kv_,
                            unpad_kv_indices,
                            max_seqlen_per_step_kv,
                        ) = fa_varlen_thd_unpad(
                            kv_,
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            cu_seqlens_kv_padded // 2,
                            packed=True,
                        )
                        dq_ = torch.zeros_like(unpad_q_)
                        dkv_ = torch.empty_like(unpad_kv_)
                        unpad_out_, _, _ = fa_varlen_thd_unpad(
                            out_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded,
                        )
                        unpad_dout_, _, _ = fa_varlen_thd_unpad(
                            dout_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded,
                        )
                        # unpad_softmax_lse = fa_varlen_lse_repad(
                        #     softmax_lse, max_seqlen_per_step_q
                        # )
                        unpad_softmax_lse = fa_varlen_lse_unpad(
                            softmax_lse, unpad_q_indices
                        )
                        flash_attn_bwd(
                            unpad_dout_,
                            unpad_q_,
                            unpad_kv_[0],
                            unpad_kv_[1],
                            unpad_out_,
                            unpad_softmax_lse,
                            dq_,
                            dkv_[0],
                            dkv_[1],
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            max_seqlen_per_step_q,
                            max_seqlen_per_step_kv,
                            causal=False,
                            **fa_backward_kwargs,
                        )
                        # pad
                        dq_ = fa_varlen_thd_pad(
                            dq_, cu_seqlens_q_padded, unpad_q_indices
                        )
                        dkv_ = fa_varlen_thd_pad(
                            dkv_,
                            cu_seqlens_kv_padded // 2,
                            unpad_kv_indices,
                            packed=True,
                        )
                else:  # q1,k,v
                    if ctx.use_fused_attention:
                        q_ = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                        out_ = tex.thd_read_half_tensor(out, cu_seqlens_q_padded, 1)
                        dout_ = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, 1)
                        kv_ = kv
                        aux_ctx_tensors = [softmax_lse_, rng_states[cp_size - i - 1]]
                        dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                            ctx.max_seqlen_q // 2,
                            ctx.max_seqlen_kv,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            q_,
                            kv_[0],
                            kv_[1],
                            out_,
                            dout_,
                            fused_attn_qkv_dtype,
                            fused_attn_dqkv_dtype,
                            aux_ctx_tensors,
                            fused_attn_backend,
                            cu_seqlens_q_padded=(
                                None
                                if cu_seqlens_q_padded is None
                                else cu_seqlens_q_padded // 2
                            ),
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type="padding" if padding else "no_mask",
                            attn_bias_type=ctx.attn_bias_type,
                            deterministic=ctx.deterministic,
                            **fp8_meta_kwargs,
                        )
                    else:
                        q_ = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, 1)
                        # dq_ = torch.zeros_like(q_)
                        # [2, b, 2, sk//2, np, hn] -> [2, b*sk, np, hn]
                        kv_ = kv.view(2, -1, *kv.shape[-2:])
                        # dkv_ = torch.empty_like(kv_)
                        out_ = tex.thd_read_half_tensor(out, cu_seqlens_q_padded, 1)
                        dout_ = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, 1)
                        # if _flash_attn_2_3_plus:
                        #     fa_backward_kwargs["window_size"] = (-1, -1)
                        fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                        # unpad
                        (
                            unpad_q_,
                            unpad_q_indices,
                            max_seqlen_per_step_q,
                        ) = fa_varlen_thd_unpad(
                            q_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded // 2,
                        )
                        (
                            unpad_kv_,
                            unpad_kv_indices,
                            max_seqlen_per_step_kv,
                        ) = fa_varlen_thd_unpad(
                            kv_,
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            cu_seqlens_kv_padded,
                            packed=True,
                        )
                        dq_ = torch.zeros_like(unpad_q_)
                        dkv_ = torch.empty_like(unpad_kv_)
                        unpad_out_, _, _ = fa_varlen_thd_unpad(
                            out_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded // 2,
                        )
                        unpad_dout_, _, _ = fa_varlen_thd_unpad(
                            dout_,
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_q_padded // 2,
                        )
                        # unpad_softmax_lse_ = fa_varlen_lse_repad(
                        #     softmax_lse_, max_seqlen_per_step_q
                        # )
                        unpad_softmax_lse_ = fa_varlen_lse_unpad(
                            softmax_lse_, unpad_q_indices
                        )
                        flash_attn_bwd(
                            unpad_dout_,
                            unpad_q_,
                            unpad_kv_[0],
                            unpad_kv_[1],
                            unpad_out_,
                            unpad_softmax_lse_,
                            dq_,
                            dkv_[0],
                            dkv_[1],
                            cu_seqlens_q_per_step[cp_size - i - 1],
                            cu_seqlens_kv_per_step[cp_size - i - 1],
                            max_seqlen_per_step_q,
                            max_seqlen_per_step_kv,
                            causal=False,
                            **fa_backward_kwargs,
                        )
                        # pad
                        dq_ = fa_varlen_thd_pad(
                            dq_, cu_seqlens_q_padded // 2, unpad_q_indices
                        )
                        dkv_ = fa_varlen_thd_pad(
                            dkv_, cu_seqlens_kv_padded, unpad_kv_indices, packed=True
                        )
            else:
                if ctx.use_fused_attention:
                    aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                    dq_, dk_, dv_, dbias_ = fused_attn_bwd(
                        ctx.max_seqlen_q,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        q,
                        kv[0],
                        kv[1],
                        out,
                        dout,
                        fused_attn_qkv_dtype,
                        fused_attn_dqkv_dtype,
                        aux_ctx_tensors,
                        fused_attn_backend,
                        cu_seqlens_q_padded=cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        attn_scale=ctx.softmax_scale,
                        dropout=ctx.dropout_p,
                        qkv_layout=qkv_layout,
                        attn_mask_type=ctx.attn_mask_type,
                        attn_bias_type=ctx.attn_bias_type,
                        deterministic=ctx.deterministic,
                        **fp8_meta_kwargs,
                    )
                else:
                    # [b, sq, np, hn] -> [b*sq, np, hn]
                    q_ = q.view(-1, *q.shape[-2:])
                    # dq_ = torch.zeros_like(q_)
                    # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
                    kv_ = kv.view(2, -1, *kv.shape[-2:])
                    # dkv_ = torch.empty_like(kv_)
                    # [b, sq, np, hn] -> [b*sq, np, hn]
                    out_ = out.view(-1, *out.shape[-2:])
                    dout_ = dout.view(-1, *dout.shape[-2:])
                    # if _flash_attn_2_3_plus:
                    #     fa_backward_kwargs["window_size"] = (-1, -1)
                    fa_backward_kwargs["rng_state"] = rng_states[cp_size - i - 1]
                    # unpad
                    # print('cu step',cu_seqlens_q_per_step[cp_size - i - 1][:-1],flush=True)
                    (
                        unpad_q_,
                        unpad_q_indices,
                        max_seqlen_per_step_q,
                    ) = fa_varlen_thd_unpad(
                        q_, cu_seqlens_q_per_step[cp_size - i - 1], cu_seqlens_q_padded
                    )
                    (
                        unpad_kv_,
                        unpad_kv_indices,
                        max_seqlen_per_step_kv,
                    ) = fa_varlen_thd_unpad(
                        kv_,
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        cu_seqlens_kv_padded,
                        packed=True,
                    )
                    dq_ = torch.zeros_like(unpad_q_)
                    dkv_ = torch.empty_like(unpad_kv_)
                    unpad_out_, _, _ = fa_varlen_thd_unpad(
                        out_,
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_q_padded,
                    )
                    unpad_dout_, _, _ = fa_varlen_thd_unpad(
                        dout_,
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_q_padded,
                    )
                    # unpad_softmax_lse = fa_varlen_lse_repad(
                    #     softmax_lse, max_seqlen_per_step_q
                    # )
                    unpad_softmax_lse = fa_varlen_lse_unpad(
                        softmax_lse, unpad_q_indices
                    )
                    flash_attn_bwd(
                        unpad_dout_,
                        unpad_q_,
                        unpad_kv_[0],
                        unpad_kv_[1],
                        unpad_out_,
                        unpad_softmax_lse,
                        dq_,
                        dkv_[0],
                        dkv_[1],
                        cu_seqlens_q_per_step[cp_size - i - 1],
                        cu_seqlens_kv_per_step[cp_size - i - 1],
                        max_seqlen_per_step_q,
                        max_seqlen_per_step_kv,
                        causal=False,
                        **fa_backward_kwargs,
                    )
                    # pad
                    dq_ = fa_varlen_thd_pad(dq_, cu_seqlens_q_padded, unpad_q_indices)
                    dkv_ = fa_varlen_thd_pad(
                        dkv_, cu_seqlens_kv_padded, unpad_kv_indices, packed=True
                    )
            if i >= (cp_size - rank - 1) or not causal:
                # [b*sq, np, hn] -> [b, 2, sq//2, np, hn] if causal
                # [b*sq, np, hn] -> [b, sq, np, hn] if not causal
                dq_ = dq_.view(*dq.shape)
            if causal:
                if i > (cp_size - rank - 1):
                    dq.add_(dq_)
                elif i == (cp_size - rank - 1):
                    if rank == (cp_size - 1):
                        dq.copy_(dq_)
                    else:
                        tex.thd_grad_correction(
                            dq, dq_, cu_seqlens_q_padded, "copy", "add"
                        )
                elif i > 0:
                    tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, "none", "add")
                else:
                    tex.thd_grad_correction(
                        dq, dq_, cu_seqlens_q_padded, "none", "copy"
                    )
            else:
                if i == 0:
                    dq.copy_(dq_)
                else:
                    dq.add_(dq_)

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            dkv = p2p_comm_buffers[(i + 1) % 2][1]
            if ctx.use_fused_attention:
                dkv_ = torch.cat(
                    (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                )  # pylint: disable=used-before-assignment
            if causal and i >= (cp_size - rank - 1) and i != (cp_size - 1):
                pass
            else:
                dkv_ = dkv_.view(*dkv.shape)

            if causal:
                if i == (cp_size - 1):
                    if rank == 0:
                        tex.thd_grad_correction(
                            dkv, dkv_, cu_seqlens_kv_padded, "add", "copy"
                        )
                    else:
                        dkv.add_(dkv_)
                elif i >= (cp_size - rank - 1):
                    if i == 0 and rank == (cp_size - 1):
                        tex.thd_grad_correction(
                            dkv, dkv_, cu_seqlens_kv_padded, "copy", "none"
                        )
                    else:
                        tex.thd_grad_correction(
                            dkv, dkv_, cu_seqlens_kv_padded, "add", "none"
                        )
                elif i > 0:
                    dkv.add_(dkv_)
                else:
                    dkv.copy_(dkv_)
            else:
                if i == 0:
                    dkv.copy_(dkv_)
                else:
                    dkv.add_(dkv_)

        dkv_ = torch.empty(
            2, ctx.total_tokens_kv, *dkv.shape[-2:], dtype=dkv.dtype, device=dkv.device
        )
        dkv_[:, : cu_seqlens_kv_padded[-1]].copy_(dkv)
        dkv_[:, cu_seqlens_kv_padded[-1] :].fill_(0)
        dkv = dkv_
        dk, dv = dkv[0], dkv[1]

        print("end bwd", dq.shape, dk.shape, dv.shape)

        return (
            None,
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
            None,
            None,
            None,
            None,
            None,
        )


#
#    [0,1] -> ulysess group
#    [2,3]
#     ^
#     |
#  ring group


# USP: ulysess + ring attn
# all2all + p2p
# dispatch pad -> unpad -> attn -> pad -> undispatch unpad
class TEUSP(AttnBaselineInterface):
    def __init__(self):
        super().__init__()

        self.packed_seq_params = {"q": None, "k": None, "v": None}

    # TODO a2a_pg, p2p_pg  |  p2p_pg, a2a_pg
    # use_ulysses_low
    def dispatch(
        self,
        x_global: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        cp_group: dist.ProcessGroup,
        **kwargs,
    ) -> torch.Tensor:
        """
        Dispatch the global tensor `x_global` along its sequence dim following the meta info,
        and return the dispatched local tensor `x_local`

        Args:
            x_global (torch.Tensor): the global tensor to be dispatched, with shape [t, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the dispatched local tensor
        """

        print(f"Extra kwargs received: {kwargs}")
        ranges = kwargs.get("ranges", None)
        attention_mask_thd = kwargs.get("attention_mask_thd", None)

        # attention_mask = kwargs.get("attention_mask", None)
        # assert (attention_mask is not None), "TEUSP dispatch needs attention mask!"
        use_ulysess_low = kwargs.get("use_ulysess_low", True)
        qkv_ = kwargs.get("qkv_", "q")

        assert isinstance(
            cp_group, list
        ), "Hierarchical CP implementation needs multi-level CP groups!"
        assert (
            len(cp_group) == 2
        ), "Current implementation only supports two-level CP groups!"

        # assert qkv_ in ["q", "k", "v"], "qkv_ must be one of ['q', 'k', 'v']!"

        input_dim = x_global.dim()
        assert input_dim >= 2
        # seqlen, batch_size, num_key_value_heads, head_dim = x_global.shape

        if use_ulysess_low:
            cp_group_a2a = cp_group[0]
            cp_group_p2p = cp_group[1]
        else:
            cp_group_a2a = cp_group[1]
            cp_group_p2p = cp_group[0]

        cp_size_a2a = get_distributed_world_size(cp_group_a2a)
        rank_a2a = get_distributed_rank(cp_group_a2a)
        cp_size_p2p = get_distributed_world_size(cp_group_p2p)
        rank_p2p = get_distributed_rank(cp_group_p2p)

        assert (
            cp_size_a2a * cp_size_p2p == cp_size
        ), "Current two-level CP groups need cp_size_a2a*cp_size_p2p == cp_size!"

        # 2*cp*up
        self.padding_factor = 2 * cp_size

        total_seqlen = attention_mask_thd.sum(dim=0, dtype=torch.int32).item()
        cu_seqlens = torch.tensor(
            ranges.to_cu_seqlens(seq_len=total_seqlen),
            device=x_global.device,
            dtype=torch.int32,
        )
        assert total_seqlen == cu_seqlens[-1], "total_seqlen != cu_seqlens[-1]"

        # indices, indices_in_thd_padded, cu_seqlens, cu_seqlens_padded, max_seqlen_in_batch, max_seqlen_in_padded =
        # _get_unpad_data(attention_mask=attention_mask, padding_factor=self.padding_factor, seq_dim = 0)
        (
            indices,
            indices_in_thd_padded,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        ) = _get_varlen_unpad_data(
            attention_mask_thd=attention_mask_thd,
            cu_seqlens=cu_seqlens,
            padding_factor=self.padding_factor,
        )
        self.packed_seq_params[qkv_] = PackedSeqParams(
            indices,
            indices_in_thd_padded,
            cu_seqlens,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        )

        # s,b,h,d -> b,s,h,d -> t,h,d
        # x_local = zigzag_teusp_dispatch(x_global.permute(1, 0, 2, 3).reshape(seqlen*batch_size, num_key_value_heads,
        # head_dim), self.packed_seq_params, cp_size_a2a, rank_a2a, cp_size_p2p, rank_p2p)

        # thd -> unpad thd pad
        x_local = zigzag_teusp_dispatch(
            x_global,
            self.packed_seq_params[qkv_],
            cp_size_a2a,
            rank_a2a,
            cp_size_p2p,
            rank_p2p,
        )

        # x_local = DISPATCH_FUNC_DICT[ring_impl_type](
        #     x_global=x_global, rank=cp_rank, world_size=cp_size, cp_group=cp_group, dim=0
        # ).detach().clone()

        return x_local

        # raise NotImplementedError("TODO (butao)")

    def undispatch(
        self,
        x_local: torch.Tensor,  # t,h,d
        cp_rank: int,
        cp_size: int,
        cp_group: Optional[Union[dist.ProcessGroup, List[dist.ProcessGroup]]],
        **kwargs,
    ) -> torch.Tensor:
        """
        Undispatch the local tensor `x_local` along its sequence dim following the meta info,
        and return the undispatched global tensor `x_global`

        Args:
            x_local (torch.Tensor): the local tensor to be undispatched, with shape [t, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the undispatched global tensor
        """

        print(f"Extra kwargs received: {kwargs}")
        # optional[basic, zigzag_te, zigzag_usp], default: basic
        # ring_impl_type = kwargs.get("ring_impl_type", "a2a+p2p")
        use_ulysess_low = kwargs.get("use_ulysess_low", True)
        qkv_ = kwargs.get("qkv_", "q")

        # if use_ulysess_low:
        #     cp_group_a2a = cp_group[0]
        #     cp_group_p2p = cp_group[1]
        # else:
        #     cp_group_a2a = cp_group[1]
        #     cp_group_p2p = cp_group[0]

        assert isinstance(
            cp_group, list
        ), "Hierarchical CP implementation needs multi-level CP groups!"
        assert (
            len(cp_group) == 2
        ), "Current implementation only supports two-level CP groups!"

        # assert qkv_ in ["q", "k", "v"], "qkv_ must be one of ['q', 'k', 'v']!"

        x_global = zigzag_teusp_undispatch(
            x_local,
            self.packed_seq_params[qkv_],
            cp_size,
            cp_group,
            use_ulysess_low=use_ulysess_low,
        )

        # x_global = UNDISPATCH_FUNC_DICT[ring_impl_type](
        #     x_global=x_local, rank=cp_rank, world_size=cp_size, cp_group=cp_group, dim=0
        # ).detach().clone()

        return x_global

        # raise NotImplementedError("TODO (butao)")

    # TODO cp_stream
    def apply_attn(
        # self,
        # q: torch.Tensor,
        # k: torch.Tensor,
        # v: torch.Tensor,
        # q_ranges: AttnRanges,
        # k_ranges: AttnRanges,
        # attn_mask_type: AttnMaskType | list[AttnMaskType],
        # max_seqlen_q: int,
        # max_seqlen_k: int,
        # softmax_scale: float,
        # deterministic: bool,
        # cp_group: Optional[Union[dist.ProcessGroup, List[dist.ProcessGroup]]],
        # cp_size: int,  # world_size
        # **kwargs,
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
        """Apply the attention with the given meta info

        Args:
            q (torch.Tensor): the query tensor, with shape [total_seqlen_q, nhq, hd]
            k (torch.Tensor): the key tensor, with shape [total_seqlen_k, nhk, hd]
            v (torch.Tensor): the value tensor, with shape [total_seqlen_k, nhk, hd]
            q_ranges (AttnRanges): the query ranges, with length of batch_size
            k_ranges (AttnRanges): the key ranges, with length of batch_size
            attn_mask_type (AttnMaskType | list[AttnMaskType]): the attention mask type,
                1. a single enum to indicate the mask type for each sample in the batch
                2. a list of enum with length of batch_size
            max_seqlen_q (int): the maximum sequence length of the query
            max_seqlen_k (int): the maximum sequence length of the key
            softmax_scale (float): the softmax scale
            deterministic (bool): whether to use deterministic mode
            **kwargs: additional arguments

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                1. the output tensor, with shape [total_seqlen_q, b, nhq, hd]
                2. the softmax lse tensor, with shape [b, nhq, max_seqlen_q]
        """

        print(f"Extra kwargs received: {kwargs}")
        # cp_comm_type = kwargs.get("cp_comm_type", "a2a+p2p")
        use_ulysess_low = kwargs.get("use_ulysess_low", True)
        use_sync = kwargs.get("use_sync", False)
        use_fused_attention = kwargs.get("use_fused_attention", True)
        cp_group = kwargs.get("cp_group", None)

        assert isinstance(
            cp_group, list
        ), "Hierarchical CP implementation needs multi-level CP groups!"
        assert (
            len(cp_group) == 2
        ), "Current implementation only supports two-level CP groups!"

        if use_ulysess_low:
            cp_group_a2a = cp_group[0]
            # cp_group_p2p = cp_group[1]
        else:
            cp_group_a2a = cp_group[1]
            # cp_group_p2p = cp_group[0]

        # cp_size_a2a = get_distributed_world_size(cp_group_a2a)
        # cp_size_p2p = get_distributed_world_size(cp_group_p2p)

        dropout_p = kwargs.get("dropout_p", 0.0)
        cp_global_ranks = kwargs.get("cp_global_ranks", None)
        cp_stream = kwargs.get("cp_stream", None)

        # ulysess all2all

        query_layer = SeqAllToAll3D.apply(cp_group_a2a, q, 1, 0, use_sync)
        key_layer = SeqAllToAll3D.apply(cp_group_a2a, k, 1, 0, use_sync)
        value_layer = SeqAllToAll3D.apply(cp_group_a2a, v, 1, 0, use_sync)

        print("fwd", query_layer.shape, key_layer.shape, value_layer.shape, flush=True)

        assert isinstance(
            attn_mask_type, AttnMaskType
        ), "attn_mask_type must be an AttnMaskType!"
        if attn_mask_type == AttnMaskType.CAUSAL:
            teusp_attn_mask_type = "padding_causal"
        else:
            teusp_attn_mask_type = "padding"

        # dist.barrier(group=cp_group_a2a)
        # torch.cuda.synchronize()

        context_layer, softmax_lse = ZigZagTEUSPAttnVarlenFunc.apply(
            True,
            query_layer,
            key_layer,
            value_layer,
            self.packed_seq_params["q"].cu_seqlens,
            self.packed_seq_params["k"].cu_seqlens,
            self.packed_seq_params["q"].max_seqlen_in_padded,
            self.packed_seq_params["k"].max_seqlen_in_padded,
            self.packed_seq_params["q"].cu_seqlens_padded,
            self.packed_seq_params["k"].cu_seqlens_padded,
            dropout_p,
            softmax_scale,
            teusp_attn_mask_type,
            deterministic,
            use_fused_attention,
            cp_group,
            cp_global_ranks,
            cp_stream,
            use_ulysess_low,
        )

        # context_layer_ = context_layer.unsqueeze(0)
        out = SeqAllToAll3D.apply(cp_group_a2a, context_layer, 0, 1, use_sync)

        # out.retain_grad()

        # 对于 heads all gather
        # lse_group = [torch.empty_like(softmax_lse) for _ in range(cp_size_a2a)]
        # dist.all_gather(lse_group, softmax_lse, group=cp_group_a2a)
        # b,h,max_seqlen_q
        # out_lse = torch.cat(lse_group, dim=1)

        # return None, None
        return out, softmax_lse

        # raise NotImplementedError("TODO (butao)")
