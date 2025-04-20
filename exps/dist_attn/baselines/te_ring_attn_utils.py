import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_pkg_version
from typing import List, Tuple

import torch
import transformer_engine  # noqa
import transformer_engine_torch as tex
from einops import rearrange, repeat
from packaging.version import Version as PkgVersion
from transformer_engine.pytorch.attention import (
    _get_supported_versions,
    fa_logger,
    flash_attn_fwd_softmax_lse_correction,
    flash_attn_p2p_communicate,
    get_cu_seqlens_on_cp_rank,
)
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import (
    get_distributed_rank,
    get_distributed_world_size,
    reduce_scatter_along_first_dim,
)

from dffa.comm.functional import all_gather_fwd_scatter_bwd
from dffa.common.enum import AttnMaskType

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


def pad_tensor_and_split(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    cp_size: int,
    rank: int,
    padded_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """pad tensor to cu_seqlens_padded and split according to rank and cp_size

    Args:
        x (torch.Tensor): tensor without padding, shape of (total_s, ...)
        cu_seqlens (torch.Tensor): sequence length before tensor padding
        cu_seqlens_padded (torch.Tensor): sequence length after tensor padding
        cp_size (int): cp_size for context parallel
        rank (int): current rank for the device

    Returns:
        torch.Tensor: split tensor with padding of (split_s, ...)
    """
    assert cu_seqlens.shape == cu_seqlens_padded.shape
    assert x.shape[0] == cu_seqlens[-1]

    dispatch_seqlens = cu_seqlens_padded // cp_size
    seqlens_padded = (dispatch_seqlens[1:] - dispatch_seqlens[:-1]) // 2
    if padded_tensor is None:
        total_seqlens = dispatch_seqlens[-1]
        padded_shape = (total_seqlens,) + x.shape[1:]
        padded_tensor = torch.zeros(padded_shape, dtype=x.dtype, device=x.device)

    for i in range(len(cu_seqlens) - 1):
        first_half_start = dispatch_seqlens[i]
        x_first_half_start = min(
            rank * seqlens_padded[i] + cu_seqlens[i], cu_seqlens[i + 1]
        )
        x_first_half_end = min(
            x_first_half_start + seqlens_padded[i], cu_seqlens[i + 1]
        )
        first_half_end = first_half_start + (x_first_half_end - x_first_half_start)
        padded_tensor[first_half_start:first_half_end] = x[
            x_first_half_start:x_first_half_end
        ]

        second_half_start = first_half_start + seqlens_padded[i]
        x_second_half_start = min(
            (2 * cp_size - rank - 1) * seqlens_padded[i] + cu_seqlens[i],
            cu_seqlens[i + 1],
        )
        x_second_half_end = min(
            x_second_half_start + seqlens_padded[i], cu_seqlens[i + 1]
        )
        second_half_end = second_half_start + (x_second_half_end - x_second_half_start)
        padded_tensor[second_half_start:second_half_end] = x[
            x_second_half_start:x_second_half_end
        ]

    return padded_tensor


def get_max_seqlen(cu_seqlens: torch.Tensor) -> int:
    """_summary_

    Args:
        cu_seqlens (torch.Tensor): _description_

    Returns:
        int: _description_
    """
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlens = torch.max(seqlens)
    return max_seqlens.item()


def compute_cu_seqlens_padded_with_attention_mask(
    cu_seqlens: torch.Tensor, attention_mask: torch.LongTensor
) -> torch.Tensor:
    """_summary_

    Args:
        cu_seqlens (torch.Tensor): _description_
        attention_mask (torch.LongTensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    valid_indices = torch.nonzero(attention_mask == 1, as_tuple=True)[0]
    next_start_indices = torch.cat(
        [
            valid_indices[cu_seqlens[1:-1]],
            torch.tensor([len(attention_mask)], device=attention_mask.device),
        ]
    )
    cu_seqlens_padded = torch.nn.functional.pad(next_start_indices, (1, 0))
    return cu_seqlens_padded


def unpad_tensor_after_gather(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """unpad tensor to cu_seqlens after all_gather

    Args:
        x (torch.Tensor): tensor after all_gather, shape of (cp_size, s, ...)
        cu_seqlens (torch.Tensor): sequence length before tensor padding
        cu_seqlens_padded (torch.Tensor): sequence length after tensor padding
        cp_size (int): cp_size for context parallel

    Returns:
        torch.Tensor: full tensor without padding
    """
    assert cu_seqlens.shape == cu_seqlens_padded.shape
    assert x.shape[0] * x.shape[1] == cu_seqlens_padded[-1]

    batch_size = cu_seqlens.shape[0] - 1
    total_seqlens_unpad = cu_seqlens[-1]
    restore_shape = (total_seqlens_unpad,) + x.shape[2:]
    restore_tensor = torch.zeros(restore_shape, dtype=x.dtype, device=x.device)
    cu_seqlens_padded = cu_seqlens_padded // cp_size

    for i in range(batch_size):
        unpad_end = cu_seqlens[i + 1]
        chunk_size = cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]
        chunk_size = chunk_size // 2

        for rank in range(cp_size):
            padded_start = cu_seqlens_padded[i]
            unpad_start = cu_seqlens[i]

            unpad_first_half_start = min(unpad_end, unpad_start + chunk_size * rank)
            unpad_first_half_end = min(unpad_end, unpad_first_half_start + chunk_size)
            first_half_length = unpad_first_half_end - unpad_first_half_start
            padded_first_half_start = padded_start
            padded_first_half_end = padded_first_half_start + first_half_length
            restore_tensor[unpad_first_half_start:unpad_first_half_end] = x[rank][
                padded_first_half_start:padded_first_half_end
            ]

            unpad_second_half_start = min(
                unpad_end, unpad_start + chunk_size * (2 * cp_size - rank - 1)
            )
            unpad_second_half_end = min(unpad_end, unpad_second_half_start + chunk_size)
            second_half_length = unpad_second_half_end - unpad_second_half_start
            padded_second_half_start = padded_start + chunk_size
            padded_second_half_end = padded_second_half_start + second_half_length
            restore_tensor[unpad_second_half_start:unpad_second_half_end] = x[rank][
                padded_second_half_start:padded_second_half_end
            ]

    return restore_tensor


def fa_varlen_thd_unpad(
    input: torch.Tensor,
    cu_seqlens_per_step: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    packed=False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_per_step = cu_seqlens_per_step[1:] - cu_seqlens_per_step[:-1]
    max_seqlen_per_step = seqlens_per_step.max().item()
    batch = len(seqlens_per_step)
    indices: List[int] = []
    for i in range(batch):
        start_idx_padded = cu_seqlens_padded[i]
        valid_length = seqlens_per_step[i]
        indices.extend(range(start_idx_padded, start_idx_padded + valid_length))
    unpad_indices = torch.tensor(indices, device=input.device, dtype=torch.int64)

    if packed:
        other_shape = input[0].shape[1:]
    else:
        other_shape = input.shape[1:]
    second_dim = other_shape.numel()

    if packed:
        unpad_0 = (
            torch.gather(
                rearrange(input[0], "b ... -> b (...)"),
                0,
                repeat(unpad_indices, "z -> z d", d=second_dim),
            )
            .reshape(-1, *other_shape)
            .contiguous()
        )
        unpad_1 = (
            torch.gather(
                rearrange(input[1], "b ... -> b (...)"),
                0,
                repeat(unpad_indices, "z -> z d", d=second_dim),
            )
            .reshape(-1, *other_shape)
            .contiguous()
        )
        unpad_input = torch.stack([unpad_0, unpad_1], dim=0)
    else:
        unpad_input = (
            torch.gather(
                rearrange(input, "b ... -> b (...)"),
                0,
                repeat(unpad_indices, "z -> z d", d=second_dim),
            )
            .reshape(-1, *other_shape)
            .contiguous()
        )
    return (unpad_input, unpad_indices, max_seqlen_per_step)


def fa_varlen_thd_pad(
    input: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpad_indices: torch.Tensor,
    packed=False,
) -> torch.Tensor:
    total_padded_len = cu_seqlens_padded[-1]
    if packed:
        other_shape = input[0].shape[1:]
    else:
        other_shape = input.shape[1:]
    second_dim = other_shape.numel()
    if packed:
        pad_input_0 = torch.zeros(
            total_padded_len, second_dim, device=input[0].device, dtype=input[0].dtype
        )
        pad_input_1 = torch.zeros(
            total_padded_len, second_dim, device=input[0].device, dtype=input[0].dtype
        )
        input_0 = rearrange(input[0], "b ... -> b (...)")
        input_1 = rearrange(input[1], "b ... -> b (...)")
        pad_input_0.scatter_(
            0, repeat(unpad_indices, "z -> z d", d=second_dim), input_0
        )
        pad_input_1.scatter_(
            0, repeat(unpad_indices, "z -> z d", d=second_dim), input_1
        )
        pad_input_0 = pad_input_0.reshape(-1, *other_shape).contiguous()
        pad_input_1 = pad_input_1.reshape(-1, *other_shape).contiguous()
        pad_input = torch.stack([pad_input_0, pad_input_1], dim=0)
    else:
        pad_input = torch.zeros(
            total_padded_len, second_dim, device=input.device, dtype=input.dtype
        )
        input = rearrange(input, "b ... -> b (...)")
        pad_input.scatter_(0, repeat(unpad_indices, "z -> z d", d=second_dim), input)
        pad_input = pad_input.reshape(-1, *other_shape).contiguous()
    return pad_input


def fa_varlen_lse_repad(lse: torch.Tensor, max_seqlen_pad: int):
    max_seqlen_pad_prime = lse.shape[2]
    repad_lse = torch.zeros(
        (lse.shape[0], lse.shape[1], max_seqlen_pad), device=lse.device, dtype=lse.dtype
    )
    lse_seq_len = min(max_seqlen_pad_prime, max_seqlen_pad)
    repad_lse[:, :, :lse_seq_len] = lse[:, :, :lse_seq_len]
    return repad_lse


def fa_varlen_lse_unpad(
    input: torch.Tensor,
    unpad_indices: torch.Tensor,
):
    head_dim = input.shape[0]
    unpad_input = (
        torch.gather(
            input,
            1,
            repeat(unpad_indices, "z -> d z", d=head_dim),
        )
        .reshape(head_dim, -1)
        .contiguous()
    )
    return unpad_input


def fa_varlen_lse_pad(
    input: torch.Tensor, total_padded_len: int, unpad_indices: torch.Tensor
):
    head_dim = input.shape[0]
    pad_input = torch.zeros(
        head_dim, total_padded_len, device=input.device, dtype=input.dtype
    )
    pad_input.scatter_(1, repeat(unpad_indices, "z -> d z", d=head_dim), input)
    pad_input = pad_input.reshape(head_dim, -1).contiguous()
    return pad_input


def fa_varlen_lse_pad2(
    input: torch.Tensor,
    cu_seqlens_per_step: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
):
    seqlens_per_step = cu_seqlens_per_step[1:] - cu_seqlens_per_step[:-1]
    # max_seqlen_per_step = seqlens_per_step.max().item()
    batch = len(seqlens_per_step)
    indices: list[int] = []
    for i in range(batch):
        start_idx_padded = cu_seqlens_padded[i]
        valid_length = seqlens_per_step[i]
        indices.extend(range(start_idx_padded, start_idx_padded + valid_length))
    unpad_indices = torch.tensor(indices, device=input.device, dtype=torch.int64)

    total_padded_len = cu_seqlens_padded[-1]
    head_dim = input.shape[0]
    pad_input = torch.zeros(
        head_dim, total_padded_len, device=input.device, dtype=input.dtype
    )
    pad_input.scatter_(1, repeat(unpad_indices, "z -> d z", d=head_dim), input)
    pad_input = pad_input.reshape(head_dim, -1).contiguous()
    return pad_input, unpad_indices


class AttnFuncTERingAttnWithKVP2P(torch.autograd.Function):
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
    ):
        attn_bias_type = "no_bias"
        attn_bias = None
        qkv_format = "thd"

        # pylint: disable=missing-function-docstring
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_group_a2a = None
        cp_size_a2a = 1
        rank_a2a = 0

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)
        send_dst = cp_global_ranks[(rank + 1) % cp_size * cp_size_a2a + rank_a2a]
        recv_src = cp_global_ranks[(rank - 1) % cp_size * cp_size_a2a + rank_a2a]
        batch_p2p_comm = int(os.getenv("NVTE_BATCH_MHA_P2P_COMM", "0")) or (
            cp_size == 2
        )

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        seq_dim = 0
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        pad_between_seqs_q = not torch.equal(cu_seqlens_q_padded, cu_seqlens_q)
        pad_between_seqs_kv = not torch.equal(cu_seqlens_kv_padded, cu_seqlens_kv)
        max_seqlen_q = max_seqlen_q // cp_size
        max_seqlen_kv = max_seqlen_kv // cp_size
        cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
        cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size
        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        fused_attn_qkv_dtype = None
        fused_attn_backend = None
        qkv_dtype = q.dtype

        q_f16 = q
        if use_fused_attention:
            fp8_meta_kwargs = {}
            fused_attn_qkv_dtype = TE_DType[q.dtype]
            fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        assert qkv_format == "thd" or (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        total_tokens_kv = k.shape[0]
        # remove padded tokens at the end
        k, v = [x[: cu_seqlens_kv_padded[-1]] for x in [k, v]]

        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

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
        rng_states = [None for _ in range(cp_size)]
        # attn_biases = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]

        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
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
                        if i == 0:
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
                                softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                    softmax_lse_per_step[i],
                                    cu_seqlens_q_padded[-1],
                                    unpad_q_indices,
                                )
                        elif i <= rank:
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
                                # unpad
                                # [2, b, sk//2, np, hn] -> [2, b*sk//2, np, hn]
                                kv_inputs[i % 2] = kv_inputs[i % 2].view(
                                    2, -1, *k.shape[-2:]
                                )
                                # if _flash_attn_2_3_plus:
                                #     fa_forward_kwargs["window_size"] = (-1, -1)
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
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_q,
                                    max_seqlen_kv // 2,
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
                                softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                    softmax_lse_per_step[i],
                                    cu_seqlens_q_padded[-1],
                                    unpad_q_indices,
                                )
                        else:
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
                                    q_inputs[i % 2],
                                    kv_inputs[i % 2][0],
                                    kv_inputs[i % 2][1],
                                    cu_seqlens_q_per_step[i],
                                    cu_seqlens_kv_per_step[i],
                                    max_seqlen_q // 2,
                                    max_seqlen_kv,
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
                                fake_dtype=fused_attn_qkv_dtype,
                                fused_attention_backend=fused_attn_backend,
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
                            # [b, sq, np, hn] -> [b*sq, np, hn]
                            q_inputs[i % 2] = q.view(-1, *q.shape[-2:])
                            # [2, b, sk, np, hn] -> [2, b*sk, np, hn]
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
                                q_inputs[i % 2],
                                kv_inputs[i % 2][0],
                                kv_inputs[i % 2][1],
                                cu_seqlens_q_per_step[i],
                                cu_seqlens_kv_per_step[i],
                                max_seqlen_q,
                                max_seqlen_kv,
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
                            softmax_lse_per_step[i] = fa_varlen_lse_pad(
                                softmax_lse_per_step[i],
                                cu_seqlens_q_padded[-1],
                                unpad_q_indices,
                            )

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(
                        fwd_results_correction_done
                    )

                if use_fused_attention:
                    # [b, np, sq, 1] -> [b, np, sq]
                    softmax_lse_per_step[i - 1].squeeze_(-1)

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if i == 1:
                        out = torch.zeros_like(q).view(q.shape)
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

        kv = p2p_comm_buffers[-1]

        if not use_fused_attention:
            out = out.view(-1, *out.shape[-2:])

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
        )
        ctx.window_size = window_size
        ctx.softcap = softcap
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
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_bias_shape = None if attn_bias is None else attn_bias.shape
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.fp8 = False
        ctx.fp8_meta = None
        ctx.is_input_fp8 = False
        ctx.is_output_fp8 = False
        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        # pylint: disable=missing-function-docstring

        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)
        send_dst = ctx.cp_global_ranks[(rank - 1) % cp_size]
        recv_src = ctx.cp_global_ranks[(rank + 1) % cp_size]
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
        # (fp8_fwd_scales, fp8_fwd_scale_invs) = saved_tensors[6:8]
        cu_seqlens_q_per_step = saved_tensors[8 : 8 + cp_size]
        cu_seqlens_kv_per_step = saved_tensors[8 + cp_size : 8 + cp_size * 2]
        rng_states = saved_tensors[8 + cp_size * 2 : 8 + cp_size * 3]
        # attn_biases = saved_tensors[8 + cp_size * 3 : 8 + cp_size * 4]

        # causal = "causal" in ctx.attn_mask_type
        # padding = "padding" in ctx.attn_mask_type

        causal = (
            "causal" in ctx.attn_mask_type
            if isinstance(ctx.attn_mask_type, str)
            else ctx.attn_mask_type == AttnMaskType.CAUSAL
        )
        padding = (
            "padding" in ctx.attn_mask_type
            if isinstance(ctx.attn_mask_type, str)
            else False
        )

        # seq_dim = None
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
        attn_dbias = None
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
        amax_per_step = None
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
            if ctx.fp8 and ctx.use_fused_attention:
                fp8_meta_kwargs["amax_dp"] = amax_per_step[0][i]
                fp8_meta_kwargs["amax_dqkv"] = amax_per_step[0][i]
            # In reversed order of fwd
            if causal:
                if i == (cp_size - 1):
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
                        )  # unpad_softmax_lse = fa_varlen_lse_repad(
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
                elif i >= (cp_size - rank - 1):
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
                        # [b, 2, sq//2, np, hn] -> [b*sq, np, hn]
                        q_ = q.view(-1, *q.shape[-2:])
                        # dq_ = torch.zeros_like(q_)
                        # [2, t, np, hn] -> [2, t/2, np, hn]
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
                else:
                    if ctx.use_fused_attention:
                        # [t, np, hn] -> [t/2, np, hn]
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
                        # kv_ = kv.view(2, -1, *kv.shape[-2:])
                        dkv_ = torch.empty_like(kv_)
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
                        kv[..., 0, :, :]
                        if ctx.qkv_format in ["bshd", "sbhd"]
                        else kv[0],
                        kv[..., 1, :, :]
                        if ctx.qkv_format in ["bshd", "sbhd"]
                        else kv[1],
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
            if not (causal and i >= (cp_size - rank - 1) and i != (cp_size - 1)):
                # [2, b*sk, np, hn] -> [2, b, 2, sk//2, np, hn] if causal
                # [2, b*sk, np, hn] -> [2, b, sk, np, hn] if not causal
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

        if ctx.qkv_format == "thd":
            dkv_ = torch.empty(
                2,
                ctx.total_tokens_kv,
                *dkv.shape[-2:],
                dtype=dkv.dtype,
                device=dkv.device,
            )
            dkv_[:, : cu_seqlens_kv_padded[-1]].copy_(dkv)
            dkv_[:, cu_seqlens_kv_padded[-1] :].fill_(0)
            dkv = dkv_

        dk, dv = dkv[0], dkv[1]

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
            attn_dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rerange_tensor_before_scatter(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    cp_chunk_size = cu_seqlens[-1] // cp_size
    restore_tensor = torch.zeros_like(x)

    for rank in range(cp_size):
        start = rank * cp_chunk_size
        end = (rank + 1) * cp_chunk_size
        pad_tensor_and_split(
            x,
            cu_seqlens,
            cu_seqlens,
            cp_size,
            rank,
            padded_tensor=restore_tensor[start:end],
        )

    return restore_tensor


def pad_tensor(
    x: torch.Tensor, cu_seqlens: torch.Tensor, cu_seqlens_padded: torch.Tensor
) -> torch.Tensor:
    """pad tensor complying with cu_seqlens to cu_seqlens_padded

    Args:
        x (torch.Tensor): tensor without padding, shape of (total_s, ...)
        cu_seqlens (list[int]): sequence length before tensor padding
        cu_seqlens_padded (list[int]): sequence length after tensor padding

    Returns:
        torch.Tensor: tensor with padding, shape of (new_total_s, ...)
    """
    assert cu_seqlens.shape == cu_seqlens_padded.shape
    assert x.shape[0] == cu_seqlens[-1]

    total_seqlens_padded = cu_seqlens_padded[-1]
    padded_shape = (total_seqlens_padded,) + x.shape[1:]
    padded_tensor = torch.zeros(padded_shape, dtype=x.dtype, device=x.device)

    for i in range(len(cu_seqlens) - 1):
        pad_start = cu_seqlens_padded[i]
        pad_end = pad_start + (cu_seqlens[i + 1] - cu_seqlens[i])
        padded_tensor[pad_start:pad_end] = x[cu_seqlens[i] : cu_seqlens[i + 1]]

    return padded_tensor


def unpad_tensor(
    x: torch.Tensor, cu_seqlens: torch.Tensor, cu_seqlens_padded: torch.Tensor
) -> torch.Tensor:
    """unpad tensor complying with cu_seqlens_padded to cu_seqlens

    Args:
        x (torch.Tensor): tensor with padding, shape of (total_s, ...)
        cu_seqlens (list[int]): sequence length before tensor padding
        cu_seqlens_padded (list[int]): sequence length after tensor padding

    Returns:
        torch.Tensor: tensor without padding, shape of (new_total_s, ...)
    """
    assert cu_seqlens.shape == cu_seqlens_padded.shape
    assert x.shape[0] == cu_seqlens_padded[-1]

    total_seqlens_unpad = cu_seqlens[-1]
    unpad_shape = (total_seqlens_unpad,) + x.shape[1:]
    unpad_tensor = torch.zeros(unpad_shape, dtype=x.dtype, device=x.device)

    for i in range(len(cu_seqlens) - 1):
        pad_start = cu_seqlens_padded[i]
        pad_end = pad_start + (cu_seqlens[i + 1] - cu_seqlens[i])
        unpad_tensor[cu_seqlens[i] : cu_seqlens[i + 1]] = x[pad_start:pad_end]

    return unpad_tensor


def create_cu_seqlens_causal(
    cu_seqlens: torch.Tensor, cu_seqlens_padded: torch.Tensor, rank: int, cp_size: int
) -> torch.Tensor:
    """_summary_

    Args:
        cu_seqlens (torch.Tensor): _description_
        cu_seqlens_padded (torch.Tensor): _description_
        rank (int): _description_
        cp_size (int): _description_

    Returns:
        torch.Tensor: _description_
    """
    assert cu_seqlens.shape == cu_seqlens_padded.shape

    cu_seqlens_padded = cu_seqlens_padded // (2 * cp_size)
    seqlens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    seqlens_unpad = cu_seqlens[1:] - cu_seqlens[:-1]
    causal_seqlens = seqlens_padded * (rank + 1)
    seqlens_unpad = torch.min(seqlens_unpad, causal_seqlens)
    cu_seqlens_causal = torch.zeros_like(cu_seqlens)
    cu_seqlens_causal[1:].add_(seqlens_unpad)
    cu_seqlens_causal.cumsum_(dim=0)
    return cu_seqlens_causal


def thd_store_half_tensor(
    x: torch.Tensor, out: torch.Tensor, cu_seqlens: torch.Tensor, index: int
) -> torch.Tensor:
    cu_seqlens_half = cu_seqlens // 2
    seqlens_half = cu_seqlens_half[1:] - cu_seqlens_half[:-1]
    for i in range(len(seqlens_half)):
        out_start = cu_seqlens[i] + index * seqlens_half[i]
        out_end = out_start + seqlens_half[i]
        out[out_start:out_end] = x[cu_seqlens_half[i] : cu_seqlens_half[i + 1]]
    return out


class AttnFuncTERingAttnWithKVAG(torch.autograd.Function):
    """
    Attention implementation with context parallelism. KV all-gather between CP ranks is exposed.
    Refer section 3.3.2 of `The Llama 3 Herd of Models <https://arxiv.org/abs/2407.21783>`_.
    """

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
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        cp_group,
        cp_stream,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type
        assert padding, f"{attn_mask_type} mask type is not supported!"
        # if use_fused_attention and causal and "bottom_right" not in attn_mask_type:
        #     attn_mask_type = attn_mask_type + "_bottom_right"
        assert (
            attn_bias_type == "no_bias"
        ), f"{attn_bias_type} bias type is not supported!"
        assert (
            attn_bias is None
        ), "attn_bias must be None when attn_bias_type is no_bias!"
        assert (
            q.shape[-1] % 8 == 0
        ), "Hidden size per attention head should be multiple of 8!"
        flash_attn_fwd = None
        softcap = 0.0
        window_size = (-1, -1)
        if not use_fused_attention:
            fa_forward_kwargs = {"softmax_scale": softmax_scale}
            # do not use flash attn v3
            flash_attn_fwd = flash_attn_varlen_fwd
            fa_forward_kwargs["dropout_p"] = dropout_p
            fa_forward_kwargs["return_softmax"] = False
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

        seq_dim = None
        if qkv_format in ["bshd", "sbhd"]:
            seq_dim = qkv_format.index("s")
            qkv_layout = qkv_format + qkv_format[:-2] + "2" + qkv_format[-2:]
        else:
            qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        pad_between_seqs_q = not torch.equal(cu_seqlens_q_padded, cu_seqlens_q)
        # pad_between_seqs_kv = not torch.equal(cu_seqlens_kv_padded, cu_seqlens_kv)
        cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size

        assert qkv_format == "thd" or (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        k_ag = all_gather_fwd_scatter_bwd(k, cp_group, dim=0).contiguous()
        v_ag = all_gather_fwd_scatter_bwd(v, cp_group, dim=0).contiguous()
        k_ag = k_ag.view((cp_size, *k.shape))
        v_ag = v_ag.view((cp_size, *v.shape))
        # pack tensor after all_gather
        k_ag = unpad_tensor_after_gather(
            k_ag, cu_seqlens_kv_padded, cu_seqlens_kv_padded, cp_size
        )
        v_ag = unpad_tensor_after_gather(
            v_ag, cu_seqlens_kv_padded, cu_seqlens_kv_padded, cp_size
        )
        cp_stream.wait_stream(torch.cuda.current_stream())

        # handle tensor shape with qkv
        if qkv_format in ["bshd", "sbhd"]:
            pass

        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]

        local_seq_chunk_idx = [rank, 2 * cp_size - rank - 1]
        # window_size_per_step = [None, None]
        cu_seqlens_q_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        max_seqlen_q_per_step = [None, None]
        max_seqlen_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        rng_states = [None, None]
        out = torch.empty_like(q)

        for i in range(len(local_seq_chunk_idx) + 1):
            if i < len(local_seq_chunk_idx):
                with torch.cuda.stream(flash_attn_streams[i]):
                    q_ = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, i)
                    if i == 0:
                        if pad_between_seqs_q:
                            cu_seqlens_q_per_step[i] = get_cu_seqlens_on_cp_rank(
                                cu_seqlens_q,
                                cu_seqlens_q_padded,
                                cp_size,
                                rank,
                                True,
                                False,
                            )
                        else:
                            cu_seqlens_q_per_step[i] = cu_seqlens_q // (2 * cp_size)
                    elif i == 1:
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
                            cu_seqlens_q_per_step[i] = cu_seqlens_q // (2 * cp_size)
                    if causal:
                        cu_seqlens_kv_per_step[i] = create_cu_seqlens_causal(
                            cu_seqlens_kv,
                            cu_seqlens_kv_padded,
                            local_seq_chunk_idx[i],
                            cp_size,
                        )
                        # window_size_per_step[i] = (-1, 0)
                    else:
                        cu_seqlens_kv_per_step[i] = cu_seqlens_kv
                        # window_size_per_step[i] = (-1, -1)
                    max_seqlen_q_per_step[i] = get_max_seqlen(cu_seqlens_q_per_step[i])
                    max_seqlen_kv_per_step[i] = get_max_seqlen(
                        cu_seqlens_kv_per_step[i]
                    )
                    if use_fused_attention:
                        out_per_step[i], [
                            softmax_lse_per_step[i],
                            rng_states[i],
                        ] = fused_attn_fwd(
                            is_training,
                            max_seqlen_q_per_step[i],
                            max_seqlen_kv_per_step[i],
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_ag,
                            v_ag,
                            TE_DType[q.dtype],
                            FusedAttnBackend["F16_arbitrary_seqlen"],
                            attn_scale=softmax_scale,
                            dropout=dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=attn_mask_type,
                            attn_bias_type=attn_bias_type,
                            attn_bias=attn_bias,
                            cu_seqlens_q_padded=cu_seqlens_q_padded // 2,
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            # window_size=window_size_per_step[i],
                        )
                    else:
                        q_ = unpad_tensor(
                            q_, cu_seqlens_q_per_step[i], cu_seqlens_q_padded // 2
                        )
                        k_ = unpad_tensor(
                            k_ag, cu_seqlens_kv_per_step[i], cu_seqlens_kv_padded
                        )
                        v_ = unpad_tensor(
                            v_ag, cu_seqlens_kv_per_step[i], cu_seqlens_kv_padded
                        )
                        fa_outputs = flash_attn_fwd(
                            q_,
                            k_,
                            v_,
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_kv_per_step[i],
                            max_seqlen_q_per_step[i]
                            if max_seqlen_q_per_step[i] > 0
                            else 1,
                            max_seqlen_kv_per_step[i]
                            if max_seqlen_kv_per_step[i] > 0
                            else 1,
                            causal=causal,
                            # window_size=window_size_per_step[i],
                            **fa_forward_kwargs,
                        )
                        out_per_step[i] = pad_tensor(
                            fa_outputs[0],
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_q_padded // 2,
                        )
                        softmax_lse_per_step[i] = fa_outputs[1]
                        rng_states[i] = fa_outputs[3]
            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if qkv_format in ["bshd", "sbhd"]:
                        pass
                    elif qkv_format == "thd":
                        thd_store_half_tensor(
                            out_per_step[i - 1], out, cu_seqlens_q_padded, i - 1
                        )

        torch.cuda.current_stream().wait_stream(cp_stream)

        out = out.view(-1, *out.shape[-2:])

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *out_per_step,
            *softmax_lse_per_step,
            *rng_states,
        )
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.max_seqlen_q_per_step = max_seqlen_q_per_step
        ctx.max_seqlen_kv_per_step = max_seqlen_kv_per_step
        # ctx.window_size_per_step = window_size_per_step
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.use_fused_attention = use_fused_attention
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q_padded, cu_seqlens_kv_padded) = saved_tensors[:5]
        cu_seqlens_q_per_step = saved_tensors[5:7]
        cu_seqlens_kv_per_step = saved_tensors[7:9]
        out_per_step = saved_tensors[9:11]
        softmax_lse_per_step = saved_tensors[11:13]
        rng_states = saved_tensors[13:15]
        # window_size_per_step = ctx.window_size_per_step
        qkv_format = ctx.qkv_format
        max_seqlen_q_per_step = ctx.max_seqlen_q_per_step
        max_seqlen_kv_per_step = ctx.max_seqlen_kv_per_step

        # seq_dim = None
        if qkv_format in ["bshd", "sbhd"]:
            # seq_dim = qkv_format.index("s")
            qkv_layout = qkv_format + qkv_format[:-2] + "2" + qkv_format[-2:]
        else:
            qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        dout = dout.view(q.shape)
        dq = torch.zeros_like(q)
        dk = torch.zeros(
            (k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device
        )
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()
        # initialize chunk size
        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        k_ag = all_gather_fwd_scatter_bwd(k, ctx.cp_group, dim=0).contiguous()
        v_ag = all_gather_fwd_scatter_bwd(v, ctx.cp_group, dim=0).contiguous()
        k_ag = k_ag.view((cp_size, *k.shape))
        v_ag = v_ag.view((cp_size, *v.shape))
        # pack tensor after all_gather
        k_ag = unpad_tensor_after_gather(
            k_ag, cu_seqlens_kv_padded, cu_seqlens_kv_padded, cp_size
        )
        v_ag = unpad_tensor_after_gather(
            v_ag, cu_seqlens_kv_padded, cu_seqlens_kv_padded, cp_size
        )
        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        flash_attn_bwd = None
        if not ctx.use_fused_attention:
            fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
            # not use flash attention 3
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

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.cuda.stream(flash_attn_streams[i]):
                    q_ = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, i)
                    out_ = out_per_step[i]
                    dout_ = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, i)
                    if ctx.use_fused_attention:
                        aux_ctx_tensors = [softmax_lse_per_step[i], rng_states[i]]
                        (
                            dq_per_step[i],
                            dk_per_step[i],
                            dv_per_step[i],
                            _,
                        ) = fused_attn_bwd(
                            max_seqlen_q_per_step[i],
                            max_seqlen_kv_per_step[i],
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_kv_per_step[i],
                            q_,
                            k_ag,
                            v_ag,
                            out_,
                            dout_,
                            TE_DType[q_.dtype],
                            TE_DType[dout_.dtype],
                            aux_ctx_tensors,
                            FusedAttnBackend["F16_arbitrary_seqlen"],
                            cu_seqlens_q_padded=cu_seqlens_q_padded // 2,
                            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                            attn_scale=ctx.softmax_scale,
                            dropout=ctx.dropout_p,
                            qkv_layout=qkv_layout,
                            attn_mask_type=ctx.attn_mask_type,
                            attn_bias_type=ctx.attn_bias_type,
                            # window_size=window_size_per_step[i],
                            deterministic=ctx.deterministic,
                        )
                    else:
                        q_ = unpad_tensor(
                            q_, cu_seqlens_q_per_step[i], cu_seqlens_q_padded // 2
                        )
                        dout_ = unpad_tensor(
                            dout_, cu_seqlens_q_per_step[i], cu_seqlens_q_padded // 2
                        )
                        out_ = unpad_tensor(
                            out_, cu_seqlens_q_per_step[i], cu_seqlens_q_padded // 2
                        )
                        k_ = unpad_tensor(
                            k_ag, cu_seqlens_kv_per_step[i], cu_seqlens_kv_padded
                        )
                        v_ = unpad_tensor(
                            v_ag, cu_seqlens_kv_per_step[i], cu_seqlens_kv_padded
                        )
                        dq_per_step[i], dk_per_step[i], dv_per_step[i] = [
                            torch.zeros_like(x) for x in [q_, k_, v_]
                        ]
                        # not use flash attn 3
                        fa_backward_kwargs["rng_state"] = rng_states[i]
                        flash_attn_bwd(
                            dout_,
                            q_,
                            k_,
                            v_,
                            out_,
                            softmax_lse_per_step[i],
                            dq_per_step[i],
                            dk_per_step[i],
                            dv_per_step[i],
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_kv_per_step[i],
                            max_seqlen_q_per_step[i],
                            max_seqlen_kv_per_step[i],
                            causal=ctx.causal,
                            # window_size=window_size_per_step[i],
                            **fa_backward_kwargs,
                        )
                        dq_per_step[i] = pad_tensor(
                            dq_per_step[i],
                            cu_seqlens_q_per_step[i],
                            cu_seqlens_q_padded // 2,
                        )
                        dk_per_step[i] = pad_tensor(
                            dk_per_step[i],
                            cu_seqlens_kv_per_step[i],
                            cu_seqlens_kv_padded,
                        )
                        dv_per_step[i] = pad_tensor(
                            dv_per_step[i],
                            cu_seqlens_kv_per_step[i],
                            cu_seqlens_kv_padded,
                        )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    thd_store_half_tensor(
                        dq_per_step[i - 1], dq, cu_seqlens_q_padded, i - 1
                    )
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    # print(f"{dk.shape} {dk_per_step[i - 1].shape}")
                    dk.add_(dk_per_step[i - 1])
                    dv.add_(dv_per_step[i - 1])
                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.cuda.current_stream().wait_stream(ctx.cp_stream)

        # dk dv is shape of (cp*t, h, d)
        dk = rerange_tensor_before_scatter(dk, cu_seqlens_kv_padded, cp_size)
        dv = rerange_tensor_before_scatter(dv, cu_seqlens_kv_padded, cp_size)
        dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

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
