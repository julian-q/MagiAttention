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

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_pkg_version
from typing import Any, Dict, Optional

# deepspeed ulysess import
import deepspeed.comm as ds_dist
import torch
import torch.distributed as dist
import transformer_engine  # noqa
import transformer_engine_torch as tex
from deepspeed.accelerator import get_accelerator
from packaging.version import Version as PkgVersion
from transformer_engine.pytorch.attention import (
    _get_supported_versions,
    fa_logger,
    flash_attn_a2a_communicate,
    get_seq_chunk_ids_for_reordering,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

from .interface import AttnBaselineInterface
from .teusp_utils import (
    _get_unpad_data,
    fa_thd_pad,
    fa_thd_unpad,
    fa_varlen_lse_pad,
    fa_varlen_lse_unpad,
    get_distributed_rank,
    get_distributed_world_size,
)

_NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))
_NVTE_FUSED_ATTN = int(os.getenv("NVTE_FUSED_ATTN", "1"))
_NVTE_UNFUSED_ATTN = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))

# Detect flash-attn v2 in the environment
_flash_attn_version = PkgVersion("0")
_flash_attn_version_required = PkgVersion("2.1.1")
_flash_attn_max_version = PkgVersion("2.7.3")
_flash_attn_2_3_plus = False
_flash_attn_2_4_plus = False
_flash_attn_2_4_1_plus = False
_flash_attn_2_5_7_plus = False
_flash_attn_2_6_0_plus = False
_flash_attn_2_7_0_plus = False

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
    parameters to TEAttnFuncWithCPAndQKVOA2A and dispatch for the
    `thd` (packed) sequence format
    """

    indices: torch.Tensor = None
    cu_seqlens: torch.Tensor = None
    cu_seqlens_padded: torch.Tensor = None
    max_seqlen_in_batch: int = 0
    max_seqlen_in_padded: int = 0
    first_axis_dim: int = 0


def post_all2all(
    scatter_idx, batch_dim_idx, seq_world_size, bs, seq_len, num_head, head_dim
):
    def post_func(input):
        if batch_dim_idx == 0:
            # b, s, n, h
            if scatter_idx < 2:
                output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(
                    bs, seq_len // seq_world_size, seq_world_size * num_head, head_dim
                ).contiguous()
            else:
                output = input.permute(1, 0, 2, 3, 4).contiguous()
                output = output.reshape(
                    bs, seq_world_size * seq_len, num_head // seq_world_size, head_dim
                ).contiguous()
        else:
            # s, b, n, h
            if scatter_idx < 2:
                output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(
                    seq_len // seq_world_size, bs, seq_world_size * num_head, head_dim
                ).contiguous()
            else:
                output = input.reshape(
                    seq_len * seq_world_size, bs, num_head // seq_world_size, head_dim
                ).contiguous()
        return output

    return post_func


def single_all_to_all(
    input,
    scatter_idx,
    gather_idx,
    batch_dim_idx,
    group,
    async_op=False,
    handle=None,
    type=None,
):
    seq_world_size = ds_dist.get_world_size(group)
    # we only need num_heads once

    if batch_dim_idx == 0:
        # b, s, n, h
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [
                    bs,
                    seq_world_size,
                    global_seq_len // seq_world_size,
                    num_local_head,
                    head_dim,
                ]
            ).contiguous()
            input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [
                    bs,
                    local_seq_len,
                    seq_world_size,
                    num_total_head // seq_world_size,
                    head_dim,
                ]
            ).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    else:
        # s, b, n, h
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [
                    seq_world_size,
                    global_seq_len // seq_world_size,
                    bs,
                    num_local_head,
                    head_dim,
                ]
            ).contiguous()
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [
                    local_seq_len,
                    bs,
                    seq_world_size,
                    num_total_head // seq_world_size,
                    head_dim,
                ]
            ).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

    if scatter_idx < 2:
        post_all2all_fun = post_all2all(
            scatter_idx,
            batch_dim_idx,
            seq_world_size,
            bs,
            global_seq_len,
            num_local_head,
            head_dim,
        )
    else:
        post_all2all_fun = post_all2all(
            scatter_idx,
            batch_dim_idx,
            seq_world_size,
            bs,
            local_seq_len,
            num_total_head,
            head_dim,
        )

    output = torch.empty_like(input_t)
    work = ds_dist.all_to_all_single(output, input_t, group=group, async_op=False)
    ds_dist.barrier(group=group)

    if async_op:
        if type in ("dq", "dk"):
            handle[type + "_work"] = work
            handle[type + "_grad"] = output
            handle[type + "_post_all2all_func"] = post_all2all_fun
            return output

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: ds_dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        batch_dim_idx: int,
        stream=None,
        handle=None,
        type=None,
        is_fwd=True,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.stream = stream
        ctx.handle = handle
        ctx.type = type
        ctx.batch_dim_idx = batch_dim_idx
        if ctx.handle is None:
            res = single_all_to_all(
                input, scatter_idx, gather_idx, batch_dim_idx, group, False
            )

        else:
            # overlap communication path
            if not is_fwd and type == "o":
                assert ctx.stream is not None
                res = single_all_to_all(
                    input, scatter_idx, gather_idx, batch_dim_idx, group, False
                )
                get_accelerator().current_stream().wait_stream(ctx.stream)
                del ctx.stream.activation_buffer_list
                # The computation of d o_weight can overlap with the communication of d o_input

            elif not is_fwd and type in ("q", "k"):
                # Achieve communication overlap by pipelining the matrix computation and communication of dq, dk, and dv
                type = "d" + type
                res = single_all_to_all(
                    input,
                    scatter_idx,
                    gather_idx,
                    batch_dim_idx,
                    group,
                    True,
                    handle,
                    type,
                )

            elif is_fwd and type in ("q", "k"):
                # Achieve communication overlap by pipelining the matrix computation and communication of q, k, and v
                type = "fwd_" + type
                res = single_all_to_all(
                    input,
                    scatter_idx,
                    gather_idx,
                    batch_dim_idx,
                    group,
                    False,
                    handle,
                    type,
                )

            else:
                res = single_all_to_all(
                    input, scatter_idx, gather_idx, batch_dim_idx, group, False
                )

        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor):
        # res = single_all_to_all(
        #     *grad_output,
        #     ctx.gather_idx,
        #     ctx.scatter_idx,
        #     ctx.batch_dim_idx,
        #     ctx.group,
        #     False,
        # )
        # return (
        #     None,
        #     res,
        #     None,
        #     None,
        #     None,
        #     None,
        #     None,
        #     None,
        #     None,
        # )
        return (
            None,
            _SeqAllToAll.apply(
                ctx.group,
                *grad_output,
                ctx.gather_idx,
                ctx.scatter_idx,
                ctx.batch_dim_idx,
                ctx.stream,
                ctx.handle,
                ctx.type,
                False,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DSUlysses(AttnBaselineInterface):
    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
        sp_stream=None,
    ):
        super().__init__()
        self.spg = cp_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.sp_overlap_comm = False
        self.sp_stream = sp_stream
        self.overlap_handles: Optional[Dict] = None
        if sp_stream is not None:
            self.overlap_handles = {}
            self.sp_overlap_comm = True
            self.dafult_stream = get_accelerator().default_stream()

        self.packed_seq_params: dict[str, PackedSeqParams] = {}

    def layer_sync(self, layer):
        if self.sp_overlap_comm and hasattr(layer, "done_event"):
            self.dafult_stream.wait_event(layer.done_event)

    def dispatch(
        # self,
        # x_global: torch.Tensor,
        # cp_rank: int,
        # cp_size: int,
        # cp_group: dist.ProcessGroup,
        # ranges: AttnRanges,
        # attention_mask_thd: torch.Tensor,
        # **kwargs,
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
            x_global (torch.Tensor): the global tensor to be dispatched, with shape [s, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the dispatched local tensor
        """
        print(f"Extra kwargs received: {kwargs}")
        assert isinstance(
            cp_group, dist.ProcessGroup
        ), "Unsupported process group for CP communication group!"
        # assert (
        #     get_distributed_world_size(cp_group) > 1
        # ), "CP group size should be greater than 1 for dispatching!"
        qkv_format = kwargs.get("qkv_format", "thd")
        qkv_ = kwargs.get("qkv_", "q")
        ranges = kwargs.get("ranges", None)
        attention_mask_thd = kwargs.get("attention_mask_thd", None)
        seq_dim = 0
        if qkv_format != "thd":
            seq_dim = qkv_format.index("s")

        valid_seqlen = attention_mask_thd.sum(dim=0, dtype=torch.int32).item()
        total_seqlen = len(attention_mask_thd)

        cu_seqlens = torch.tensor(
            ranges.to_cu_seqlens(seq_len=valid_seqlen),
            device=x_global.device,
            dtype=torch.int32,
        )
        assert valid_seqlen == cu_seqlens[-1], "valid_seqlen != cu_seqlens[-1]"
        assert (
            total_seqlen % cp_size == 0
        ), "total_seqlen is not divisible by cp_size or heads num!"

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        (
            indices,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        ) = _get_unpad_data(attention_mask_thd, cu_seqlens)
        self.packed_seq_params[qkv_] = PackedSeqParams(
            indices,
            cu_seqlens,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        )

        rank = get_distributed_rank(cp_group)
        # t,h,d -> s,b,h,d
        if qkv_format != "thd":
            bsz = len(cu_seqlens) - 1
            assert (
                total_seqlen % bsz == 0 and (total_seqlen // bsz) % cp_size == 0
            ), "total_seqlen is not divisible by bsz or bsz is not divisible by cp_size"
            seq_len = total_seqlen // bsz
            other_shape = x_global.shape[1:]
        if qkv_format == "sbhd":
            input = (
                x_global.view(bsz, seq_len, *other_shape)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
        elif qkv_format == "bshd":
            input = x_global.view(bsz, seq_len, *other_shape).contiguous()
        else:
            input = x_global

        x_local = input.chunk(cp_size, dim=seq_dim)[rank].contiguous()

        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        cp_group: dist.ProcessGroup,
        **kwargs,
    ) -> torch.Tensor:
        """
        Undispatch the local tensor `x_local` along its sequence dim following the meta info,
        and return the undispatched global tensor `x_global`

        Args:
            x_local (torch.Tensor): the local tensor to be undispatched, with shape [s, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the undispatched global tensor
        """
        print(f"Extra kwargs received: {kwargs}")
        assert isinstance(
            cp_group, dist.ProcessGroup
        ), "Unsupported process group for CP communication group!"
        # assert get_distributed_world_size(cp_group)>1, f"CP group size should be greater than 1 for dispatching!"
        qkv_format = kwargs.get("qkv_format", "thd")
        other_shape = x_local.shape[1:]
        seq_dim = 0
        if qkv_format != "thd":
            seq_dim = qkv_format.index("s")
            other_shape = x_local.shape[2:]

        # ulysess all gather
        local_u_group = [torch.empty_like(x_local) for _ in range(cp_size)]
        ds_dist.all_gather(local_u_group, x_local, group=cp_group)
        x_global = torch.cat(local_u_group, dim=seq_dim)

        # b,s,h,d -> t,h,d
        if qkv_format == "bshd":
            x_global = x_global.view(-1, *other_shape).contiguous()
        elif qkv_format == "sbhd":
            x_global = (
                x_global.permute(1, 0, 2, 3)
                .contiguous()
                .view(-1, *other_shape)
                .contiguous()
            )

        return x_global

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
        # # cp_size: int,
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

        def bwd_hook(layer_type):
            def pre_hook_fun(grad):
                type = "d" + layer_type
                self.overlap_handles[type + "_work"].wait()
                self.sp_stream.wait_stream(self.dafult_stream)
                all2all_output = self.overlap_handles[type + "_grad"]
                grad = list(grad)
                grad[0] = self.overlap_handles[type + "_post_all2all_func"](
                    all2all_output
                )
                grad = tuple(grad)

            return pre_hook_fun

        print(f"Extra kwargs received: {kwargs}")
        cp_group = kwargs.get("cp_group", None)
        assert isinstance(
            cp_group, dist.ProcessGroup
        ), "Unsupported process group for CP communication group!"
        # assert get_distributed_world_size(cp_group)>1, f"CP group size should be greater than 1 for dispatching!"
        qkv_format = kwargs.get("qkv_format", "thd")
        dropout_p = kwargs.get("dropout_p", 0.0)
        window_size = (-1, -1)

        assert isinstance(
            attn_mask_type, AttnMaskType
        ), "attn_mask_type must be an AttnMaskType!"
        if attn_mask_type == AttnMaskType.CAUSAL:
            teulysess_attn_mask_type = "padding_causal"
            # window_size = (-1, 0)
        else:
            teulysess_attn_mask_type = "padding"

        batch_dim_idx = 0
        if qkv_format == "thd":
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        if qkv_format == "sbhd":
            batch_dim_idx = 1

        self.layer_sync(q)
        query_layer = _SeqAllToAll.apply(
            self.spg,
            q,
            self.scatter_idx,
            self.gather_idx,
            batch_dim_idx,
            None,
            self.overlap_handles,
            "q",
        )
        self.layer_sync(k)
        key_layer = _SeqAllToAll.apply(
            self.spg,
            k,
            self.scatter_idx,
            self.gather_idx,
            batch_dim_idx,
            None,
            self.overlap_handles,
            "k",
        )
        if self.sp_overlap_comm:
            self.dafult_stream.wait_stream(self.sp_stream)

        value_layer = _SeqAllToAll.apply(
            self.spg,
            v,
            self.scatter_idx,
            self.gather_idx,
            batch_dim_idx,
            None,
            self.overlap_handles,
            "v",
        )

        if qkv_format == "thd":
            query_layer = query_layer.squeeze(0)
            key_layer = key_layer.squeeze(0)
            value_layer = value_layer.squeeze(0)

        if self.sp_overlap_comm:
            # Register a hook to synchronize dq and dk after the all-to-all
            # operation when the gradient data is used.
            # Place this logic after the q, k, v all-to-all operation to
            # improve interpreter speed to
            # call and launch of the forward all-to-all communication.
            grad_fn_q = q.grad_fn.next_functions[0][0]
            grad_fn_q.register_prehook(bwd_hook(layer_type="q"))
            grad_fn_k = k.grad_fn.next_functions[0][0]
            grad_fn_k.register_prehook(bwd_hook(layer_type="k"))

        # attn
        context_layer, softmax_lse = TEAttnFuncWithCPAndQKVOA2A.apply(
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
            qkv_format,
            teulysess_attn_mask_type,
            deterministic,
            False,
            window_size,
            cp_group,
            self.sp_stream,
            self.packed_seq_params["q"],
            self.packed_seq_params["k"],
            False,
        )

        if qkv_format == "thd":
            context_layer = context_layer.unsqueeze(0)
        output = _SeqAllToAll.apply(
            self.spg,
            context_layer,
            self.gather_idx,
            self.scatter_idx,
            batch_dim_idx,
            self.sp_stream,
            self.overlap_handles,
            "o",
        )
        if qkv_format == "thd":
            output = output.squeeze(0)

        return output, softmax_lse


class TEAttnFuncWithCPAndQKVOA2A(torch.autograd.Function):
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
        deterministic,
        use_fused_attention,
        window_size,
        cp_group,
        cp_stream,
        packed_seq_params_q,
        packed_seq_params_kv,
        enable_a2a=True,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)

        causal = "causal" in attn_mask_type
        # padding = "padding" in attn_mask_type
        attn_bias_type = "no_bias"
        attn_bias = None
        assert (
            q.shape[-1] % 8 == 0
        ), "Hidden size per attention head should be multiple of 8!"
        assert (
            window_size == (-1, 0)
            or window_size == (-1, -1)
            or use_fused_attention
            or _flash_attn_2_3_plus
        ), "Sliding window attention only can work with FusedAttention or FlashAttention >= 2.3!"

        softcap = 0.0
        flash_attn_fwd = None
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

        assert (
            q.shape[-2] % cp_size == 0 and k.shape[-2] % cp_size == 0
        ), "The number of attention heads needs to be divisible by CP size!"

        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        if qkv_format == "thd":
            # t,h,d -> 1,t,h,d
            seq_dim = 1
            batch_dim = 0
            batch_size = 1
        else:
            batch_dim = qkv_format.index("b")
            seq_dim = qkv_format.index("s")
            batch_size = q.shape[batch_dim]
        assert (
            q.shape[seq_dim] % 2 == 0 and k.shape[seq_dim] % 2 == 0
        ), "Sequence length per GPU needs to be divisible by 2!"

        # qkv_dtype = q.dtype
        fused_attn_backend = None
        fused_attn_qkv_dtype = None

        if use_fused_attention:
            fp8_meta_kwargs = {}
            fused_attn_qkv_dtype = TE_DType[q.dtype]
            fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if enable_a2a:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(
                cp_size, q.device, True
            )
            if qkv_format == "thd":
                _q, _k, _v = flash_attn_a2a_communicate(
                    [q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)],
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    cp_group,
                    cp_stream,
                    True,
                )
                q, k, v = _q.squeeze(0), _k.squeeze(0), _v.squeeze(0)
            else:
                q, k, v = flash_attn_a2a_communicate(
                    [q, k, v],
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    cp_group,
                    cp_stream,
                    True,
                )

        if use_fused_attention:
            out, aux_ctx_tensors = fused_attn_fwd(
                is_training,
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q,
                k,
                v,
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
                # window_size=window_size,
                **fp8_meta_kwargs,
            )
            softmax_lse, *rest = aux_ctx_tensors
            softmax_lse = softmax_lse.squeeze(-1)
        else:
            # unpad + reshape
            unpad_q = fa_thd_unpad(q, packed_seq_params_q.indices, qkv_format)
            unpad_k = fa_thd_unpad(k, packed_seq_params_kv.indices, qkv_format)
            unpad_v = fa_thd_unpad(v, packed_seq_params_kv.indices, qkv_format)
            # [b*cp*s, np//cp, hn] -> [b, cp*s, np//cp, hn]
            fa_outputs = flash_attn_fwd(
                unpad_q,
                unpad_k,
                unpad_v,
                cu_seqlens_q,
                cu_seqlens_kv,
                packed_seq_params_q.max_seqlen_in_batch,
                packed_seq_params_kv.max_seqlen_in_batch,
                causal=causal,
                **fa_forward_kwargs,
            )
            # out, softmax_lse = fa_outputs[4], fa_outputs[5]
            out, softmax_lse = fa_outputs[0], fa_outputs[1]
            # [b*cp*s, np//cp, hn] -> [b, cp*s, np//cp, hn]
            # out = out.view(batch_size, -1, *out.shape[-2:])
            # pad + reshape
            out = fa_thd_pad(
                out, packed_seq_params_q.indices, cu_seqlens_q_padded, qkv_format
            )

            # FIXME: softmax lse now is onle 2-dim tensor!!
            # softmax_lse = fa_varlen_lse_repad(softmax_lse, max_seqlen_q)
            softmax_lse = fa_varlen_lse_pad(
                softmax_lse, cu_seqlens_q_padded[-1], packed_seq_params_q.indices
            )

            rng_state = fa_outputs[3]
            aux_ctx_tensors = [softmax_lse, rng_state]

        if enable_a2a:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(
                cp_size, out.device, False
            )
            if qkv_format == "thd":
                _out = flash_attn_a2a_communicate(
                    out.unsqueeze(0),
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    cp_group,
                    cp_stream,
                    False,
                )
                out = _out.squeeze(0)
            else:
                out = flash_attn_a2a_communicate(
                    out, chunk_ids_for_a2a, seq_dim, cp_size, cp_group, cp_stream, False
                )

        if qkv_format == "bshd":
            # [b*s, np, hn] -> [b, s, np, hn]
            out = out.view(batch_size, -1, *out.shape[-2:])
        elif qkv_format == "sbhd":
            # [s*b, np, hn] -> [s, b, np, hn]
            out = out.view(-1, batch_size, *out.shape[-2:])

        out_ret = out
        q_save, k_save, v_save, out_save = q, k, v, out
        fp8_fwd_scales, fp8_fwd_scale_invs = None, None

        ctx.save_for_backward(
            q_save,
            k_save,
            v_save,
            out_save,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            fp8_fwd_scales,
            fp8_fwd_scale_invs,
            *aux_ctx_tensors,
        )
        ctx.batch_size = batch_size
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.softcap = softcap
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.deterministic = deterministic
        ctx.window_size = window_size
        ctx.use_fused_attention = use_fused_attention
        ctx.fp8 = False
        ctx.fp8_meta = None
        ctx.is_input_fp8 = False
        ctx.is_output_fp8 = False

        ctx.packed_seq_params_q = packed_seq_params_q
        ctx.packed_seq_params_kv = packed_seq_params_kv
        ctx.enable_a2a = enable_a2a

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        # pylint: disable=missing-function-docstring
        cp_size = get_distributed_world_size(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        q, k, v, out = saved_tensors[:4]
        (
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        ) = saved_tensors[4:8]
        fp8_fwd_scales, fp8_fwd_scale_invs = saved_tensors[8:10]
        aux_ctx_tensors = saved_tensors[10:]

        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
        causal = "causal" in ctx.attn_mask_type
        if ctx.qkv_format == "thd":
            # t,h,d -> 1,t,h,d
            seq_dim = 1
        else:
            seq_dim = ctx.qkv_format.index("s")

        fused_attn_backend = None
        fused_attn_dqkv_dtype = None
        fused_attn_qkv_dtype = None
        # dout_dtype = dout.dtype
        if ctx.use_fused_attention:
            fp8_meta_kwargs = {}
            fused_attn_qkv_dtype = TE_DType[q.dtype]
            fused_attn_dqkv_dtype = TE_DType[dout.dtype]
            fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        # if not ctx.use_fused_attention:
        #     if ctx.qkv_format != "thd":
        #         out = out.view(ctx.batch_size, -1, *out.shape[-2:])
        dout = dout.view(*out.shape)

        if ctx.enable_a2a:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(
                cp_size, out.device, True
            )
            if ctx.qkv_format == "thd":
                _out, _dout = flash_attn_a2a_communicate(
                    [out.unsqueeze(0), dout.unsqueeze(0)],
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    ctx.cp_group,
                    ctx.cp_stream,
                    True,
                )
                out, dout = _out.squeeze(0), _dout.squeeze(0)
            else:
                out, dout = flash_attn_a2a_communicate(
                    [out, dout],
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    ctx.cp_group,
                    ctx.cp_stream,
                    True,
                )

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

        if ctx.use_fused_attention:
            dq, dk, dv, _ = fused_attn_bwd(
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q,
                k,
                v,
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
                # window_size=ctx.window_size,
                deterministic=ctx.deterministic,
                **fp8_meta_kwargs,
            )
        else:
            # unpad + reshape
            softmax_lse, rng_state = aux_ctx_tensors
            unpad_q = fa_thd_unpad(q, ctx.packed_seq_params_q.indices, ctx.qkv_format)
            unpad_k = fa_thd_unpad(k, ctx.packed_seq_params_kv.indices, ctx.qkv_format)
            unpad_v = fa_thd_unpad(v, ctx.packed_seq_params_kv.indices, ctx.qkv_format)
            unpad_dout = fa_thd_unpad(
                dout, ctx.packed_seq_params_q.indices, ctx.qkv_format
            )
            unpad_out = fa_thd_unpad(
                out, ctx.packed_seq_params_q.indices, ctx.qkv_format
            )
            # unpad_softmax_lse = fa_varlen_lse_repad(
            #     softmax_lse, ctx.packed_seq_params_q.max_seqlen_in_batch
            # )
            unpad_softmax_lse = fa_varlen_lse_unpad(
                softmax_lse, ctx.packed_seq_params_q.indices
            )
            fa_backward_kwargs["rng_state"] = rng_state
            # out, dout = [x.view(-1, *x.shape[-2:]) for x in [out, dout]]
            dq, dk, dv = [torch.empty_like(x) for x in [unpad_q, unpad_k, unpad_v]]

            flash_attn_bwd(
                unpad_dout,
                unpad_q,
                unpad_k,
                unpad_v,
                unpad_out,
                unpad_softmax_lse,
                dq,
                dk,
                dv,
                cu_seqlens_q,
                cu_seqlens_kv,
                ctx.packed_seq_params_q.max_seqlen_in_batch,
                ctx.packed_seq_params_kv.max_seqlen_in_batch,
                causal=causal,
                **fa_backward_kwargs,
            )
            # dq, dk, dv = [x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]]
            # pad + reshape
            dq = fa_thd_pad(
                dq, ctx.packed_seq_params_q.indices, cu_seqlens_q_padded, ctx.qkv_format
            )
            dk = fa_thd_pad(
                dk,
                ctx.packed_seq_params_kv.indices,
                cu_seqlens_kv_padded,
                ctx.qkv_format,
            )
            dv = fa_thd_pad(
                dv,
                ctx.packed_seq_params_kv.indices,
                cu_seqlens_kv_padded,
                ctx.qkv_format,
            )
        if ctx.enable_a2a:
            chunk_ids_for_a2a = get_seq_chunk_ids_for_reordering(
                cp_size, q.device, False
            )
            if ctx.qkv_format == "thd":
                _dq, _dk, _dv = flash_attn_a2a_communicate(
                    [dq.unsqueeze(0), dk.unsqueeze(0), dv.unsqueeze(0)],
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    ctx.cp_group,
                    ctx.cp_stream,
                    False,
                )
                dq, dk, dv = [_dq.squeeze(0), _dk.squeeze(0), _dv.squeeze(0)]
            else:
                dq, dk, dv = flash_attn_a2a_communicate(
                    [dq, dk, dv],
                    chunk_ids_for_a2a,
                    seq_dim,
                    cp_size,
                    ctx.cp_group,
                    ctx.cp_stream,
                    False,
                )

        if ctx.qkv_format == "bshd":
            dq, dk, dv = [
                x.view(ctx.batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]
            ]
        elif ctx.qkv_format == "sbhd":
            dq, dk, dv = [
                x.view(-1, ctx.batch_size, *x.shape[-2:]) for x in [dq, dk, dv]
            ]

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
            None,
            None,
            None,
        )


class TEUlysses(AttnBaselineInterface):
    def __init__(self):
        super().__init__()

        self.packed_seq_params = {"q": None, "k": None, "v": None}

    def dispatch(
        # self,
        # x_global: torch.Tensor,
        # cp_rank: int,
        # cp_size: int,
        # cp_group: dist.ProcessGroup,
        # ranges: AttnRanges,
        # attention_mask_thd: torch.Tensor,
        # **kwargs,
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
            x_global (torch.Tensor): the global tensor to be dispatched, with shape [s, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the dispatched local tensor
        """
        print(f"Extra kwargs received: {kwargs}")
        assert isinstance(
            cp_group, dist.ProcessGroup
        ), "Unsupported process group for CP communication group!"
        # assert get_distributed_world_size(cp_group)>1, f"CP group size should be greater than 1 for dispatching!"
        qkv_format = kwargs.get("qkv_format", "thd")
        qkv_ = kwargs.get("qkv_", "q")
        ranges = kwargs.get("ranges", None)
        attention_mask_thd = kwargs.get("attention_mask_thd", None)
        seq_dim = 0
        if qkv_format != "thd":
            seq_dim = qkv_format.index("s")
        valid_seqlen = attention_mask_thd.sum(dim=0, dtype=torch.int32).item()
        total_seqlen = len(attention_mask_thd)

        cu_seqlens = torch.tensor(
            ranges.to_cu_seqlens(seq_len=valid_seqlen),
            device=x_global.device,
            dtype=torch.int32,
        )
        assert valid_seqlen == cu_seqlens[-1], "valid_seqlen != cu_seqlens[-1]"
        assert (
            total_seqlen % 2 * cp_size == 0
        ), "total_seqlen is not divisible by cp_size or heads num!"

        (
            indices,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        ) = _get_unpad_data(attention_mask_thd, cu_seqlens)
        self.packed_seq_params[qkv_] = PackedSeqParams(
            indices,
            cu_seqlens,
            cu_seqlens_padded,
            max_seqlen_in_batch,
            max_seqlen_in_padded,
        )

        rank = get_distributed_rank(cp_group)
        # t,h,d -> s,b,h,d
        if qkv_format != "thd":
            bsz = len(cu_seqlens) - 1
            assert (
                total_seqlen % bsz == 0 and (total_seqlen // bsz) % 2 * cp_size == 0
            ), "total_seqlen is not divisible by bsz or bsz is not divisible by cp_size"
            seq_len = total_seqlen // bsz
            other_shape = x_global.shape[1:]
        if qkv_format == "sbhd":
            input = (
                x_global.view(bsz, seq_len, *other_shape)
                .permute(1, 0, 2, 3)
                .contiguous()
            )
        elif qkv_format == "bshd":
            input = x_global.view(bsz, seq_len, *other_shape).contiguous()
        else:
            input = x_global

        x_chunks = input.chunk(2 * cp_size, dim=seq_dim)
        x_local = torch.cat(
            [x_chunks[rank], x_chunks[2 * cp_size - rank - 1]], dim=seq_dim
        )

        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        cp_rank: int,
        cp_size: int,
        cp_group: dist.ProcessGroup,
        **kwargs,
    ) -> torch.Tensor:
        """
        Undispatch the local tensor `x_local` along its sequence dim following the meta info,
        and return the undispatched global tensor `x_global`

        Args:
            x_local (torch.Tensor): the local tensor to be undispatched, with shape [s, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the undispatched global tensor
        """
        print(f"Extra kwargs received: {kwargs}")
        assert isinstance(
            cp_group, dist.ProcessGroup
        ), "Unsupported process group for CP communication group!"
        # assert get_distributed_world_size(cp_group)>1, f"CP group size should be greater than 1 for dispatching!"
        qkv_format = kwargs.get("qkv_format", "thd")
        if qkv_format == "bshd":
            x_local = x_local.permute(1, 0, 2, 3).contiguous()
        other_shape = x_local.shape[1:]
        # seq_dim = 0

        # ulysess all gather
        local_u_group = [torch.empty_like(x_local) for _ in range(cp_size)]
        dist.all_gather(local_u_group, x_local, group=cp_group)
        x_chunks = torch.stack(local_u_group)
        x_chunks = (
            x_chunks.view(cp_size, 2, -1, *other_shape)
            .view(2 * cp_size, -1, *other_shape)
            .contiguous()
        )

        chunk_ids = get_seq_chunk_ids_for_reordering(cp_size, x_local.device, True)
        x_global = torch.index_select(x_chunks, dim=0, index=chunk_ids)
        x_global = x_global.view(-1, *other_shape).contiguous()

        if qkv_format != "thd":
            x_global = (
                x_global.permute(1, 0, 2, 3)
                .contiguous()
                .view(-1, *other_shape[1:])
                .contiguous()
            )
        return x_global

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
        # # cp_size: int,
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
        cp_group = kwargs.get("cp_group", None)
        assert isinstance(
            cp_group, dist.ProcessGroup
        ), "Unsupported process group for CP communication group!"
        # assert get_distributed_world_size(cp_group)>1, f"CP group size should be greater than 1 for dispatching!"
        qkv_format = kwargs.get("qkv_format", "thd")
        dropout_p = kwargs.get("dropout_p", 0.0)
        use_fused_attention = kwargs.get("use_fused_attention", True)
        cp_stream = kwargs.get("cp_stream", None)

        window_size = (-1, -1)

        assert isinstance(
            attn_mask_type, AttnMaskType
        ), "attn_mask_type must be an AttnMaskType!"
        if attn_mask_type == AttnMaskType.CAUSAL:
            teulysess_attn_mask_type = "padding_causal"
            # window_size = (-1, 0)
        else:
            teulysess_attn_mask_type = "padding"

        context_layer, softmax_lse = TEAttnFuncWithCPAndQKVOA2A.apply(
            True,
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
            qkv_format,
            teulysess_attn_mask_type,
            deterministic,
            use_fused_attention,
            window_size,
            cp_group,
            cp_stream,
            self.packed_seq_params["q"],
            self.packed_seq_params["k"],
            True,
        )

        return context_layer, softmax_lse
