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

import torch
import torch.distributed as dist

from magi_attention.comm.functional import all_gather_fwd_scatter_bwd
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

# isort: split
from exps.dist_attn.baselines.te_ring_attn_utils import (
    AttnFuncTERingAttnWithKVAG,
    AttnFuncTERingAttnWithKVP2P,
    compute_cu_seqlens_padded_with_attention_mask,
    get_max_seqlen,
    pad_tensor_and_split,
    unpad_tensor_after_gather,
)

from .interface import AttnBaselineInterface


class TERingAttnWithKVP2P(AttnBaselineInterface):
    def __init__(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: AttnMaskType | list[AttnMaskType],
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        deterministic: bool,
    ):
        super().__init__()
        self.q_ranges = q_ranges
        self.k_ranges = k_ranges
        self.attn_mask_type = attn_mask_type
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.softmax_scale = softmax_scale
        self.deterministic = deterministic

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
            x_global (torch.Tensor): the global tensor to be dispatched, with shape [s, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the dispatched local tensor
        """
        attention_mask = kwargs.get("attention_mask", None)
        tensor_type = kwargs.get("tensor_type", None)
        assert tensor_type in [
            "q",
            "k",
            "v",
        ], f"tensor_type is None or {tensor_type} is not supported"

        if tensor_type == "q":
            cu_seqlens = self.q_ranges.to_cu_seqlens(self.q_ranges.total_seqlen)
        else:
            cu_seqlens = self.k_ranges.to_cu_seqlens(self.k_ranges.total_seqlen)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=x_global.device)

        cu_seqlens_valid = compute_cu_seqlens_padded_with_attention_mask(
            cu_seqlens=cu_seqlens, attention_mask=attention_mask
        )
        seqlens = cu_seqlens_valid[1:] - cu_seqlens_valid[:-1]
        seqlens = (seqlens + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size)
        cu_seqlens_padded = torch.nn.functional.pad(seqlens.cumsum_(dim=0), (1, 0)).to(
            torch.int32
        )
        x_local = pad_tensor_and_split(
            x_global,
            cu_seqlens=cu_seqlens_valid,
            cu_seqlens_padded=cu_seqlens_padded,
            cp_size=cp_size,
            rank=cp_rank,
        )
        if tensor_type == "q":
            self.cu_seqlens_q = cu_seqlens
            self.cu_seqlens_q_padded = cu_seqlens_padded
            self.cu_seqlens_q_valid = cu_seqlens_valid
            self.max_seqlen_q = get_max_seqlen(cu_seqlens_padded)
        elif tensor_type == "k":
            self.cu_seqlens_kv = cu_seqlens
            self.cu_seqlens_kv_padded = cu_seqlens_padded
            self.cu_seqlens_kv_valid = cu_seqlens_valid
            self.max_seqlen_kv = get_max_seqlen(cu_seqlens_padded)
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
        qkv_format = kwargs.get("qkv_format", None)
        tensor_type = kwargs.get("tensor_type", None)
        s_local = x_local.shape[0]
        assert s_local % 2 == 0, "Sequence length per GPU needs to be divisible by 2"
        assert (
            qkv_format == "thd" or qkv_format == "sbhd"
        ), f"qkv_format is None or {qkv_format} is not support"
        assert tensor_type in [
            "q",
            "k",
            "v",
        ], f"tensor_type is None or {tensor_type} is not supported"
        # x_global shape of (s_local * cp_size, ...)
        x_global = all_gather_fwd_scatter_bwd(x_local, cp_group, dim=0).contiguous()
        x_global = x_global.view((cp_size, *x_local.shape))
        if tensor_type == "q":
            out = unpad_tensor_after_gather(
                x_global,
                cu_seqlens=self.cu_seqlens_q_valid,
                cu_seqlens_padded=self.cu_seqlens_q_padded,
                cp_size=cp_size,
            )
        else:
            out = unpad_tensor_after_gather(
                x_global,
                cu_seqlens=self.cu_seqlens_kv_valid,
                cu_seqlens_padded=self.cu_seqlens_kv_padded,
                cp_size=cp_size,
            )
        return out

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
                1. the output tensor, with shape [total_seqlen_q, nhq, hd]
                2. the softmax lse tensor, with shape [b, nhq, max_seqlen_q]
        """
        is_training = kwargs.get("is_training", True)
        use_fused_attention = kwargs.get("use_fused_attention", False)
        dropout_p = kwargs.get("dropout_p", 0.0)
        cp_global_ranks = kwargs.get("cp_global_ranks", None)
        cp_group = kwargs.get("cp_group", None)
        cp_stream = kwargs.get("cp_stream", None)

        assert isinstance(
            attn_mask_type, AttnMaskType
        ), "attn_mask_type must be an AttnMaskType!"
        if attn_mask_type == AttnMaskType.CAUSAL:
            attn_mask_type_ = "padding_causal"
        else:
            attn_mask_type_ = "padding"

        out_ret, softmax_lse = AttnFuncTERingAttnWithKVP2P.apply(
            is_training,
            q,
            k,
            v,
            torch.tensor(
                self.q_ranges.to_cu_seqlens(self.q_ranges.total_seqlen),
                dtype=torch.int32,
                device=q.device,
            ),
            torch.tensor(
                self.k_ranges.to_cu_seqlens(self.k_ranges.total_seqlen),
                dtype=torch.int32,
                device=k.device,
            ),
            self.max_seqlen_q,
            self.max_seqlen_k,
            self.cu_seqlens_q_padded,
            self.cu_seqlens_kv_padded,
            dropout_p,
            self.softmax_scale,
            attn_mask_type_,
            self.deterministic,
            use_fused_attention,
            cp_group,
            cp_global_ranks,
            cp_stream,
        )

        return out_ret, softmax_lse


class TERingAttnWithKVAG(AttnBaselineInterface):
    def __init__(
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
    ):
        super().__init__()
        self.q_ranges = q_ranges
        self.k_ranges = k_ranges
        self.attn_mask_type = attn_mask_type
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_k
        self.softmax_scale = softmax_scale
        self.deterministic = deterministic

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
            x_global (torch.Tensor): the global tensor to be dispatched, with shape [s, ...]
            cp_rank (int): the cp local rank
            cp_size (int): the cp world size
            cp_group (dist.ProcessGroup): the cp process group
            kwargs: additional arguments

        Returns:
            torch.Tensor: the dispatched local tensor
        """
        attention_mask = kwargs.get("attention_mask", None)
        qkv_format = kwargs.get("qkv_format", None)
        tensor_type = kwargs.get("tensor_type", None)
        self.cp_group = cp_group
        assert (
            qkv_format == "thd" or qkv_format == "sbhd"
        ), f"qkv_format is None or {qkv_format} is not support"
        assert (
            tensor_type == "q" or tensor_type == "v" or tensor_type == "k"
        ), f"tensor_type is None or {tensor_type} is not supported"
        if tensor_type == "q":
            cu_seqlens = self.q_ranges.to_cu_seqlens(self.q_ranges.total_seqlen)
        else:
            cu_seqlens = self.k_ranges.to_cu_seqlens(self.k_ranges.total_seqlen)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=x_global.device)
        if qkv_format == "sbhd":
            pass
        elif qkv_format == "thd":
            cu_seqlens_valid = compute_cu_seqlens_padded_with_attention_mask(
                cu_seqlens=cu_seqlens, attention_mask=attention_mask
            )
            seqlens = cu_seqlens_valid[1:] - cu_seqlens_valid[:-1]
            seqlens = (seqlens + 2 * cp_size - 1) // (2 * cp_size) * (2 * cp_size)
            cu_seqlens_padded = torch.nn.functional.pad(
                seqlens.cumsum_(dim=0), (1, 0)
            ).to(torch.int32)
            out = pad_tensor_and_split(
                x_global,
                cu_seqlens=cu_seqlens_valid,
                cu_seqlens_padded=cu_seqlens_padded,
                cp_size=cp_size,
                rank=cp_rank,
            )
            if tensor_type == "q":
                self.cu_seqlens_q = cu_seqlens
                self.cu_seqlens_q_padded = cu_seqlens_padded
                self.cu_seqlens_q_valid = cu_seqlens_valid
            elif tensor_type == "k":
                self.cu_seqlens_kv = cu_seqlens
                self.cu_seqlens_kv_padded = cu_seqlens_padded
                self.cu_seqlens_kv_valid = cu_seqlens_valid
        return out

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
        s_local = x_local.shape[0]
        assert s_local % 2 == 0, "Sequence length per GPU needs to be divisible by 2"
        qkv_format = kwargs.get("qkv_format", None)
        tensor_type = kwargs.get("tensor_type", None)
        assert (
            qkv_format == "thd" or qkv_format == "sbhd"
        ), f"qkv_format is None or {qkv_format} is not support"
        assert (
            tensor_type == "q" or tensor_type == "v" or tensor_type == "k"
        ), f"tensor_type is None or {tensor_type} is not supported"
        # x_global shape of (s_local * cp_size, ...)
        x_global = all_gather_fwd_scatter_bwd(x_local, cp_group, dim=0).contiguous()
        x_global = x_global.view((cp_size, *x_local.shape))
        if qkv_format == "sbhd":
            pass
        elif qkv_format == "thd":
            if tensor_type == "q":
                out = unpad_tensor_after_gather(
                    x_global,
                    cu_seqlens=self.cu_seqlens_q_valid,
                    cu_seqlens_padded=self.cu_seqlens_q_padded,
                    cp_size=cp_size,
                )
            else:
                out = unpad_tensor_after_gather(
                    x_global,
                    cu_seqlens=self.cu_seqlens_kv_valid,
                    cu_seqlens_padded=self.cu_seqlens_kv_padded,
                    cp_size=cp_size,
                )
        return out

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
                1. the output tensor, with shape [total_seqlen_q, nhq, hd]
                2. the softmax lse tensor, with shape [b, nhq, max_seqlen_q]
        """
        assert isinstance(
            attn_mask_type, AttnMaskType
        ), "multi AttnMaskType is not supported!"
        if attn_mask_type.name == "CAUSAL":
            attn_mask_type_ = "padding_causal"
        else:
            attn_mask_type_ = "padding"
        cp_stream = kwargs.get("cp_stream", None)

        return AttnFuncTERingAttnWithKVAG.apply(
            True,
            q,
            k,
            v,
            self.cu_seqlens_q,
            self.cu_seqlens_kv,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.cu_seqlens_q_padded,
            self.cu_seqlens_kv_padded,
            0,
            softmax_scale,
            "thd",
            attn_mask_type_,
            "no_bias",
            None,
            self.deterministic,
            False,
            self.cp_group,
            cp_stream,
        )
