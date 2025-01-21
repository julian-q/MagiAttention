import torch
import torch.distributed as dist

from zeus.common.enum import AttnMaskType
from zeus.common.ranges import AttnRanges

from .interface import AttnBaselineInterface


class TERingAttnWithKVP2P(AttnBaselineInterface):
    def __init__(self):
        super().__init__()

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
        raise NotImplementedError("TODO (ysy)")

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
        raise NotImplementedError("TODO (ysy)")

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
                1. the output tensor, with shape [total_seqlen_q, b, nhq, hd]
                2. the softmax lse tensor, with shape [b, nhq, max_seqlen_q]
        """
        raise NotImplementedError("TODO (ysy)")


class TERingAttnWithKVAG(AttnBaselineInterface):
    def __init__(self):
        super().__init__()

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
        raise NotImplementedError("TODO (wqg)")

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
        raise NotImplementedError("TODO (wqg)")

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
                1. the output tensor, with shape [total_seqlen_q, b, nhq, hd]
                2. the softmax lse tensor, with shape [b, nhq, max_seqlen_q]
        """
        raise NotImplementedError("TODO (wqg)")
