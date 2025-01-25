from dataclasses import dataclass
from logging import getLogger
from typing import Callable, Optional

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn_interface import _flex_flash_attn_backward, _flex_flash_attn_forward
from torch.distributed import Work

from zeus.comm.primitive import group_cast_collective, group_reduce_collective
from zeus.common.enum import AttnOverlapMode
from zeus.meta.collection import AttnCalcMeta, CommMeta

logger = getLogger("zeus")


@dataclass
class DistFlashAttnConfig:
    """
    静态config, 在程序初始化的时候就应该被定义
    """

    num_heads: int
    head_dim: int
    dtype: torch.dtype
    overlap_mode: AttnOverlapMode = AttnOverlapMode.STATIC
    overlap_degree: int | None = 1
    deterministic: bool = False

    def __post_init__(self):
        if self.overlap_mode is AttnOverlapMode.STATIC:
            assert self.overlap_degree is not None, (
                "When using static overlap mode, "
                "the overlap_degree should be set explicitly."
            )
        elif self.overlap_mode is AttnOverlapMode.DYNAMIC:
            assert self.overlap_degree is None, (
                "When using dynamic overlap mode, "
                "the overlap_degree should not be set."
            )


class DistFlashAttnRuntime:
    """
    分布式Flash Attention的运行时类。

    Args:
        config (DistFlashAttnConfig): 分布式Flash Attention的静态配置
        runtime_meta (DistFlashAttnRuntimeMeta): 分布式Flash Attention的运行时元数据


    NOTE(xiaowu):
        每一次forward都需要实例化一个DistFlashAttnRuntime
        这个runtime实例需要做以下事情:
        初始化时:
            - 申请remote kv buffer, 每一层都会复用
            - 将remote kv buffer提前切分, 减少每层slice的cpu overhead

        为每一层forward提供可供调度的原语:
            - fetch_remote_kv: 将remote kv buffer从其他rank拉取到本地
            - do_attn_partially: 计算attn结果的一部分
            - result_correction: 对attn结果进行correction
    """

    def __init__(
        self,
        comm_meta: CommMeta,
        calc_meta: AttnCalcMeta,
        cp_group_nccl: dist.ProcessGroup,
    ):
        self.comm_meta = comm_meta
        self.calc_meta = calc_meta
        self.cp_group_nccl = cp_group_nccl

        self.overlap_degree = comm_meta.overlap_degree

    def fetch_remote_kv(
        self, k: torch.Tensor, v: torch.Tensor, overlap_stage: int
    ) -> tuple[
        Work,
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        将remote kv buffer从其他rank拉取到本地, 并返回对应的Work

        Args:
            k(torch.Tensor):
            v(torch.Tensor):
            overlap_stage(int): 当前的overlap stage

        Returns:
            remote_kv_work(Work): 通信handle, 用于等待通信完成
            remote_kv_buffer(torch.Tensor): 远程kv buffer
            remote_kv_post_process_fn(Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]): 通信后处理函数

        Shape:
            - k: [num_tokens_kv_local, num_heads, head_dim]
            - v: [num_tokens_kv_local, num_heads, head_dim]
            - remote_kv_buffer: [num_tokens_k_remote_i + num_tokens_v_remote_i, num_heads, head_dim],
                i = 0, 1, ..., overlap_degree - 1
        """

        _, num_heads, head_dim = k.shape
        dtype = k.dtype
        device = k.device

        remote_kv_buffer = torch.empty(
            [
                self.comm_meta.num_remote_tokens_per_overlap_stage[overlap_stage] * 2,
                num_heads,
                head_dim,
            ],
            dtype=dtype,
            device=device,
        )

        logger.debug(
            f"RANK: {dist.get_rank()}, {remote_kv_buffer.shape=}, {k.shape=}, {v.shape=}"
        )

        group_cast_collective_args = self.comm_meta.group_cast_collective_args_list[
            overlap_stage
        ]

        local_kv = torch.cat([k, v], dim=0)

        remote_kv_work, remote_kv_preprocess_fn = group_cast_collective(
            input=local_kv,
            output=remote_kv_buffer,
            input_split_size_list=group_cast_collective_args.input_split_size_list * 2,
            output_split_size_list=group_cast_collective_args.output_split_size_list
            * 2,
            dst_indices_list=group_cast_collective_args.dst_indices_list * 2,
            src_index_list=group_cast_collective_args.src_index_list * 2,
            group=self.cp_group_nccl,
            async_op=True,
        )

        def custom_post_process_fn(remote_kv_buffer: torch.Tensor) -> torch.Tensor:
            remote_kv_buffer = remote_kv_preprocess_fn(remote_kv_buffer)
            remote_k, remote_v = torch.chunk(remote_kv_buffer, 2, dim=0)
            return remote_k, remote_v

        return (
            remote_kv_work,
            remote_kv_buffer,
            custom_post_process_fn,
        )

    def attn_fwd_partial(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        overlap_stage: Optional[int] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算attn结果的一部分

        Args:
            q(torch.Tensor):
            k(torch.Tensor):
            v(torch.Tensor):
            overlap_stage(Optional[int]): 当前的overlap stage, 如果为None, 则表示是local的attn, 否则表示是remote的attn

        Returns:
            out(torch.Tensor):
            lse(torch.Tensor): log sum exp


        Shape:
            - q: [num_tokens_q, num_heads, head_dim]
            - k: [num_tokens_k, num_heads, head_dim]
            - v: [num_tokens_v, num_heads, head_dim]
            - out: [num_tokens_q, num_heads, head_dim]
            - lse: [num_heads, num_tokens_q]
            REVIEW(xiaowu): 这里lse的shape是否正确?
        """

        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        logger.debug(
            f"RANK: {dist.get_rank()}, {q.shape=}, {k.shape=}, {v.shape=}, {q.device=}, {k.device=}, {v.device=}"
        )
        logger.debug(
            f"RANK: {dist.get_rank()}, {attn_arg.q_ranges=}, {attn_arg.k_ranges=}, "
            f"{attn_arg.is_causal_mapping=}, {attn_arg.max_seqlen_q=}, {attn_arg.max_seqlen_k=}, {attn_arg.skip_attn=}"
        )

        # 计算attn
        if attn_arg.skip_attn:
            num_tokens, num_heads, head_dim = q.shape
            out = torch.zeros_like(q)
            # REVIEW(xiaowu): dtype
            lse = (
                torch.ones(
                    [num_heads, num_tokens], dtype=torch.float32, device=q.device
                )
                * -torch.inf
            )
        else:
            out, _, _, _, _, lse = _flex_flash_attn_forward(
                q=q,
                k=k,
                v=v,
                **attn_arg.to_ffa_args(),
                softmax_scale=q.shape[-1] ** (-0.5),
                deterministic=deterministic,
            )

            # TODO(xiaowu): opt performance
            start, end = 0, q.size(0)
            for q_range in attn_arg.q_ranges:
                if q_range[0] > start:
                    out[start : q_range[0]] = 0
                start = q_range[1]
            if end > start:
                out[start:end] = 0

        return out, lse

    def result_correction(
        self,
        out_list: list[torch.Tensor],
        lse_list: list[torch.Tensor],
        overlap_degree: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对attn结果进行correction

        Args:
            out_list(list[torch.Tensor]):
            lse_list(list[torch.Tensor]):

        Returns:
            out(torch.Tensor):
            lse(torch.Tensor):

        Shape:
            - out: [num_tokens_q, num_heads, head_dim]
            - lse: [num_heads, num_tokens_q]
        """
        # 当overlap_degree为1时, 也至少有2个lse和out, 一个是local的, 一个是remote的
        assert len(out_list) == len(lse_list) == overlap_degree + 1

        curr_lse = None
        curr_out = None

        for i in range(overlap_degree):
            if i == 0:
                curr_lse = self.correct_attn_lse(lse_list[0], lse_list[1])
                curr_out = self.correct_attn_output(
                    out_list[0], lse_list[0], out_list[1], lse_list[1], curr_lse
                )
            else:
                original_lse = curr_lse
                original_out = curr_out
                curr_lse = self.correct_attn_lse(original_lse, lse_list[i + 1])
                curr_out = self.correct_attn_output(
                    original_out,
                    original_lse,
                    out_list[i + 1],
                    lse_list[i + 1],
                    curr_lse,
                )

        return curr_out, curr_lse

    def attn_bwd_partial(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        overlap_stage: Optional[int] = None,
        deterministic: bool = False,
    ):
        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        partial_dq, partial_dk, partial_dv = (
            torch.zeros_like(q),  # WTFFFFF???
            torch.empty_like(k),
            torch.empty_like(v),
        )

        if attn_arg.skip_attn:
            partial_dk.fill_(0)
            partial_dv.fill_(0)
        else:
            partial_dq, partial_dk, partial_dv, _ = _flex_flash_attn_backward(
                dout=do,
                q=q,
                k=k,
                v=v,
                out=o,
                softmax_lse=lse,
                dq=partial_dq,
                dk=partial_dk,
                dv=partial_dv,
                **attn_arg.to_ffa_args(),
                softmax_scale=q.shape[-1] ** (-0.5),
                deterministic=deterministic,
            )

        return partial_dq, partial_dk, partial_dv

    def reduce_partial_dkv(
        self,
        partial_dk: torch.Tensor,
        partial_dv: torch.Tensor,
        partial_local_dk: torch.Tensor,
        partial_local_dv: torch.Tensor,
        overlap_stage: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_cast_collective_args = self.comm_meta.group_cast_collective_args_list[
            overlap_stage
        ]

        (
            partial_local_dk_work,
            partial_local_dk_post_process_fn,
        ) = group_reduce_collective(
            input=partial_dk,
            output=partial_local_dk,
            input_split_size_list=group_cast_collective_args.output_split_size_list,
            output_split_size_list=group_cast_collective_args.input_split_size_list,
            dst_index_list=group_cast_collective_args.src_index_list,
            src_indices_list=group_cast_collective_args.dst_indices_list,
            group=self.cp_group_nccl,
            async_op=True,
        )

        (
            partial_local_dv_work,
            partial_local_dv_post_process_fn,
        ) = group_reduce_collective(
            input=partial_dv,
            output=partial_local_dv,
            input_split_size_list=group_cast_collective_args.output_split_size_list,
            output_split_size_list=group_cast_collective_args.input_split_size_list,
            dst_index_list=group_cast_collective_args.src_index_list,
            src_indices_list=group_cast_collective_args.dst_indices_list,
            group=self.cp_group_nccl,
            async_op=True,
        )

        partial_local_dk_work.wait()
        partial_local_dv_work.wait()

        partial_local_dk = partial_local_dk_post_process_fn(partial_local_dk)
        partial_local_dv = partial_local_dv_post_process_fn(partial_local_dv)

        return partial_local_dk, partial_local_dv

    def reduce_partial_dq(
        self,
        partial_remote_dq: torch.Tensor,
        partial_local_dq: torch.Tensor,
        overlap_stage: int,
    ) -> torch.Tensor:
        return partial_local_dq + partial_remote_dq

    @staticmethod
    def safe_subtract(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Safely subtracts two tensors. where the subtraction results of two -inf will be set to -inf.
        """

        eq = (a == b) & (a == float("-inf"))
        sub = a - b
        sub = torch.where(eq, torch.fill(sub, float("-inf")), sub)

        # A faster way to do the same thing as the above
        # neg_inf = (a == b) & (a == float("-inf"))
        # not_neg_inf = torch.logical_not(neg_inf)
        # b_not_neg_inf = b * not_neg_inf.float()
        # sub_comp = a - b_not_neg_inf

        # assert torch.testing.assert_close(sub, sub_comp)
        return sub

    @staticmethod
    def correct_attn_lse(
        lse1: torch.Tensor,
        lse2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Corrects the log sum exp tensor for online attention.

        Args:
            lse1(torch.Tensor): log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
            lse2(torch.Tensor): log sum exp tensor, with shape: [batch_size, num_heads, seq_len]

        Returns:
            lse(torch.Tensor): corrected log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
        """
        min_lse = torch.min(lse1, lse2).to(torch.float32)
        max_lse = torch.max(lse1, lse2).to(torch.float32)

        # formula: lse = log(exp(lse1) + exp(lse2))
        #              = lse1 + log(1 + exp(lse2 - lse1))
        #              = max_lse + log(1 + exp(min_lse - max_lse))
        #              = max_lse + log1p(exp(min_lse - max_lse))
        #              = max_lse + softplus(min_lse - max_lse)
        lse = max_lse + F.softplus(DistFlashAttnRuntime.safe_subtract(min_lse, max_lse))

        return lse.to(lse1.dtype)

    @staticmethod
    def correct_attn_output(
        o1: torch.Tensor,
        lse1: torch.Tensor,
        o2: torch.Tensor,
        lse2: torch.Tensor,
        lse: torch.Tensor,
    ) -> torch.Tensor:
        """
        Corrects the output tensor for online attention.

        Args:
            o1(torch.Tensor): local output tensor o1, with shape: [batch_size, seq_len, num_heads, head_dim]
            lse1(torch.Tensor): local lse for o1, with shape: [batch_size, num_heads, seq_len]
            o2(torch.Tensor): local output tensor o2, with shape: [batch_size, seq_len, num_heads, head_dim]
            lse2(torch.Tensor): local lse for o2, with shape: [batch_size, num_heads, seq_len]
            lse(torch.Tensor): global lse, with shape: [batch_size, num_heads, seq_len]

        Returns:
            o(torch.Tensor): corrected global output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """
        # formula: lsei_ = exp(lsei - lse)
        # shape: [b, h, s] -> [b, s, h] -> [b, s, h, 1]
        lse1_, lse2_ = [
            DistFlashAttnRuntime.safe_subtract(lsei, lse)
            .exp()
            .transpose(-1, -2)
            .unsqueeze(-1)
            .to(torch.float32)
            for lsei in [lse1, lse2]
        ]

        o = lse1_ * o1 + lse2_ * o2

        return o.to(o1.dtype)

    @classmethod
    def from_attn_meta(
        cls,
        comm_meta: CommMeta,
        calc_meta: AttnCalcMeta,
        cp_group_nccl: dist.ProcessGroup,
    ):
        return cls(comm_meta, calc_meta, cp_group_nccl)


class DistFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        dist_attn_config: DistFlashAttnConfig,
        dist_attn_runtime: DistFlashAttnRuntime,
    ):
        """
        分布式Flash Attention的forward函数

        Args:
            local_q(torch.Tensor):
            local_k(torch.Tensor):
            local_v(torch.Tensor):
            dist_attn_config(DistFlashAttnConfig):
            dist_attn_runtime(DistFlashAttnRuntime):

        Returns:
            out(torch.Tensor):

        Shape:
            - local_q: [num_tokens_q_local, num_heads, head_dim]
            - local_k: [num_tokens_k_local, num_heads, head_dim]
            - local_v: [num_tokens_v_local, num_heads, head_dim]
        """

        out_list = []
        lse_list = []

        ###########################################
        # Pre-fetch remote kv for overlap stage 0 #
        ###########################################
        (
            remote_kv_work,
            remote_kv_buffer,
            remote_kv_post_process_fn,
        ) = dist_attn_runtime.fetch_remote_kv(k=local_k, v=local_v, overlap_stage=0)

        #########################
        # Do attn with local kv #
        #########################
        out, lse = dist_attn_runtime.attn_fwd_partial(
            q=local_q,
            k=local_k,
            v=local_v,
            overlap_stage=None,
            deterministic=dist_attn_config.deterministic,
        )
        out_list.append(out)
        lse_list.append(lse)

        for overlap_stage in range(dist_attn_runtime.overlap_degree):
            # Wait for remote kv to be fetched
            remote_kv_work.wait()

            curr_remote_k, curr_remote_v = remote_kv_post_process_fn(remote_kv_buffer)
            ###########################################
            # Pre-fetch remote kv for overlap stage i #
            ###########################################
            if overlap_stage < dist_attn_runtime.overlap_degree - 1:
                (
                    remote_kv_work,
                    remote_kv_buffer,
                    remote_kv_post_process_fn,
                ) = dist_attn_runtime.fetch_remote_kv(
                    k=local_k, v=local_v, overlap_stage=overlap_stage + 1
                )

            ##########################
            # Do attn with remote kv #
            ##########################
            out, lse = dist_attn_runtime.attn_fwd_partial(
                q=local_q,
                k=curr_remote_k,
                v=curr_remote_v,
                overlap_stage=overlap_stage,
                deterministic=dist_attn_config.deterministic,
            )

            out_list.append(out)
            lse_list.append(lse)

        ########################
        # Do result correction #
        ########################
        out, final_lse = dist_attn_runtime.result_correction(
            out_list=out_list,
            lse_list=lse_list,
            overlap_degree=dist_attn_runtime.overlap_degree,
        )

        ctx.save_for_backward(local_q, local_k, local_v, out, final_lse)
        ctx.dist_attn_config = dist_attn_config
        ctx.dist_attn_runtime = dist_attn_runtime

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_q, local_k, local_v, out, final_lse = ctx.saved_tensors
        dist_attn_config: DistFlashAttnConfig = ctx.dist_attn_config
        dist_attn_runtime: DistFlashAttnRuntime = ctx.dist_attn_runtime

        (
            remote_kv_work,
            remote_kv_buffer,
            remote_kv_post_process_fn,
        ) = dist_attn_runtime.fetch_remote_kv(k=local_k, v=local_v, overlap_stage=0)

        (
            partial_local_dq,
            partial_local_dk,
            partial_local_dv,
        ) = dist_attn_runtime.attn_bwd_partial(
            do=grad_output,
            q=local_q,
            k=local_k,
            v=local_v,
            o=out,
            lse=final_lse,
            overlap_stage=None,
            deterministic=dist_attn_config.deterministic,
        )

        for overlap_stage in range(dist_attn_runtime.overlap_degree):
            remote_kv_work.wait()

            curr_remote_k, curr_remote_v = remote_kv_post_process_fn(remote_kv_buffer)

            if overlap_stage < dist_attn_runtime.overlap_degree - 1:
                (
                    remote_kv_work,
                    remote_kv_buffer,
                    remote_kv_post_process_fn,
                ) = dist_attn_runtime.fetch_remote_kv(
                    k=local_k, v=local_v, overlap_stage=overlap_stage + 1
                )

            (
                partial_remote_dq,
                partial_remote_dk,
                partial_remote_dv,
            ) = dist_attn_runtime.attn_bwd_partial(
                do=grad_output,
                q=local_q,
                k=curr_remote_k,
                v=curr_remote_v,
                o=out,
                lse=final_lse,
                overlap_stage=overlap_stage,
                deterministic=dist_attn_config.deterministic,
            )

            partial_local_dk, partial_local_dv = dist_attn_runtime.reduce_partial_dkv(
                partial_dk=partial_remote_dk,
                partial_dv=partial_remote_dv,
                partial_local_dk=partial_local_dk,
                partial_local_dv=partial_local_dv,
                overlap_stage=overlap_stage,
            )

            partial_local_dq = dist_attn_runtime.reduce_partial_dq(
                partial_remote_dq=partial_remote_dq,
                partial_local_dq=partial_local_dq,
                overlap_stage=overlap_stage,
            )

        return partial_local_dq, partial_local_dk, partial_local_dv, None, None


class DistFlashAttn(torch.nn.Module):
    """
    Distributed Flash Attention module class

    Args:
        config(DistFlashAttnConfig): Configuration for distributed Flash Attention
    """

    def __init__(self, config: DistFlashAttnConfig):
        super().__init__()
        self.config = config

    def forward(self, q, k, v, dist_attn_runtime: DistFlashAttnRuntime):
        return DistFlashAttnFunc.apply(q, k, v, self.config, dist_attn_runtime)
