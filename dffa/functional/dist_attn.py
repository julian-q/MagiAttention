from logging import getLogger
from typing import Optional

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn_interface import _flex_flash_attn_backward, _flex_flash_attn_forward

import dffa
from dffa.comm.primitive import group_cast_collective, group_reduce_collective
from dffa.comm.work import WorkWithPostProcessFn
from dffa.common.ranges import NaiveRanges
from dffa.meta.collection import AttnCalcMeta, CommMeta
from dffa.utils import nvtx, to_higher_fp_dtype

from .sdpa import sdpa_bwd, sdpa_fwd
from .utils import safe_subtract

logger = getLogger("dffa")


@nvtx.instrument_nvtx
@torch.compile
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

    min_lse = to_higher_fp_dtype(torch.min(lse1, lse2), torch.float32)
    max_lse = to_higher_fp_dtype(torch.max(lse1, lse2), torch.float32)

    # formula derivation:
    # lse = log(exp(lse1) + exp(lse2))
    #     = lse1 + log(1 + exp(lse2 - lse1))
    #     = max_lse + log(1 + exp(min_lse - max_lse))
    #     = max_lse + log1p(exp(min_lse - max_lse))
    #     = max_lse + softplus(min_lse - max_lse)
    lse = max_lse + F.softplus(safe_subtract(min_lse, max_lse))

    return lse.to(lse1.dtype)


@nvtx.instrument_nvtx
@torch.compile
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
        to_higher_fp_dtype(
            safe_subtract(lsei, lse).exp().transpose(-1, -2).unsqueeze(-1),
            torch.float32,
        )
        for lsei in [lse1, lse2]
    ]

    o = lse1_ * o1 + lse2_ * o2

    return o.to(o1.dtype)


# TODO: fuse this kernel in the future
@nvtx.instrument_nvtx
def result_correction(
    out_list: list[torch.Tensor],
    lse_list: list[torch.Tensor],
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
    curr_lse = None
    curr_out = None

    for i in range(len(lse_list) - 1):
        if i == 0:
            curr_lse = correct_attn_lse(lse_list[0], lse_list[1])
            curr_out = correct_attn_output(
                out_list[0], lse_list[0], out_list[1], lse_list[1], curr_lse
            )
        else:
            original_lse = curr_lse
            original_out = curr_out
            curr_lse = correct_attn_lse(original_lse, lse_list[i + 1])
            curr_out = correct_attn_output(
                original_out,
                original_lse,
                out_list[i + 1],
                lse_list[i + 1],
                curr_lse,
            )

    return curr_out, curr_lse


# TODO: put this logic into kernel
@nvtx.instrument_nvtx
def out_zero_fill_correction(
    out: torch.Tensor,
    out_zero_fill_ranges: NaiveRanges,
) -> torch.Tensor:
    for fill_start, fill_end in out_zero_fill_ranges:
        out[fill_start:fill_end].fill_(0)

    return out


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
        cp_group_kv: dist.ProcessGroup,
        cp_group_dkv: dist.ProcessGroup,
        deterministic: bool,
    ):
        assert dist.get_backend(cp_group_kv) == dist.Backend.NCCL
        assert dist.get_backend(cp_group_dkv) == dist.Backend.NCCL

        self.comm_meta = comm_meta
        self.calc_meta = calc_meta
        self.cp_group_kv = cp_group_kv
        self.cp_group_dkv = cp_group_dkv
        self.deterministic = deterministic

        # NOTE: get the real overlap degree from comm meta
        # instead of the initial one from overlap config
        self.overlap_degree = comm_meta.overlap_degree

        assert self.overlap_degree >= 1, f"{self.overlap_degree} must be >= 1"

    @nvtx.instrument_nvtx
    def fetch_remote_kv(
        self,
        local_kv: torch.Tensor,
        overlap_stage: int,
    ) -> tuple[WorkWithPostProcessFn, torch.Tensor]:
        """
        将remote kv buffer从其他rank拉取到本地, 并返回对应的Work和对应的buffer

        Args:
            local_kv(torch.Tensor): the concatenated local kv tensor
            overlap_stage(int): 当前的overlap stage

        Returns:
            remote_kv_work(WorkWithPostProcessFn): 通信handle, 用于等待通信完成
            remote_kv_buffer(torch.Tensor): 远程kv buffer

        Shape:
            - local_kv: [num_tokens_kv_local, num_heads, head_dim]
            - remote_kv_buffer: [num_tokens_kv_remote_i, num_heads, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        _, num_heads, head_dim = local_kv.shape
        dtype = local_kv.dtype
        device = local_kv.device

        group_collective_args = self.comm_meta.group_collective_args_list[overlap_stage]

        remote_kv_buffer = torch.empty(
            [
                self.comm_meta.num_remote_tokens_per_stage[overlap_stage] * 2,
                num_heads,
                head_dim,
            ],
            dtype=dtype,
            device=device,
        )

        # DE-BUG
        logger.debug(
            f"RANK: {dist.get_rank()}, {remote_kv_buffer.shape=}, {local_kv.shape=}"
        )

        remote_kv_work = group_cast_collective(
            input=local_kv,
            output=remote_kv_buffer,
            **group_collective_args.to_group_cast_args(),
            group=self.cp_group_kv,
            async_op=True,
        )

        return remote_kv_work, remote_kv_buffer

    @nvtx.instrument_nvtx
    def attn_fwd_partial(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        overlap_stage: Optional[int] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算attn结果的一部分

        Args:
            q(torch.Tensor):
            kv(torch.Tensor):
            overlap_stage(Optional[int]): 当前的overlap stage,
                如果为None, 则表示是local的attn, 否则表示是remote的attn

        Returns:
            out(torch.Tensor):
            lse(torch.Tensor): log sum exp

        Shape:
            - q: [num_tokens_q, num_heads, head_dim]
            - kv: [num_tokens_kv, num_heads, head_dim]
            - out: [num_tokens_q, num_heads, head_dim]
            - lse: [num_heads, num_tokens_q]
            REVIEW(xiaowu): 这里lse的shape是否正确?
        """

        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        # DE-BUG
        logger.debug(
            f"RANK: {dist.get_rank()}, {q.shape=}, {kv.shape=}, "
            f"{q.device=}, {kv.device=}, "
            f"{attn_arg=}"
        )

        # 计算attn
        if attn_arg.skip_attn:
            out = torch.zeros_like(q)
            if len(q.shape) == 3:
                num_tokens_q, num_heads, _ = q.shape
            elif len(q.shape) == 4:
                batch_size, num_heads, num_tokens_q, _ = q.shape

            lse = to_higher_fp_dtype(
                torch.full(
                    [num_heads, num_tokens_q]
                    if len(q.shape) == 3
                    else [batch_size, num_heads, num_tokens_q],
                    fill_value=-torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                ),
                q.dtype,
            )
        else:
            k, v = self.chunk_kv(kv)
            if dffa.is_sdpa_backend_enable():
                out, lse = sdpa_fwd(
                    q,
                    k,
                    v,
                    attn_arg=attn_arg,
                )
            else:
                with nvtx.add_nvtx_event(
                    f"attn-fwd: area={attn_arg.total_area} | "
                    f"qr={attn_arg.q_ranges} | kr={attn_arg.k_ranges}"
                ):
                    out, _, _, _, _, lse = _flex_flash_attn_forward(
                        q=q,
                        k=k,
                        v=v,
                        **attn_arg.to_ffa_args(),
                        softmax_scale=q.shape[-1] ** -0.5,
                        deterministic=deterministic,
                    )

                # fill output with zero indexed by "hole" q ranges
                # TODO: put this logic into kernel
                out_zero_fill_correction(out, attn_arg.out_zero_fill_ranges)

        return out, lse

    @nvtx.instrument_nvtx
    def attn_bwd_partial(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        overlap_stage: Optional[int] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply ffa bwd kernel to get partial dqkv."""

        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        if attn_arg.skip_attn:
            partial_dq = torch.zeros_like(q)
            partial_dkv = torch.zeros_like(kv)
        else:
            k, v = self.chunk_kv(kv)
            if dffa.is_sdpa_backend_enable():
                partial_dq, partial_dk, partial_dv = sdpa_bwd(
                    do=do,
                    q=q,
                    k=k,
                    v=v,
                    o=o,
                    lse=lse,
                    attn_arg=attn_arg,
                )
                partial_dkv = torch.cat([partial_dk, partial_dv], dim=0)
            else:
                # FIXME: here q needs to use 'zeros_like' to initialize
                # since ffa only zero those covered by q_ranges
                # needed to be fixed by ffa kernel in the future
                partial_dq = torch.zeros_like(q)
                partial_dkv = torch.empty_like(kv)
                partial_dk, partial_dv = self.chunk_kv(partial_dkv)

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
                    softmax_scale=q.shape[-1] ** -0.5,
                    deterministic=deterministic,
                )

        return partial_dq, partial_dkv

    @nvtx.instrument_nvtx
    def reduce_partial_dkv(
        self,
        partial_remote_dkv: torch.Tensor,
        partial_local_dkv: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        """reduce remote dkv to add to local dkv for the given overlap stage."""
        group_collective_args = self.comm_meta.group_collective_args_list[overlap_stage]

        partial_local_dkv_work = group_reduce_collective(
            input=partial_remote_dkv,
            output=partial_local_dkv,
            **group_collective_args.to_group_reduce_args(),
            group=self.cp_group_dkv,
            async_op=True,
        )

        return partial_local_dkv_work

    @nvtx.instrument_nvtx
    def reduce_partial_dq(
        self,
        partial_remote_dq: torch.Tensor,
        partial_local_dq: torch.Tensor,
    ) -> torch.Tensor:
        return partial_local_dq.add_(partial_remote_dq)

    @staticmethod
    def concat_kv(
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """concatenate k, v tensors into a single coalesced kv"""
        # TODO: whether can we pack kv togather along certain dim
        # to enhance the performance of ffa kernel
        return torch.cat([k, v], dim=0)

    @staticmethod
    def chunk_kv(
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """chunk the kv tensor into k, v tensor views"""
        return torch.chunk(kv, 2, dim=0)


class DistFlashAttnFunc(torch.autograd.Function):
    """Distributed Flash Attention Function"""

    @staticmethod
    def forward(
        ctx,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        dist_attn_runtime: DistFlashAttnRuntime,
    ):
        """
        分布式Flash Attention的forward函数

        Args:
            local_q(torch.Tensor):
            local_k(torch.Tensor):
            local_v(torch.Tensor):
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

        # cat local k, v into a single coalesced kv
        local_kv = dist_attn_runtime.concat_kv(local_k, local_v)

        # pre-fetch 0th remote kv
        (
            remote_kv_work,
            remote_kv_buffer,
        ) = dist_attn_runtime.fetch_remote_kv(local_kv=local_kv, overlap_stage=0)

        # do attn fwd with local kv
        # overlapped with 0th remote kv comm
        out, lse = dist_attn_runtime.attn_fwd_partial(
            q=local_q,
            kv=local_kv,
            overlap_stage=None,
            deterministic=dist_attn_runtime.deterministic,
        )
        out_list.append(out)
        lse_list.append(lse)

        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote kv prepared
            curr_remote_kv = remote_kv_work.wait_post_process(remote_kv_buffer)

            # pre-fetch (i+1)th remote kv
            if ith_overlap_stage < dist_attn_runtime.overlap_degree - 1:
                (
                    remote_kv_work,
                    remote_kv_buffer,
                ) = dist_attn_runtime.fetch_remote_kv(
                    local_kv=local_kv, overlap_stage=ith_overlap_stage + 1
                )

            # do attn fwd with ith remote kv
            # overlapped with (i+1)th remote kv comm
            out, lse = dist_attn_runtime.attn_fwd_partial(
                q=local_q,
                kv=curr_remote_kv,
                overlap_stage=ith_overlap_stage,
                deterministic=dist_attn_runtime.deterministic,
            )

            out_list.append(out)
            lse_list.append(lse)

        # do result correction to get final out and lse
        out, lse = result_correction(
            out_list=out_list,
            lse_list=lse_list,
        )

        ctx.save_for_backward(local_q, local_kv, out, lse)
        ctx.dist_attn_runtime = dist_attn_runtime

        return out, lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        local_q, local_kv, out, final_lse = ctx.saved_tensors
        dist_attn_runtime: DistFlashAttnRuntime = ctx.dist_attn_runtime

        # pre-fetch 0th remote kv
        (
            remote_kv_work,
            remote_kv_buffer,
        ) = dist_attn_runtime.fetch_remote_kv(local_kv=local_kv, overlap_stage=0)

        # do attn bwd with local kv
        # overlapped with 0th remote kv comm
        (
            partial_local_dq,
            partial_local_dkv,
        ) = dist_attn_runtime.attn_bwd_partial(
            do=grad_output,
            q=local_q,
            kv=local_kv,
            o=out,
            lse=final_lse,
            overlap_stage=None,
            deterministic=dist_attn_runtime.deterministic,
        )

        partial_local_dkv_work = WorkWithPostProcessFn()
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote kv prepared
            curr_remote_kv = remote_kv_work.wait_post_process(remote_kv_buffer)

            # pre-fetch (i+1)th remote kv
            if ith_overlap_stage < dist_attn_runtime.overlap_degree - 1:
                (
                    remote_kv_work,
                    remote_kv_buffer,
                ) = dist_attn_runtime.fetch_remote_kv(
                    local_kv=local_kv, overlap_stage=ith_overlap_stage + 1
                )

            # do attn bwd with ith remote kv
            # overlapped with (i+1)th remote kv comm
            (
                partial_remote_dq,
                partial_remote_dkv,
            ) = dist_attn_runtime.attn_bwd_partial(
                do=grad_output,
                q=local_q,
                kv=curr_remote_kv,
                o=out,
                lse=final_lse,
                overlap_stage=ith_overlap_stage,
                deterministic=dist_attn_runtime.deterministic,
            )

            # reduce ith partial dkv
            partial_local_dkv_work_new = dist_attn_runtime.reduce_partial_dkv(
                partial_remote_dkv=partial_remote_dkv,
                partial_local_dkv=partial_local_dkv,
                overlap_stage=ith_overlap_stage,
            )

            # reduce ith partial dq
            # overlapped with ith remote dkv comm
            partial_local_dq = dist_attn_runtime.reduce_partial_dq(
                partial_remote_dq=partial_remote_dq,
                partial_local_dq=partial_local_dq,
            )

            # wait for (i-1)th partial dkv reduced
            # overlapped with ith remote dkv comm
            if ith_overlap_stage > 0:
                partial_local_dkv = partial_local_dkv_work.wait_post_process(
                    partial_local_dkv
                )
            partial_local_dkv_work = partial_local_dkv_work_new

        # wait for last partial dkv reduced
        partial_local_dkv = partial_local_dkv_work.wait_post_process(partial_local_dkv)

        # chunk final dkv into dk and dv
        partial_local_dk, partial_local_dv = dist_attn_runtime.chunk_kv(
            partial_local_dkv
        )

        return partial_local_dq, partial_local_dk, partial_local_dv, None, None


def dist_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dist_attn_runtime: DistFlashAttnRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    return DistFlashAttnFunc.apply(q, k, v, dist_attn_runtime)
