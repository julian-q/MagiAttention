from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn_interface import flex_flash_attn_func
from torch.distributed import Work


def multi_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size: list[int],
    output_split_size: list[int],
    dst_indices: list[list[int]],
    src_indices: list[list[int]],
    group: Optional[dist.ProcessGroup] = None,
) -> Work:
    """
    Args:
        input: [sum(input_split_size), ...]
        output: [sum(output_split_size), ...]
        input_split_size: [N]
        output_split_size: [M]
        dst_indices: [N, ?]
        src_indices: [M, ?]

    NOTE(xiaowu): 使用a2a-v实现
    """
    assert len(input_split_size) == len(dst_indices)
    assert len(output_split_size) == len(src_indices)

    pass


@dataclass
class DistFlashAttnConfig:
    """
    静态config, 在程序初始化的时候就应该被定义
    """

    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device

    # REVIEW(xiaowu): overlap_degree应该是静态的嘛？
    overlap_degree: int = 1


@dataclass
class AttnArg:
    q_ranges: torch.Tensor
    kv_ranges: torch.Tensor
    is_causal_mapping: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


# HACK(xiaowu): mock数据以通过mypy
@dataclass
class AttnCalcMeta:
    local_attn_arg: AttnArg
    remote_attn_args_list: list[AttnArg]


# HACK(xiaowu): mock数据以通过mypy
@dataclass
class DispatchMeta:
    num_remote_token: int
    split_size: int
    input_split_size_list: list[list[int]]
    output_split_size_list: list[list[int]]
    dst_indices_list: list[list[list[int]]]
    src_indices_list: list[list[list[int]]]


@dataclass
class DistFlashAttnRuntimeMeta:
    """
    分布式Flash Attention的运行时元数据类。

    Args:
        attn_calc_meta (AttnCalcMeta): 注意力计算相关的元数据
        q_dispatch_meta (DispatchMeta): query张量的分发相关元数据
        kv_dispatch_meta (DispatchMeta): key/value张量的分发相关元数据
    """

    attn_calc_meta: AttnCalcMeta
    q_dispatch_meta: DispatchMeta
    kv_dispatch_meta: DispatchMeta


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
        self, runtime_meta: DistFlashAttnRuntimeMeta, config: DistFlashAttnConfig
    ):
        self.runtime_meta = runtime_meta
        self.config = config

        # 申请远程kv buffer
        self.remote_k_buffer = torch.empty(
            [
                runtime_meta.kv_dispatch_meta.num_remote_token,
                config.num_heads,
                config.head_dim,
            ],
            dtype=config.dtype,
            device=config.device,
        )
        self.remote_v_buffer = torch.empty(
            [
                runtime_meta.kv_dispatch_meta.num_remote_token,
                config.num_heads,
                config.head_dim,
            ],
            dtype=config.dtype,
            device=config.device,
        )

        self.remote_k_buffer_list = torch.split(
            self.remote_k_buffer, self.runtime_meta.kv_dispatch_meta.split_size, dim=0
        )
        self.remote_v_buffer_list = torch.split(
            self.remote_v_buffer, self.runtime_meta.kv_dispatch_meta.split_size, dim=0
        )

    def fetch_remote_kv(
        self, k: torch.Tensor, v: torch.Tensor, overlap_stage: int
    ) -> tuple[torch.Tensor, Work, torch.Tensor, Work]:
        """
        将remote kv buffer从其他rank拉取到本地, 并返回对应的Work

        Args:
            k(torch.Tensor):
            v(torch.Tensor):
            overlap_stage(int): 当前的overlap stage

        Returns:
            remote_k(torch.Tensor):
            remote_k_work(Work): 通信handle, 用于等待通信完成
            remote_v(torch.Tensor):
            remote_v_work(Work): 通信handle, 用于等待通信完成

        Shape:
            - k: [num_tokens_k_local, num_heads, head_dim]
            - v: [num_tokens_v_local, num_heads, head_dim]
            - remote_k: [num_tokens_k_remote_i, num_heads, head_dim], i = 0, 1, ..., overlap_degree - 1
            - remote_v: [num_tokens_v_remote_i, num_heads, head_dim], i = 0, 1, ..., overlap_degree - 1
        """

        remote_k_work = multi_cast_collective(
            input=k,
            output=self.remote_k_buffer,
            input_split_size=self.runtime_meta.kv_dispatch_meta.input_split_size_list[
                overlap_stage
            ],
            output_split_size=self.runtime_meta.kv_dispatch_meta.output_split_size_list[
                overlap_stage
            ],
            dst_indices=self.runtime_meta.kv_dispatch_meta.dst_indices_list[
                overlap_stage
            ],
            src_indices=self.runtime_meta.kv_dispatch_meta.src_indices_list[
                overlap_stage
            ],
        )

        remote_v_work = multi_cast_collective(
            input=v,
            output=self.remote_v_buffer,
            input_split_size=self.runtime_meta.kv_dispatch_meta.input_split_size_list[
                overlap_stage
            ],
            output_split_size=self.runtime_meta.kv_dispatch_meta.output_split_size_list[
                overlap_stage
            ],
            dst_indices=self.runtime_meta.kv_dispatch_meta.dst_indices_list[
                overlap_stage
            ],
            src_indices=self.runtime_meta.kv_dispatch_meta.src_indices_list[
                overlap_stage
            ],
        )

        return (
            self.remote_k_buffer_list[overlap_stage],
            remote_k_work,
            self.remote_v_buffer_list[overlap_stage],
            remote_v_work,
        )

    def do_attn_partially(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        overlap_stage: Optional[int] = None,
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
            attn_arg = self.runtime_meta.attn_calc_meta.local_attn_arg
        else:
            attn_arg = self.runtime_meta.attn_calc_meta.remote_attn_args_list[
                overlap_stage
            ]

        # 计算attn
        out, lse = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            **asdict(attn_arg),
        )

        return out, lse

    def result_correction(
        self,
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
        # 当overlap_degree为1时, 也至少有2个lse和out, 一个是local的, 一个是remote的
        assert len(out_list) == len(lse_list)
        assert len(out_list) == self.config.overlap_degree + 1

        curr_lse = self.correct_attn_lse(lse_list[0], lse_list[1])
        curr_out = self.correct_attn_output(
            out_list[0], lse_list[0], out_list[1], lse_list[1], curr_lse
        )

        for i in range(self.config.overlap_degree):
            original_lse = curr_lse
            original_out = curr_out
            curr_lse = self.correct_attn_lse(original_lse, lse_list[i + 1])
            curr_out = self.correct_attn_output(
                original_out, original_lse, out_list[i + 1], lse_list[i + 1], curr_lse
            )

        return curr_out, curr_lse

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
            remote_k,
            remote_k_work,
            remote_v,
            remote_v_work,
        ) = dist_attn_runtime.fetch_remote_kv(k=local_k, v=local_v, overlap_stage=0)

        #########################
        # Do attn with local kv #
        #########################
        out, lse = dist_attn_runtime.do_attn_partially(
            q=local_q,
            k=local_k,
            v=local_v,
        )
        out_list.append(out)
        lse_list.append(lse)

        for overlap_stage in range(dist_attn_config.overlap_degree):
            # Wait for remote kv to be fetched
            remote_k_work.wait()
            remote_v_work.wait()

            ###########################################
            # Pre-fetch remote kv for overlap stage i #
            ###########################################
            if overlap_stage < dist_attn_config.overlap_degree - 1:
                (
                    remote_k,
                    remote_k_work,
                    remote_v,
                    remote_v_work,
                ) = dist_attn_runtime.fetch_remote_kv(
                    k=local_k, v=local_v, overlap_stage=overlap_stage + 1
                )

            ##########################
            # Do attn with remote kv #
            ##########################
            out, lse = dist_attn_runtime.do_attn_partially(
                q=local_q, k=remote_k, v=remote_v, overlap_stage=overlap_stage
            )

            out_list.append(out)
            lse_list.append(lse)

        ########################
        # Do result correction #
        ########################
        out, final_lse = dist_attn_runtime.result_correction(
            out_list=out_list,
            lse_list=lse_list,
        )

        # REVIEW(xiaowu): 需要保存哪些信息用于backward??
        ctx.save_for_backward(local_q, local_k, local_v)
        ctx.dist_attn_config = dist_attn_config
        ctx.dist_attn_runtime = dist_attn_runtime

        return out

    @staticmethod
    def backward(ctx, grad_output):
        pass


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
