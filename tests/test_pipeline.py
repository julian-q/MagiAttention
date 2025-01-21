from typing import Any

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import zeus
import zeus.testing
from zeus.comm.primitive import group_cast_collective, group_reduce_collective
from zeus.common.enum import AttnMaskType
from zeus.common.ranges import AttnRanges
from zeus.functional.dispatch import dispatch_func, undispatch_func
from zeus.functional.dist_attn import (
    DistFlashAttn,
    DistFlashAttnConfig,
    DistFlashAttnRuntime,
)
from zeus.meta.collection import CommMeta
from zeus.meta.solver import (
    calc_attn_meta_from_dispatch_meta,
    calc_dispatch_meta_from_qk_ranges,
)
from zeus.testing import parameterize
from zeus.testing.dist_common import DistTestBase, with_comms
from zeus.testing.precision import get_mask_from_ranges, torch_attn_ref


class TestPipeline(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self) -> int:
        return 42

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "test_case",
        [
            # full attn
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 14336],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 14336],
                    ]
                ),
                "is_causal_mapping": [False],
                "total_seqlen_q": 14336,
                "total_seqlen_k": 14336,
                "chunk_size": 512,
            },
            # varlen full attn
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [12288, 14336],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [12288, 14336],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 14336,
                "total_seqlen_k": 14336,
                "chunk_size": 512,
            },
            # varlen block causal chunk size 512
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [12288, 14336],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [0, 4096],
                        [0, 6144],
                        [0, 8192],
                        [8192, 10240],
                        [8192, 12288],
                        [12288, 14336],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 14336,
                "total_seqlen_k": 14336,
                "chunk_size": 512,
            },
            # varlen block causal chunk size 1113
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [12288, 17808],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [0, 4096],
                        [0, 6144],
                        [0, 8192],
                        [8192, 10240],
                        [8192, 12288],
                        [12288, 17808],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 17808,
                "total_seqlen_k": 17808,
                "chunk_size": 1113,
            },
        ],
    )
    def test_zeus_pipeline_block_causal_degree1(self, test_case: dict[str, Any]):
        q_ranges = test_case["q_ranges"]
        k_ranges = test_case["k_ranges"]
        is_causal_mapping = test_case["is_causal_mapping"]
        total_seqlen_q = test_case["total_seqlen_q"]
        total_seqlen_k = test_case["total_seqlen_k"]
        chunk_size = test_case["chunk_size"]

        atol = 8e-4
        rtol = 5e-2
        device = torch.cuda.current_device()
        process_group_gloo = dist.new_group(
            ranks=list(range(self.world_size)), backend="gloo"
        )

        dist_attn_config = DistFlashAttnConfig(
            num_heads=1,
            head_dim=128,
            dtype=torch.float16,
            overlap_degree=1,
            deterministic=False,
        )

        meta_q, meta_k, buckets_per_rank = calc_dispatch_meta_from_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            chunk_size=chunk_size,
            overlap_degree=dist_attn_config.overlap_degree,
            cp_size=self.world_size,
            cp_rank=self.rank,
            cp_group_nccl=self.process_group,
            cp_group_gloo=process_group_gloo,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
        )

        comm_meta, calc_meta = calc_attn_meta_from_dispatch_meta(
            dispatch_meta_q=meta_q,
            dispatch_meta_k=meta_k,
            bucket_per_rank=buckets_per_rank,
            cp_group_nccl=self.process_group,
            cp_group_gloo=process_group_gloo,
            overlap_degree=dist_attn_config.overlap_degree,
        )

        self.check_group_cast_and_group_reduce(
            comm_meta=comm_meta,
            device=device,
            atol=atol,
            rtol=rtol,
        )

        # test dist_attn
        dist_attn_runtime = DistFlashAttnRuntime.from_attn_meta(
            comm_meta=comm_meta,
            calc_meta=calc_meta,
            cp_group_nccl=self.process_group,
        )
        dist_attn = DistFlashAttn(dist_attn_config)

        total_q = torch.randn(
            total_seqlen_q,
            dist_attn_config.num_heads,
            dist_attn_config.head_dim,
            device=device,
            dtype=dist_attn_config.dtype,
            requires_grad=True,
        )
        total_k = torch.randn(
            total_seqlen_k,
            dist_attn_config.num_heads,
            dist_attn_config.head_dim,
            device=device,
            dtype=dist_attn_config.dtype,
            requires_grad=True,
        )
        total_v = torch.randn(
            total_seqlen_k,
            dist_attn_config.num_heads,
            dist_attn_config.head_dim,
            device=device,
            dtype=dist_attn_config.dtype,
            requires_grad=True,
        )
        dist.all_reduce(total_q.data, group=self.process_group)
        dist.all_reduce(total_k.data, group=self.process_group)
        dist.all_reduce(total_v.data, group=self.process_group)

        local_q, local_k, local_v = [
            dispatch_func(
                x_global=x_global,
                meta=x_meta,
                seq_dim=0,
            )
            for x_global, x_meta in zip(
                (total_q, total_k, total_v), (meta_q, meta_k, meta_k)
            )
        ]
        local_out = dist_attn(local_q, local_k, local_v, dist_attn_runtime)
        total_out = undispatch_func(
            x_local=local_out,
            meta=meta_q,
            seq_dim=0,
        )

        grad_total_out = torch.randn_like(total_out).detach()
        dist.all_reduce(grad_total_out.data, group=self.process_group)
        total_out.backward(grad_total_out)
        grad_total_q, grad_total_k, grad_total_v = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )
        total_q.grad, total_k.grad, total_v.grad = (
            total_q.grad.fill_(0),
            total_k.grad.fill_(0),
            total_v.grad.fill_(0),
        )

        mask = get_mask_from_ranges(
            q_ranges=q_ranges.to_naive_ranges(),
            k_ranges=k_ranges.to_naive_ranges(),
            q_len=total_seqlen_q,
            k_len=total_seqlen_k,
            is_causal_mapping=is_causal_mapping,
        )
        total_out_ref_high_precision = torch_attn_ref(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            high_precision=True,
        )
        total_out_ref_high_precision.backward(grad_total_out)
        (
            grad_total_q_ref_high_precision,
            grad_total_k_ref_high_precision,
            grad_total_v_ref_high_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )
        total_q.grad, total_k.grad, total_v.grad = (
            total_q.grad.fill_(0),
            total_k.grad.fill_(0),
            total_v.grad.fill_(0),
        )
        total_out_ref_low_precision = torch_attn_ref(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            high_precision=False,
        )
        total_out_ref_low_precision.backward(grad_total_out)
        (
            grad_total_q_ref_low_precision,
            grad_total_k_ref_low_precision,
            grad_total_v_ref_low_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )
        total_q.grad, total_k.grad, total_v.grad = None, None, None

        assert (total_out - total_out_ref_high_precision).abs().max().item() <= 2 * (
            total_out_ref_low_precision - total_out_ref_high_precision
        ).abs().max().item(), (
            f"{(total_out - total_out_ref_high_precision).abs().max().item()=}\n"
        )
        f"{(total_out_ref_low_precision - total_out_ref_high_precision).abs().max().item()=}"
        assert (
            grad_total_q - grad_total_q_ref_high_precision
        ).abs().max().item() <= 2 * (
            grad_total_q_ref_low_precision - grad_total_q_ref_high_precision
        ).abs().max().item(), (
            f"{(grad_total_q - grad_total_q_ref_high_precision).abs().max().item()=}\n"
        )
        f"{(grad_total_q_ref_low_precision - grad_total_q_ref_high_precision).abs().max().item()=}"
        assert (
            grad_total_k - grad_total_k_ref_high_precision
        ).abs().max().item() <= 2 * (
            grad_total_k_ref_low_precision - grad_total_k_ref_high_precision
        ).abs().max().item(), (
            f"{(grad_total_k - grad_total_k_ref_high_precision).abs().max().item()=}\n"
        )
        f"{(grad_total_k_ref_low_precision - grad_total_k_ref_high_precision).abs().max().item()=}"
        assert (
            grad_total_v - grad_total_v_ref_high_precision
        ).abs().max().item() <= 2 * (
            grad_total_v_ref_low_precision - grad_total_v_ref_high_precision
        ).abs().max().item(), (
            f"{(grad_total_v - grad_total_v_ref_high_precision).abs().max().item()=}\n"
        )
        f"{(grad_total_v_ref_low_precision - grad_total_v_ref_high_precision).abs().max().item()=}"

        zeus.testing.assert_close(
            total_out,
            total_out_ref_high_precision,
            atol=atol,
            rtol=rtol,
            mismatch_threshold=0.01,
        )
        zeus.testing.assert_close(
            grad_total_q,
            grad_total_q_ref_high_precision,
            atol=atol,
            rtol=rtol,
            mismatch_threshold=0.17,
        )
        zeus.testing.assert_close(
            grad_total_k,
            grad_total_k_ref_high_precision,
            atol=atol,
            rtol=rtol,
            mismatch_threshold=0.17,
        )
        zeus.testing.assert_close(
            grad_total_v,
            grad_total_v_ref_high_precision,
            atol=atol,
            rtol=rtol,
            mismatch_threshold=0.03,
        )

    def check_group_cast_and_group_reduce(
        self,
        comm_meta: CommMeta,
        device: torch.device,
        atol: float,
        rtol: float,
    ):
        # test group_cast和group_reduce的对称性
        group_cast_collective_args = comm_meta.group_cast_collective_args_list[0]
        input_ttk = sum(group_cast_collective_args.input_split_size_list)
        test_input = (
            torch.randn(input_ttk, device=device, dtype=torch.float32) * 10**self.rank
        )
        output_ttk = sum(group_cast_collective_args.output_split_size_list)
        test_output = torch.zeros(output_ttk, device=device, dtype=torch.float32)
        ans = test_input
        ans = list(torch.split(ans, group_cast_collective_args.input_split_size_list))
        ans = torch.cat(
            [
                ans[i] * (1 + len(group_cast_collective_args.dst_indices_list[i]))
                for i in range(len(group_cast_collective_args.dst_indices_list))
            ]
        )
        work, post_process_fn = group_cast_collective(
            input=test_input,
            output=test_output,
            input_split_size_list=group_cast_collective_args.input_split_size_list,
            output_split_size_list=group_cast_collective_args.output_split_size_list,
            dst_indices_list=group_cast_collective_args.dst_indices_list,
            src_index_list=group_cast_collective_args.src_index_list,
            group=self.process_group,
            async_op=True,
        )
        work.wait()
        test_output = post_process_fn(test_output)
        work, post_process_fn = group_reduce_collective(
            input=test_output,
            output=test_input,
            input_split_size_list=group_cast_collective_args.output_split_size_list,
            output_split_size_list=group_cast_collective_args.input_split_size_list,
            dst_index_list=group_cast_collective_args.src_index_list,
            src_indices_list=group_cast_collective_args.dst_indices_list,
            group=self.process_group,
            async_op=True,
        )
        work.wait()
        test_input = post_process_fn(test_input)
        torch.testing.assert_close(test_input, ans, atol=atol, rtol=rtol)


if __name__ == "__main__":
    run_tests()
