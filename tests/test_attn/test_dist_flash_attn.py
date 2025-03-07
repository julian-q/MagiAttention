import torch
import torch.distributed
import torch.distributed as dist
from einops import rearrange
from torch.distributed.nn.functional import all_gather
from torch.nn.functional import scaled_dot_product_attention
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from zeus.common.ranges import AttnRanges
from zeus.functional.dist_attn import DistFlashAttnRuntime, dist_attn_func
from zeus.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from zeus.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from zeus.testing import parameterize
from zeus.testing.dist_common import DistTestBase, with_comms


# TODO: add more unitest for dist ffa
class TestDistFlashAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self) -> int:
        return 42

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_full_attn(self, dtype):
        device = torch.cuda.current_device()

        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=AttnArg(
                q_ranges=AttnRanges.from_ranges([[0, 128]]),
                k_ranges=AttnRanges.from_ranges([[0, 128]]),
                is_causal_mapping=[False],
                shard_seqlen_q=128,
                total_area=128 * 128,
            ),
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=AttnRanges.from_ranges([[0, 128]]),
                    k_ranges=AttnRanges.from_ranges([[0, 128 * 3]]),
                    is_causal_mapping=[False],
                    shard_seqlen_q=128,
                    total_area=128 * 128 * 3,
                ),
            ],
        )

        comm_meta = CommMeta(
            num_remote_tokens_per_stage=[128 * 3],
            group_collective_args_list=[
                GroupCollectiveArg(
                    input_split_size_list=[128],
                    output_split_size_list=[128, 128, 128],
                    dst_indices_list=[
                        [rank for rank in range(self.world_size) if rank != self.rank]
                    ],
                    src_index_list=[
                        rank for rank in range(self.world_size) if rank != self.rank
                    ],
                    world_size=self.world_size,
                )
            ],
        )

        dist_attn_runtime = DistFlashAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=attn_calc_meta,
            cp_group_kv=self.nccl_groups[0],
            cp_group_dkv=self.nccl_groups[1],
            deterministic=False,
        )

        local_q = torch.randn(
            128, 1, 128, device=device, dtype=dtype, requires_grad=True
        )
        local_k = torch.randn(
            128, 1, 128, device=device, dtype=dtype, requires_grad=True
        )
        local_v = torch.randn(
            128, 1, 128, device=device, dtype=dtype, requires_grad=True
        )

        local_out, _ = dist_attn_func(local_q, local_k, local_v, dist_attn_runtime)
        total_out = torch.cat(all_gather(local_out, group=self.nccl_group), dim=0)

        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        local_grad_q, local_grad_k, local_grad_v = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad, local_k.grad, local_v.grad = None, None, None

        total_q = torch.cat(all_gather(local_q, group=self.nccl_group), dim=0)
        total_k = torch.cat(all_gather(local_k, group=self.nccl_group), dim=0)
        total_v = torch.cat(all_gather(local_v, group=self.nccl_group), dim=0)

        total_out_ref = scaled_dot_product_attention(
            rearrange(total_q, "t h d -> 1 h t d"),
            rearrange(total_k, "t h d -> 1 h t d"),
            rearrange(total_v, "t h d -> 1 h t d"),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        total_out_ref = rearrange(total_out_ref, "1 h t d -> t h d")
        total_out_ref.backward(grad_total_out)
        local_grad_q_ref, local_grad_k_ref, local_grad_v_ref = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad, local_k.grad, local_v.grad = None, None, None

        torch.testing.assert_close(total_out, total_out_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(local_grad_q, local_grad_q_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(local_grad_k, local_grad_k_ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(local_grad_v, local_grad_v_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
