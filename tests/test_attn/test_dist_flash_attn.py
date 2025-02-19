import torch
import torch.distributed
import torch.distributed as dist
from einops import rearrange
from torch.distributed.nn.functional import all_gather
from torch.nn.functional import scaled_dot_product_attention
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from zeus.common.config import DistFlashAttnConfig, OverlapConfig
from zeus.common.enum import AttnOverlapMode
from zeus.functional.dist_attn import DistFlashAttn, DistFlashAttnRuntime
from zeus.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from zeus.meta.collection.comm_meta import CommMeta, GroupCastCollectiveArg
from zeus.testing.dist_common import DistTestBase, with_comms


# TODO: add more unitest for dist ffa
class TestDistFlashAttn(DistTestBase):
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
    def test_full_attn(self):
        device = torch.cuda.current_device()

        # NOTE: this test does not include overlap solver
        # so this overlap config is a dummy one
        # in which only overlap_degree is used
        overlap_config = OverlapConfig(
            mode=AttnOverlapMode.STATIC,
            degree=1,  # TODO: test overlap degree > 1
            max_num_chunks=1,
        )

        attn_config = DistFlashAttnConfig(
            num_heads=1,
            head_dim=128,
            dtype=torch.bfloat16,
            overlap_config=overlap_config,
            deterministic=False,
        )
        dist_attn = DistFlashAttn(attn_config)

        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=AttnArg(
                q_ranges=[[0, 128]],
                k_ranges=[[0, 128]],
                is_causal_mapping=[False],
                max_seqlen_q=128,
                max_seqlen_k=128,
            ),
            remote_attn_args_list=[
                AttnArg(
                    q_ranges=[[0, 128]],
                    k_ranges=[[0, 128 * 3]],
                    is_causal_mapping=[False],
                    max_seqlen_q=128,
                    max_seqlen_k=128 * 3,
                ),
            ],
        )

        comm_meta = CommMeta(
            num_remote_tokens_per_overlap_stage=[128 * 3],
            group_cast_collective_args_list=[
                GroupCastCollectiveArg(
                    input_split_size_list=[128],
                    output_split_size_list=[128, 128, 128],
                    dst_indices_list=[
                        [rank for rank in range(self.world_size) if rank != self.rank]
                    ],
                    src_index_list=[
                        rank for rank in range(self.world_size) if rank != self.rank
                    ],
                )
            ],
        )

        dist_attn_runtime = DistFlashAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=attn_calc_meta,
            cp_group_nccl=self.process_group,
        )

        local_q = torch.randn(
            128, 1, 128, device=device, dtype=attn_config.dtype, requires_grad=True
        )
        local_k = torch.randn(
            128, 1, 128, device=device, dtype=attn_config.dtype, requires_grad=True
        )
        local_v = torch.randn(
            128, 1, 128, device=device, dtype=attn_config.dtype, requires_grad=True
        )

        local_out = dist_attn(local_q, local_k, local_v, dist_attn_runtime)
        total_out = torch.cat(all_gather(local_out, group=self.process_group), dim=0)

        grad_total_out = torch.randn_like(total_out)
        total_out.backward(grad_total_out)
        local_grad_q, local_grad_k, local_grad_v = (
            local_q.grad,
            local_k.grad,
            local_v.grad,
        )
        local_q.grad, local_k.grad, local_v.grad = None, None, None

        total_q = torch.cat(all_gather(local_q, group=self.process_group), dim=0)
        total_k = torch.cat(all_gather(local_k, group=self.process_group), dim=0)
        total_v = torch.cat(all_gather(local_v, group=self.process_group), dim=0)

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
