import random

import torch
import torch.distributed
import torch.distributed as dist
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.nn.functional import scaled_dot_product_attention
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from dffa.common.enum import AttnMaskType
from dffa.common.ranges import AttnRanges
from dffa.testing.dist_common import DistTestBase, with_comms

# isort: split
from exps.dist_attn.baselines.ring_attn import TERingAttnWithKVP2P


class TestTERingAttnWithKVP2P(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self) -> int:
        return 42

    def init_tensor(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        valid_seqlens = []
        valid_seqlens_padded = []
        attention_mask = []

        seqlen = random.randint(10, 128)
        seqlen_padded = seqlen + random.randint(0, 10)
        valid_seqlens.append(seqlen)
        valid_seqlens_padded.append(seqlen_padded)
        attention_mask.extend([1] * seqlen)
        attention_mask.extend([0] * (seqlen_padded - seqlen))

        valid_seqlens = [0] + valid_seqlens
        valid_seqlens_padded = [0] + valid_seqlens_padded
        return valid_seqlens, valid_seqlens_padded, attention_mask

    def broadcast(self, tensor) -> torch.Tensor:
        dist.broadcast(tensor, src=0, group=self.process_group)
        return tensor

    def get_global_grad(
        self, dist_attn, grad, rank, attention_mask, qkv_format, tensor_type
    ) -> torch.Tensor:
        te_global_grad = dist_attn.dispatch(
            grad,
            rank,
            self.world_size,
            self.process_group,
            attention_mask=attention_mask,
            tensor_type=tensor_type,
        )
        te_global_grad = dist_attn.undispatch(
            te_global_grad,
            rank,
            self.world_size,
            self.process_group,
            qkv_format=qkv_format,
            tensor_type=tensor_type,
        )
        return te_global_grad

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_full_attn(self):
        cp_global_ranks = list(range(self.world_size))
        rank = dist.get_rank(self.process_group)

        valid_seqlens, valid_seqlens_padded, attention_mask = self.init_tensor()
        device = torch.cuda.current_device()

        total_seq, nh, hd = valid_seqlens_padded[-1], 1, 128
        query_init = torch.randn(
            total_seq, nh, hd, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        key_init = torch.randn(
            total_seq, nh, hd, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        value_init = torch.randn(
            total_seq, nh, hd, device=device, dtype=torch.bfloat16, requires_grad=True
        )

        with torch.no_grad():
            global_q = self.broadcast(query_init)
            global_k = self.broadcast(key_init)
            global_v = self.broadcast(value_init)

        # initialize cu_seqlens and mask
        cu_seqlens = valid_seqlens
        attention_mask = torch.tensor(attention_mask, dtype=torch.int32, device=device)
        q_ranges = AttnRanges.from_cu_seqlens(cu_seqlens, cu_seqlens[-1])
        kv_ranges = AttnRanges.from_cu_seqlens(cu_seqlens, cu_seqlens[-1])
        attention_mask_type = AttnMaskType.FULL
        cp_stream = torch.cuda.Stream()
        max_seqlens = 128

        te_ring_attn_with_KV_P2P = TERingAttnWithKVP2P(
            q_ranges,
            kv_ranges,
            attention_mask_type,
            max_seqlens,
            max_seqlens,
            0.5,
            False,
        )

        local_q = te_ring_attn_with_KV_P2P.dispatch(
            global_q,
            rank,
            self.world_size,
            self.process_group,
            attention_mask=attention_mask,
            tensor_type="q",
        )
        local_k = te_ring_attn_with_KV_P2P.dispatch(
            global_k,
            rank,
            self.world_size,
            self.process_group,
            attention_mask=attention_mask,
            tensor_type="k",
        )
        local_v = te_ring_attn_with_KV_P2P.dispatch(
            global_v,
            rank,
            self.world_size,
            self.process_group,
            attention_mask=attention_mask,
            tensor_type="v",
        )

        te_ring_attn_with_KV_P2P_output, _ = te_ring_attn_with_KV_P2P.apply_attn(
            local_q,
            local_k,
            local_v,
            q_ranges,
            kv_ranges,
            attention_mask_type,
            max_seqlens,
            max_seqlens,
            0.5,
            False,
            cp_group=self.process_group,
            cp_global_ranks=cp_global_ranks,
            cp_stream=cp_stream,
        )
        te_total_output = te_ring_attn_with_KV_P2P.undispatch(
            te_ring_attn_with_KV_P2P_output,
            rank,
            self.world_size,
            self.process_group,
            qkv_format="thd",
            tensor_type="q",
        )

        te_total_output.sum().backward()
        te_global_grad_q = self.get_global_grad(
            te_ring_attn_with_KV_P2P, global_q.grad, rank, attention_mask, "thd", "q"
        )
        te_global_grad_k = self.get_global_grad(
            te_ring_attn_with_KV_P2P, global_k.grad, rank, attention_mask, "thd", "k"
        )
        te_global_grad_v = self.get_global_grad(
            te_ring_attn_with_KV_P2P, global_v.grad, rank, attention_mask, "thd", "v"
        )

        if rank == 0:
            # Flash Attention
            # fa_total_output = torch.zeros_like(global_q)
            attention_mask_bool = attention_mask.bool()
            query_fa = global_q[attention_mask_bool]
            key_fa = global_k[attention_mask_bool]
            value_fa = global_v[attention_mask_bool]
            valid_seqlens_tensor = torch.tensor(
                valid_seqlens, dtype=torch.int32, device=global_q.device
            )
            fa_total_output = flash_attn_varlen_func(
                query_fa,
                key_fa,
                value_fa,
                valid_seqlens_tensor,
                valid_seqlens_tensor,
                max_seqlens,
                max_seqlens,
                0,
                0.5,
                causal=False,
                window_size=(-1, -1),
            )

            fa_total_output.sum().backward()
            fa_global_grad_q, fa_global_grad_k, fa_global_grad_v = (
                global_q.grad,
                global_k.grad,
                global_v.grad,
            )
            global_q.grad, global_k.grad, global_v.grad = None, None, None

            # SDPA
            sdpa_total_output = scaled_dot_product_attention(
                rearrange(global_q, "t h d -> 1 h t d"),
                rearrange(global_k, "t h d -> 1 h t d"),
                rearrange(global_v, "t h d -> 1 h t d"),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            sdpa_total_output = rearrange(sdpa_total_output, "1 h t d -> t h d")
            sdpa_total_output.sum().backward()
            sdpa_global_grad_q, sdpa_global_grad_k, sdpa_global_grad_v = (
                global_q.grad,
                global_k.grad,
                global_v.grad,
            )
            global_q.grad, global_k.grad, global_v.grad = None, None, None

            te_total_output = te_total_output[attention_mask_bool]
            sdpa_total_output = sdpa_total_output[attention_mask_bool]

            # Compare diff
            te_fa_total_out_max = torch.max(
                torch.abs(te_total_output - fa_total_output)
            )
            te_fa_local_grad_q_max = torch.max(
                torch.abs(te_global_grad_q - fa_global_grad_q)
            )
            te_fa_local_grad_k_max = torch.max(
                torch.abs(te_global_grad_k - fa_global_grad_k)
            )
            te_fa_local_grad_v_max = torch.max(
                torch.abs(te_global_grad_v - fa_global_grad_v)
            )

            fa_sdpa_total_out_max = torch.max(
                torch.abs(fa_total_output - sdpa_total_output)
            )
            fa_sdpa_local_grad_q_max = torch.max(
                torch.abs(fa_global_grad_q - sdpa_global_grad_q)
            )
            fa_sdpa_local_grad_k_max = torch.max(
                torch.abs(fa_global_grad_k - sdpa_global_grad_k)
            )
            fa_sdpa_local_grad_v_max = torch.max(
                torch.abs(fa_global_grad_v - sdpa_global_grad_v)
            )

            diff_total_output = te_fa_total_out_max <= 2 * fa_sdpa_total_out_max
            diff_grad_q = te_fa_local_grad_q_max <= 2 * fa_sdpa_local_grad_q_max
            diff_grad_k = te_fa_local_grad_k_max <= 2 * fa_sdpa_local_grad_k_max
            diff_grad_v = te_fa_local_grad_v_max <= 2 * fa_sdpa_local_grad_v_max

            print("\nTest Result:", flush=True)
            print(f"diff_total_output: {diff_total_output.item()}", flush=True)
            print(f"diff_grad_q: {diff_grad_q.item()}", flush=True)
            print(f"diff_grad_k: {diff_grad_k.item()}", flush=True)
            print(f"diff_grad_v: {diff_grad_v.item()}", flush=True)


if __name__ == "__main__":
    run_tests()
