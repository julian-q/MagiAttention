from typing import Any

import torch
import torch.distributed as dist
from flex_flash_attn_interface import flex_flash_attn_func
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import dffa
import dffa.testing
from dffa import init_dist_attn_runtime_mgr
from dffa.common.enum import AttnMaskType
from dffa.common.ranges import AttnRanges
from dffa.config import DistAttnConfig
from dffa.meta.collection.calc_meta import AttnArg
from dffa.testing import parameterize
from dffa.testing.dist_common import DistTestBase, with_comms


class TestDistAttnRuntimeMgr(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]
        self.gloo_groups = [
            dist.new_group(list(range(self.world_size)), backend="gloo")
            for _ in range(1)
        ]

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def gloo_group(self) -> dist.ProcessGroup:
        return self.gloo_groups[0]

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self) -> int:
        return 42

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "test_config",
        [
            # full attn with total seqlen 14k
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
                "xattn_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1222],
                    ]
                ),
                "is_causal_mapping": [False],
                "total_seqlen_q": 14336,
                "total_seqlen_k": 14336,
                "total_seqlen_xattn_k": 1222,
                "chunk_size": 512,
            },
            # varlen full attn with total seqlen 12k
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
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
                    ]
                ),
                "xattn_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1222],
                        [0, 1222],
                        [1222, 1558],
                        [1558, 1894],
                        [1894, 2230],
                        [2230, 2566],
                    ]
                ),
                "is_causal_mapping": [False] * 6,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "total_seqlen_xattn_k": 2566,
                "chunk_size": 512,
            },
        ],
    )
    def test_update_xattn_k_ranges(
        self,
        test_config: dict[str, Any],
    ):
        q_ranges: AttnRanges = test_config["q_ranges"]
        k_ranges: AttnRanges = test_config["k_ranges"]
        xattn_k_ranges: AttnRanges = test_config["xattn_k_ranges"]
        total_seqlen_q: int = test_config["total_seqlen_q"]
        total_seqlen_k: int = test_config["total_seqlen_k"]
        total_seqlen_xattn_k: int = test_config["total_seqlen_xattn_k"]
        chunk_size: int = test_config["chunk_size"]

        dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=DistAttnConfig(),
        )

        host_xattn_attn_arg: AttnArg = dist_attn_runtime_mgr.get_xattn_args(
            xattn_k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(xattn_k_ranges),
            return_host_only=True,
        )

        total_q = torch.randn(
            total_seqlen_q,
            1,
            128,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        xattn_k = torch.randn(
            total_seqlen_xattn_k,
            1,
            128,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        xattn_v = torch.randn(
            total_seqlen_xattn_k,
            1,
            128,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        dist.all_reduce(total_q, group=self.nccl_group)
        dist.all_reduce(xattn_k, group=self.nccl_group)
        dist.all_reduce(xattn_v, group=self.nccl_group)

        local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)

        total_o_ref, _ = flex_flash_attn_func(
            q=total_q,
            k=xattn_k,
            v=xattn_v,
            q_ranges=q_ranges.to_tensor(device=torch.cuda.current_device()),
            k_ranges=xattn_k_ranges.to_tensor(device=torch.cuda.current_device()),
            is_causal_mapping=torch.zeros(
                len(q_ranges), dtype=torch.bool, device=torch.cuda.current_device()
            ),
            max_seqlen_q=q_ranges.max_seqlen,
            max_seqlen_k=xattn_k_ranges.max_seqlen,
        )

        local_o, _ = flex_flash_attn_func(
            q=local_q,
            k=xattn_k,
            v=xattn_v,
            **host_xattn_attn_arg.to_ffa_args(is_bwd=False),
        )

        total_o = dist_attn_runtime_mgr.undispatch_qo(local_o)

        dffa.testing.assert_close(
            total_o,
            total_o_ref,
        )

        total_xattn_attn_arg: AttnArg = dist_attn_runtime_mgr.get_xattn_args(
            xattn_k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(xattn_k_ranges),
            return_host_only=False,
        )

        total_o, _ = flex_flash_attn_func(
            q=total_q,
            k=xattn_k,
            v=xattn_v,
            **total_xattn_attn_arg.to_ffa_args(is_bwd=False),
        )

        dffa.testing.assert_close(
            total_o,
            total_o_ref,
        )


if __name__ == "__main__":
    run_tests()
