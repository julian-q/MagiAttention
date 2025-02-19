import os
from typing import Any

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import zeus
import zeus.testing
from zeus.comm.primitive import group_cast_collective, group_reduce_collective
from zeus.common.config import DistFlashAttnConfig, OverlapConfig
from zeus.common.enum import AttnMaskType, AttnOverlapMode, OverlapAlgorithm
from zeus.common.ranges import AttnRanges
from zeus.functional.dispatch import dispatch_func, undispatch_func
from zeus.functional.dist_attn import DistFlashAttn, DistFlashAttnRuntime
from zeus.meta.collection import CommMeta, DispatchMeta
from zeus.meta.container import AttnBucket
from zeus.meta.solver import (
    calc_attn_meta_from_dispatch_meta,
    calc_dispatch_meta_from_qk_ranges,
)
from zeus.testing import parameterize
from zeus.testing.dist_common import DistTestBase, with_comms
from zeus.testing.precision import (
    EPSILON,
    calc_inf_norm,
    get_mask_from_ranges,
    torch_attn_ref,
)

# tell if using profile mode
profile_mode = os.environ.get("ZEUS_UNITEST_PROFILE_MODE", "0") == "1"

# NOTE: enable sanity check if not using profile mode
if not profile_mode:
    os.environ["ZEUS_SANITY_CHECK"] = "1"


PROFILE_ONLY = "profile_only"
NAME = "name"


IB_BANDWIDTH = 50e9  # 500 GB/s, single-end

# H100 spec: https://www.nvidia.com/en-us/data-center/h100/
H100_TFLOPS_16 = 989.5e12  # 989 teraFLOPS
H100_NVLINK_BANDWIDTH = 450e9  # 450 GB/s, single-end

# H800 spec: https://chaoqing-i.com/upload/20231128/NVIDIA%20H800%20GPU%20Datasheet.pdf
H800_TFLOPS_16 = 989.5e12  # 989 teraFLOPS
H800_NVLINK_BANDWIDTH = 200e9  # 200 GB/s, single-end

# A100 spec: https://www.nvidia.com/en-us/data-center/a100/
A100_TFLOPS_16 = 312e12  # 312 teraFLOPS
A100_NVLINK_BANDWIDTH = 300e9  # 300 GB/s, single-end


# assuming that:
#   num_heads (nh) = 1, head_dim (hd) = 128
#   mfu = 0.5, bwu = 0.6
#   cp = 4, a2a_corr_factor = (cp-1)/cp = 0.75
#   unit: μs
NUM_HEADS = 1
HEAD_DIM = 128
MFU = 0.5
BWU = 0.6
A2A_CORR_FACTOR = 0.75
SEC_RATIO = 1e6  # 1s = 1e6 μs

# formula:
#   calc cost factor = 2 * 2 * nh * hd / TFLOPS / mfu * sec_ratio
#   comm cost factor = 2 * nh * hd / BANDWIDTH / a2a_corr_factor / bwu * sec_ratio
# then:
CALC_COST_FACTOR = 2 * 2 * NUM_HEADS * HEAD_DIM / H800_TFLOPS_16 / MFU * SEC_RATIO
INTRA_NODE_COMM_COST_FACTOR = (
    2 * NUM_HEADS * HEAD_DIM / H800_NVLINK_BANDWIDTH / A2A_CORR_FACTOR / BWU * SEC_RATIO
)
INTER_NODE_COMM_COST_FACTOR = (
    2 * NUM_HEADS * HEAD_DIM / IB_BANDWIDTH / A2A_CORR_FACTOR / BWU * SEC_RATIO
)


class TestPipeline(DistTestBase):
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
        # TODO:
        # 1. test more diverse and complicated attn mask
        "attn_config",
        [
            # full attn with total seqlen 14k
            {
                NAME: "full_attn_14k",
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
            # varlen full attn with total seqlen 14k
            {
                NAME: "varlen_full_attn_14k",
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
            # varlen block causal with total seqlen 14k
            {
                NAME: "varlen_block_causal_14k",
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
            # varlen block causal with total seqlen 17k
            {
                NAME: "varlen_block_causal_17k",
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
            # full attn with total seqlen 12k
            {
                NAME: "full_attn_12k",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 12288],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [0, 2048],
                    ]
                ),
                "is_causal_mapping": [False],
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "chunk_size": 512,
            },
            # NOTE: profile only case
            # full attn with total seqlen 140k
            # {
            #     PROFILE_ONLY: True,
            #     NAME: "full_attn_140k",
            #     "q_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 143360],
            #         ]
            #     ),
            #     "k_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 143360],
            #         ]
            #     ),
            #     "is_causal_mapping": [False],
            #     "total_seqlen_q": 143360,
            #     "total_seqlen_k": 143360,
            #     "chunk_size": 1024,
            # },
            # NOTE: profile only case
            # varlen block causal with total seqlen 144k
            {
                PROFILE_ONLY: True,
                NAME: "varlen_block_causal_144k",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 20480],
                        [20480, 40960],
                        [40960, 61440],
                        [61440, 81920],
                        [81920, 102400],
                        [102400, 122880],
                        [122880, 147456],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 20480],
                        [0, 40960],
                        [0, 61440],
                        [0, 81920],
                        [81920, 102400],
                        [81920, 122880],
                        [122880, 147456],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 147456,
                "total_seqlen_k": 147456,
                "chunk_size": 4096,
            },
        ],
    )
    @parameterize(
        # TODO:
        #   1. test non-trivial algorithms
        #   2. profile real comm/calc factors
        "overlap_config",
        [
            # disable multi-stage overlap to roll back to the original code
            {
                NAME: "disable_mso",
                "enable": False,
            },
            # static, overlap degree = 1, max num chunks = 1
            {
                NAME: "static_od1_cz+inf",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 1,
                "max_num_chunks": 1,
                "alg": OverlapAlgorithm.UNIFORM,
                "alg_kwargs": dict(
                    random_costs=False,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 1, min chunk size = 1023
            {
                NAME: "static_od1_cz1023",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 1,
                "min_chunk_size": 1023,
                "max_num_chunks": 64,
                "alg": OverlapAlgorithm.UNIFORM,
                "alg_kwargs": dict(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 2, min chunk size = 513
            {
                NAME: "static_od2_cz513",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 2,
                "min_chunk_size": 513,
                "max_num_chunks": 64,
                "alg": OverlapAlgorithm.UNIFORM,
                "alg_kwargs": dict(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 4, min chunk size = 253
            {
                NAME: "static_od4_cz253",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 4,
                "min_chunk_size": 253,
                "max_num_chunks": 64,
                "alg": OverlapAlgorithm.UNIFORM,
                "alg_kwargs": dict(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # dynamic, min chunk size = 256, no max overlap degree limit
            {
                NAME: "dynamic_cz256",
                "enable": True,
                "mode": AttnOverlapMode.DYNAMIC,
                "degree": None,
                "dynamic_max_degree": None,
                "min_chunk_size": 256,
                "max_num_chunks": 64,
                "alg": OverlapAlgorithm.UNIFORM,
                "alg_kwargs": dict(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # NOTE: profile only case
            # static, overlap degree = 4, min chunk size = 512, max num chunks = 64
            {
                PROFILE_ONLY: True,
                NAME: "static_d4",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 4,
                "min_chunk_size": 512,
                "max_num_chunks": 64,
                "alg": OverlapAlgorithm.UNIFORM,
                "alg_kwargs": dict(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # NOTE: profile only case
            # dynamic, min chunk size = 512, max num chunks = 64, max overlap degree = 8
            # {
            #     PROFILE_ONLY: True,
            #     NAME: "dynamic_md8",
            #     "enable": True,
            #     "mode": AttnOverlapMode.DYNAMIC,
            #     "degree": None,
            #     "dynamic_max_degree": 8,
            #     "min_chunk_size": 512,
            #     "max_num_chunks": 64,
            #     "alg": OverlapAlgorithm.UNIFORM,
            #     "alg_kwargs": dict(
            #         random_costs=True,
            #         random_seed=42,
            #     ),
            #     "calc_cost_factor": CALC_COST_FACTOR,
            #     "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            # },
        ],
    )
    @parameterize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_pipeline(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        dtype: torch.dtype,
    ):
        # HACK: (optional) set the same torch manual random seed for each test case
        # self._set_random_seed()

        # HACK: filter to test a single test case
        # if dtype != torch.float16:
        #     return
        # if attn_config[NAME] != "varlen_block_causal_17k":
        #     return
        # if overlap_config[NAME] != "dynamic_cz256":
        #     return

        # -----    switch mode   ---- #

        if profile_mode:
            prof_iters, prof_start_iter, prof_end_iter = 10, 4, 6
        else:
            prof_iters, prof_start_iter, prof_end_iter = 1, -1, -1

        if profile_mode ^ attn_config.get(PROFILE_ONLY, False):
            return
        if profile_mode ^ overlap_config.get(PROFILE_ONLY, False):
            return

        # -----    construct test case name   ---- #

        assert (
            NAME in attn_config and NAME in overlap_config
        ), f"{attn_config=}\n{overlap_config=}"
        test_case = f"attn_config=[{attn_config[NAME]}] x overlap_config=[{overlap_config[NAME]}] x dtype=[{dtype}]"

        # -----    contruct config from test cases   ---- #

        q_ranges: AttnRanges = attn_config["q_ranges"]
        k_ranges: AttnRanges = attn_config["k_ranges"]
        is_causal_mapping: list[bool] = attn_config["is_causal_mapping"]
        total_seqlen_q: int = attn_config["total_seqlen_q"]
        total_seqlen_k: int = attn_config["total_seqlen_k"]
        chunk_size: int = attn_config["chunk_size"]

        device = torch.cuda.current_device()

        overlap_config = OverlapConfig(
            **{k: v for k, v in overlap_config.items() if k not in (NAME, PROFILE_ONLY)}
        )  # type: ignore

        dist_attn_config = DistFlashAttnConfig(
            num_heads=1,
            head_dim=128,
            dtype=dtype,
            deterministic=False,
            overlap_config=overlap_config,  # type: ignore
        )

        # -----    run pipeline test   ---- #

        for iter in range(prof_iters):
            # -----    profile control if using profile mode   ---- #

            if profile_mode:
                if self.rank == 0 and iter == prof_start_iter:
                    torch.cuda.profiler.start()
                if self.rank == 0 and iter == prof_end_iter:
                    torch.cuda.profiler.stop()

            # -----    barrier at the beginning of each iteration   ---- #

            dist.barrier()
            torch.cuda.synchronize()

            # -----   calc dispatch meta   ---- #

            meta_q, meta_k, buckets_per_rank = calc_dispatch_meta_from_qk_ranges(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                chunk_size=chunk_size,
                cp_size=self.world_size,
                cp_rank=self.rank,
                cp_group_nccl=self.nccl_group,
                cp_group_gloo=self.gloo_group,
                is_same_source=True,
                is_q_permutable=True,
                is_k_permutable=True,
            )

            # -----   calc comm/calc attn meta   ---- #

            comm_meta, calc_meta = calc_attn_meta_from_dispatch_meta(
                dispatch_meta_q=meta_q,
                dispatch_meta_k=meta_k,
                bucket_per_rank=buckets_per_rank,
                cp_group_nccl=self.nccl_group,
                cp_group_gloo=self.gloo_group,
                overlap_config=dist_attn_config.overlap_config,
            )

            # -----   sanity check about group cast/reduce   ---- #

            if not profile_mode:
                self.check_group_cast_and_group_reduce(
                    comm_meta=comm_meta,
                    device=device,
                )

            # -----   init dist attn   ---- #

            dist_attn_runtime = DistFlashAttnRuntime(
                comm_meta=comm_meta,
                calc_meta=calc_meta,
                cp_group_kv=self.nccl_groups[0],
                cp_group_dkv=self.nccl_groups[1],
            )
            dist_attn = DistFlashAttn(dist_attn_config)

            # -----   init global qkv   ---- #

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
            dist.all_reduce(total_q.data, group=self.nccl_group)
            dist.all_reduce(total_k.data, group=self.nccl_group)
            dist.all_reduce(total_v.data, group=self.nccl_group)

            # -----   dispatch global qkv to local qkv   ---- #

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

            # -----   run dist attn forward on local qkv for local o   ---- #

            local_out = dist_attn(local_q, local_k, local_v, dist_attn_runtime)

            # -----   undispatch local o to global o   ---- #

            total_out = undispatch_func(
                x_local=local_out,
                meta=meta_q,
                seq_dim=0,
            )

            # -----   run backward   ---- #

            grad_total_out = torch.randn_like(total_out).detach()
            dist.all_reduce(grad_total_out.data, group=self.nccl_group)
            total_out.backward(grad_total_out)
            grad_total_q, grad_total_k, grad_total_v = (
                total_q.grad,
                total_k.grad,
                total_v.grad,
            )

            # -----   assert close if not using profile mode   ---- #

            if not profile_mode:
                # -----   assert close to the one w/o mso   ---- #

                self.assert_close_to_ref_wo_mso(
                    dist_attn_config=dist_attn_config,
                    meta_q=meta_q,
                    meta_k=meta_k,
                    buckets_per_rank=buckets_per_rank,
                    cp_group_nccl=self.nccl_group,
                    cp_group_gloo=self.gloo_group,
                    device=device,
                    total_q=total_q,
                    total_k=total_k,
                    total_v=total_v,
                    total_out=total_out,
                    grad_total_out=grad_total_out,
                    grad_total_q=grad_total_q,
                    grad_total_k=grad_total_k,
                    grad_total_v=grad_total_v,
                    dtype=dtype,
                    test_case=test_case,
                )

                # -----   assert close to torch ref   ---- #

                self.assert_close_to_torch_ref(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    is_causal_mapping=is_causal_mapping,
                    total_seqlen_q=total_seqlen_q,
                    total_seqlen_k=total_seqlen_k,
                    total_q=total_q,
                    total_k=total_k,
                    total_v=total_v,
                    total_out=total_out,
                    grad_total_q=grad_total_q,
                    grad_total_k=grad_total_k,
                    grad_total_v=grad_total_v,
                    grad_total_out=grad_total_out,
                    dtype=dtype,
                    test_case=test_case,
                )

    def assert_close_to_ref_wo_mso(
        self,
        dist_attn_config: DistFlashAttnConfig,
        meta_q: DispatchMeta,
        meta_k: DispatchMeta,
        buckets_per_rank: list[AttnBucket],
        cp_group_nccl: dist.ProcessGroup,
        cp_group_gloo: dist.ProcessGroup,
        device: torch.device,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_out: torch.Tensor,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        test_case: str = "",
    ):
        # -----   customize tolerance threshold  ---- #

        o_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        o_rtol = {torch.bfloat16: 0.015, torch.float16: 0.005}.get(dtype, 0.005)
        o_thres = {torch.bfloat16: 0.02, torch.float16: 0.01}.get(dtype, 0.01)

        dq_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        dq_rtol = {torch.bfloat16: 0.16, torch.float16: 0.12}.get(dtype, 0.12)
        dq_thres = {torch.bfloat16: 0.16, torch.float16: 0.12}.get(dtype, 0.12)

        dk_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        dk_rtol = {torch.bfloat16: 0.1, torch.float16: 0.05}.get(dtype, 0.05)
        dk_thres = {torch.bfloat16: 0.06, torch.float16: 0.03}.get(dtype, 0.03)

        dv_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        dv_rtol = {torch.bfloat16: 0.015, torch.float16: 0.005}.get(dtype, 0.005)
        dv_thres = {torch.bfloat16: 0.02, torch.float16: 0.01}.get(dtype, 0.01)

        # -----   replace the overlap config w/o mso   ---- #

        overlap_config_wo_mso = OverlapConfig(enable=False)
        dist_attn_config_wo_mso = DistFlashAttnConfig(
            num_heads=dist_attn_config.num_heads,
            head_dim=dist_attn_config.head_dim,
            dtype=dist_attn_config.dtype,
            deterministic=dist_attn_config.deterministic,
            overlap_config=overlap_config_wo_mso,
        )

        # -----   calc comm/calc attn meta for the ref w/o mso   ---- #

        comm_meta, calc_meta = calc_attn_meta_from_dispatch_meta(
            dispatch_meta_q=meta_q,
            dispatch_meta_k=meta_k,
            bucket_per_rank=buckets_per_rank,
            cp_group_nccl=cp_group_nccl,
            cp_group_gloo=cp_group_gloo,
            overlap_config=dist_attn_config_wo_mso.overlap_config,
        )

        # -----   sanity check about group cast/reduce for the ref w/o mso   ---- #

        self.check_group_cast_and_group_reduce(
            comm_meta=comm_meta,
            device=device,
        )

        # -----   init dist attn for the ref w/o mso  ---- #

        dist_attn_runtime = DistFlashAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=calc_meta,
            cp_group_kv=cp_group_nccl,
            cp_group_dkv=cp_group_nccl,  # use the same group as kv to make sure "safe"
        )
        dist_attn = DistFlashAttn(dist_attn_config_wo_mso)

        # -----   init global qkv by copying   ---- #

        total_q_ref = total_q.detach().clone().requires_grad_(True)
        total_k_ref = total_k.detach().clone().requires_grad_(True)
        total_v_ref = total_v.detach().clone().requires_grad_(True)

        # -----   dispatch global qkv to local qkv   ---- #

        local_q_ref, local_k_ref, local_v_ref = [
            dispatch_func(
                x_global=x_global,
                meta=x_meta,
                seq_dim=0,
            )
            for x_global, x_meta in zip(
                (total_q_ref, total_k_ref, total_v_ref), (meta_q, meta_k, meta_k)
            )
        ]

        # -----   run dist attn forward on local qkv for local o for the ref w/o mso   ---- #

        local_out_ref = dist_attn(
            local_q_ref, local_k_ref, local_v_ref, dist_attn_runtime
        )

        # -----   undispatch local o to global o for the ref w/o mso   ---- #

        total_out_ref = undispatch_func(
            x_local=local_out_ref,
            meta=meta_q,
            seq_dim=0,
        )

        # -----   run backward for the ref w/o mso   ---- #

        grad_total_out_ref = torch.empty_like(grad_total_out.data).copy_(
            grad_total_out.data
        )
        total_out_ref.backward(grad_total_out_ref)
        grad_total_q_ref, grad_total_k_ref, grad_total_v_ref = (
            total_q_ref.grad,
            total_k_ref.grad,
            total_v_ref.grad,
        )

        # -----   assert close for fwd out   ---- #

        # torch style with atol + rtol
        zeus.testing.assert_close(
            total_out,
            total_out_ref,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_threshold=o_thres,
            test_case=f"ref_wo_mso: {test_case} => o",
        )

        # -----   assert close for bwd dq   ---- #

        # torch style with atol + rtol
        zeus.testing.assert_close(
            grad_total_q,
            grad_total_q_ref,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_threshold=dq_thres,
            test_case=f"ref_wo_mso: {test_case} => dq",
        )

        # -----   assert close for bwd dk   ---- #

        # torch style with atol + rtol
        zeus.testing.assert_close(
            grad_total_k,
            grad_total_k_ref,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_threshold=dk_thres,
            test_case=f"ref_wo_mso: {test_case} => dk",
        )

        # -----   assert close for bwd dv   ---- #

        # torch style with atol + rtol
        zeus.testing.assert_close(
            grad_total_v,
            grad_total_v_ref,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_threshold=dv_thres,
            test_case=f"ref_wo_mso: {test_case} => dv",
        )

    def assert_close_to_torch_ref(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        is_causal_mapping: list[bool],
        total_seqlen_q: int,
        total_seqlen_k: int,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_out: torch.Tensor,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        test_case: str = "",
    ):
        # -----   customize tolerance threshold  ---- #

        o_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        o_thres = {torch.bfloat16: 0.02, torch.float16: 0.02}.get(dtype, 0.02)

        dq_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)
        dq_thres = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)

        dk_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)
        dk_thres = {torch.bfloat16: 0.06, torch.float16: 0.04}.get(dtype, 0.04)

        dv_atol = {torch.bfloat16: EPSILON, torch.float16: EPSILON}.get(dtype, EPSILON)
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        dv_thres = {torch.bfloat16: 0.03, torch.float16: 0.03}.get(dtype, 0.03)

        norm_rtol: float = 2.0  # NOTE: an experimental value from fa testing

        # -----   build attn mask   ---- #

        mask = get_mask_from_ranges(
            q_ranges=q_ranges.to_naive_ranges(),
            k_ranges=k_ranges.to_naive_ranges(),
            q_len=total_seqlen_q,
            k_len=total_seqlen_k,
            is_causal_mapping=is_causal_mapping,
        )

        # -----   ref1. torch ref with high precision (fp32)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

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

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

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

        # -----   assert close for fwd out   ---- #

        # fa style with Linf norm
        out_norm = calc_inf_norm(total_out, total_out_ref_high_precision)
        out_ref_norm = calc_inf_norm(
            total_out_ref_low_precision, total_out_ref_high_precision
        )
        self.assertLessEqual(
            out_norm,
            norm_rtol * out_ref_norm,
            msg=f"torch_ref: {out_norm=} should be no greater than {norm_rtol}x of {out_ref_norm=}",
        )
        # torch style with atol + rtol
        zeus.testing.assert_close(
            total_out,
            total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_threshold=o_thres,
            test_case=f"torch_ref: {test_case} => o",
        )

        # -----   assert close for bwd dq   ---- #

        # fa style with Linf norm
        dq_norm = calc_inf_norm(grad_total_q, grad_total_q_ref_high_precision)
        dq_ref_norm = calc_inf_norm(
            grad_total_q_ref_low_precision, grad_total_q_ref_high_precision
        )
        self.assertLessEqual(
            dq_norm,
            norm_rtol * dq_ref_norm,
            msg=f"torch_ref: {dq_norm=} should be no greater than {norm_rtol}x of {dq_ref_norm=}",
        )
        # torch style with atol + rtol
        zeus.testing.assert_close(
            grad_total_q,
            grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_threshold=dq_thres,
            test_case=f"torch_ref: {test_case} => dq",
        )

        # -----   assert close for bwd dk   ---- #

        # fa style with Linf norm
        dk_norm = calc_inf_norm(grad_total_k, grad_total_k_ref_high_precision)
        dk_ref_norm = calc_inf_norm(
            grad_total_k_ref_low_precision, grad_total_k_ref_high_precision
        )
        self.assertLessEqual(
            dk_norm,
            norm_rtol * dk_ref_norm,
            msg=f"torch_ref: {dk_norm=} should be no greater than {norm_rtol}x of {dk_ref_norm=}",
        )
        # torch style with atol + rtol
        zeus.testing.assert_close(
            grad_total_k,
            grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_threshold=dk_thres,
            test_case=f"torch_ref: {test_case} => dk",
        )

        # -----   assert close for bwd dv   ---- #

        # fa style with Linf norm
        dv_norm = calc_inf_norm(grad_total_v, grad_total_v_ref_high_precision)
        dv_ref_norm = calc_inf_norm(
            grad_total_v_ref_low_precision, grad_total_v_ref_high_precision
        )
        self.assertLessEqual(
            dv_norm,
            norm_rtol * dv_ref_norm,
            msg=f"torch_ref: {dv_norm=} should be no greater than {norm_rtol}x of {dv_ref_norm=}",
        )
        # torch style with atol + rtol
        zeus.testing.assert_close(
            grad_total_v,
            grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_threshold=dv_thres,
            test_case=f"torch_ref: {test_case} => dv",
        )

    def check_group_cast_and_group_reduce(
        self,
        comm_meta: CommMeta,
        device: torch.device,
        atol: float = 8e-4,
        rtol: float = 5e-2,
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

        work = group_cast_collective(
            input=test_input,
            output=test_output,
            input_split_size_list=group_cast_collective_args.input_split_size_list,
            output_split_size_list=group_cast_collective_args.output_split_size_list,
            dst_indices_list=group_cast_collective_args.dst_indices_list,
            src_index_list=group_cast_collective_args.src_index_list,
            group=self.nccl_group,
            async_op=True,
        )
        test_output = work.wait_post_process(test_output)

        work = group_reduce_collective(
            input=test_output,
            output=test_input,
            input_split_size_list=group_cast_collective_args.output_split_size_list,
            output_split_size_list=group_cast_collective_args.input_split_size_list,
            dst_index_list=group_cast_collective_args.src_index_list,
            src_indices_list=group_cast_collective_args.dst_indices_list,
            group=self.nccl_group,
            async_op=True,
        )

        test_input = work.wait_post_process(test_input)

        torch.testing.assert_close(test_input, ans, atol=atol, rtol=rtol)


if __name__ == "__main__":
    run_tests()
