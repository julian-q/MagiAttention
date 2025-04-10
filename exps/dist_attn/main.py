import os
from datetime import timedelta

import torch
import torch.distributed as dist

from dffa import init_dist_attn_runtime_mgr
from dffa.common.enum import AttnMaskType, AttnOverlapMode
from dffa.common.ranges import AttnRanges
from dffa.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
)
from dffa.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from dffa.utils import nvtx

if __name__ == "__main__":
    # -------------------       setup env   ------------------- #

    # env config
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    prof_iters, prof_start_iter, prof_end_iter = 10, 4, 6

    # init dist env
    dist.init_process_group(
        backend="nccl",
        init_method=None,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=30),
        store=None,
    )

    # init device
    device_count = torch.cuda.device_count()
    device = dist.get_rank() % device_count
    assert local_rank == device, "local rank does not match device"
    torch.cuda.set_device(device)
    device = torch.cuda.current_device()

    # init cp group(s)
    nccl_groups = [
        dist.new_group(list(range(world_size)), backend="nccl") for _ in range(2)
    ]

    # -------------------       prepare model   ------------------- #

    # model config
    hidden_size = 1024
    num_heads_q = 48
    num_heads_k = 8
    head_dim = 128
    dtype = torch.float16

    w = torch.randn(hidden_size, num_heads_q * head_dim, device=device, dtype=dtype)

    # -------------------       prepare data   ------------------- #

    # data config
    total_seqlen = 307200

    # block size config
    head_dim_to_q_block_size = {
        64: 128,
        128: 80,
        256: 64,
    }
    q_block_size = head_dim_to_q_block_size[head_dim]

    # mask config
    q_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
    k_ranges = AttnRanges.from_ranges([[0, total_seqlen]])
    is_causal_mapping = False

    # init global data
    total_q = torch.randn(
        total_seqlen,
        num_heads_q,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    total_k = torch.randn(
        total_seqlen,
        num_heads_k,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    total_v = torch.randn(
        total_seqlen,
        num_heads_k,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    grad_total_out = torch.randn_like(total_q).detach()
    # dist.all_reduce(total_q.data, group=nccl_groups[0])
    # dist.all_reduce(total_k.data, group=nccl_groups[0])
    # dist.all_reduce(total_v.data, group=nccl_groups[0])
    # dist.all_reduce(grad_total_out.data, group=nccl_groups[0])

    # -------------------       init dffa   ------------------- #

    # dffa config
    chunk_size = q_block_size * 10
    # TODO: test top-p minhp dispatch alg
    dispatch_config = DispatchConfig(alg=MinHeapDispatchAlg())
    overlap_config = OverlapConfig(
        # enable=False,
        enable=True,
        mode=AttnOverlapMode.STATIC,
        degree=4,
        min_chunk_size=256,
        max_num_chunks=64,
        alg=UniformOverlapAlg(
            random_costs=True,
            random_seed=42,
        ),
    )
    hb_domain_size = 1

    dist_attn_config = DistAttnConfig(
        dispatch_config=dispatch_config,
        overlap_config=overlap_config,
        high_bandwith_domain_size=hb_domain_size,
        deterministic=False,
    )

    # dffa mgr
    dist_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
        total_seqlen_q=total_seqlen,
        total_seqlen_k=total_seqlen,
        chunk_size=chunk_size,
        cp_group=nccl_groups[0],
        is_same_source=True,
        is_q_permutable=True,
        is_k_permutable=True,
        dist_attn_config=dist_attn_config,
    )
    # HACK: double cp group for kv/dkv
    dist_attn_runtime_mgr.dist_attn_runtime.cp_group_dkv = nccl_groups[1]

    # -------------------       run   ------------------- #

    for iter in range(prof_iters):
        # init for nvtx
        nvtx.switch_profile(
            iter_id=iter,
            start=prof_start_iter,
            end=prof_end_iter,
            profile_ranks=[0],
        )

        dist.barrier()
        torch.cuda.synchronize()

        # -----    dispatch   ---- #

        local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

        # -----    forward   ---- #

        local_out, _ = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)

        # -----    undispatch   ---- #

        total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)

        # -----    backward   ---- #

        total_out.backward(grad_total_out)

    # -------------------       clearup env   ------------------- #

    dist.barrier()
    dist.destroy_process_group()
