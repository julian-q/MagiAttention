import random
import time
from functools import partial
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from zeus.comm.functional import all_gather_v_func
from zeus.utils import nvtx

from ..common.enum import AttnMaskType, AttnRole, AttnType
from ..common.meta import DispatchMeta
from ..common.ranges import (
    AttnRange,
    AttnRanges,
    NaiveRanges,
    RangesType,
    find_hole_ranges,
    find_overlap_ranges,
    find_overlap_ranges_new,
)
from ..meta.containers.bucket import AttnBucket, AttnChunk, AttnSlice
from ..utils import (
    cu_seqlens2seqlens,
    flatten_nested_list,
    is_list_all,
    perm_idxs2unperm_idxs,
    seqlens2cu_seqlens,
    wrap_to_list,
)
from .kv_transfer import KVTransferTable
from .solver import DispatchAlgorithm, DispatchSolver


class SequenceDispatcher(nn.Module):
    """Balance dispatching tokens towards each cp rank along the sequence dimension for distributed attention
    where the "balance" has several folds of meaning:
        1. The number of tokens in each cp rank should be exactly balanced, i.e. equal to each other
        2. The computation cost, i.e. the area of the attn mask matrix, in each cp rank should be roughly balanced
        3. The locality of the dispatched tokens in each cp rank should be maximized, i.e.
    """

    def __init__(
        self,
        alg: DispatchAlgorithm = DispatchAlgorithm.MIN_HEAP,
        **alg_kwargs,
    ) -> None:
        super().__init__()

        assert alg in (
            supported_algs := (DispatchAlgorithm.MIN_HEAP,)
        ), f"The algorithm ({alg}) is not supported, choosing from {supported_algs}"
        self.alg = alg

        self.solver = DispatchSolver(self.alg)
        self.solve_func = partial(self.solver.solve, **alg_kwargs)

    @nvtx.instrument_nvtx
    def dispatch(
        self,
        x_global: torch.Tensor,
        meta: DispatchMeta,
        seq_dim: int = 0,
    ) -> torch.Tensor:
        """Dispatch the global tensor `x_global` along its sequence dim following the meta info,
        and return the dispatched local tensor `x_local`

        Args:
            x_global (torch.Tensor): the global tensor to be dispatched
            meta (DispatchMeta): the meta info of the dispatch
            seq_dim (int): the sequence dimension of the tensor

        Returns:
            torch.Tensor: the dispatched local tensor `x_local`
        """

        assert (
            meta.attn_type is AttnType.SELF_ATTN
        ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

        x_split = torch.split(
            x_global, split_size_or_sections=meta.seqlens, dim=seq_dim
        )
        x_perm = torch.concat(
            [x_split[i] for i in meta.seqlens_perm_idxs],
            dim=seq_dim,
        )
        x_chunked = torch.chunk(
            x_perm,
            chunks=meta.num_chunks,
            dim=seq_dim,
        )
        x_local = torch.concat(
            [x_chunked[i] for i in meta.partitions_permed[meta.cp_rank]],
            dim=seq_dim,
        )

        return x_local

    @nvtx.instrument_nvtx
    def undispatch(
        self,
        x_local: torch.Tensor,
        meta: DispatchMeta,
        seq_dim: int = 0,
    ) -> torch.Tensor:
        """Undispatch the local tensor `x_local` along its sequence dim following the meta info,
        and return the undispatched global tensor `x_global`

        Args:
            x_local (torch.Tensor): the local tensor to be undispatched
            meta (DispatchMeta): the meta info of the undispatch
            seq_dim (int): the sequence dimension of the tensor

        Returns:
            torch.Tensor: the undispatched global tensor `x_global`
        """

        assert (
            meta.attn_type is AttnType.SELF_ATTN
        ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

        x_gather = all_gather_v_func(x_local, group=meta.cp_group_nccl, dim=0)
        x_chunked = torch.chunk(
            x_gather,
            chunks=meta.num_chunks,
            dim=seq_dim,
        )
        x_perm = torch.concat(
            [x_chunked[i] for i in meta.partitions_unperm_idxs],
            dim=seq_dim,
        )
        x_split = torch.split(
            x_perm,
            split_size_or_sections=meta.seqlens_permed,
            dim=seq_dim,
        )
        x_global = torch.concat(
            [x_split[i] for i in meta.seqlens_unperm_idxs],
            dim=seq_dim,
        )

        return x_global

    @nvtx.instrument_nvtx
    def compute_varlen_meta(
        self,
        cu_seqlens_q: List[int],
        cu_seqlens_k: List[int] | None = None,
        attn_type: AttnType = AttnType.SELF_ATTN,
        attn_mask_type: AttnMaskType | List[AttnMaskType] = AttnMaskType.FULL,
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_group_nccl: dist.ProcessGroup | None = None,
        cp_group_gloo: dist.ProcessGroup | None = None,
        chunk_size: int = 1,
        overlap_degree: int = 1,
        shuffle_times: int = 100,
        shuffle_timeout: int = 10,
        shuffle_seed: int = 42,
        **kwargs,
    ) -> Tuple[DispatchMeta, DispatchMeta]:
        """Compute the dispatch meta information for this cp rank,
            based on some basic varlen data packing structure like cu_seqlens_q and optional cu_seqlens_k

        Args:
            cu_seqlens_q (List[int]): the global cumulative sequence lengths of the query tokens,
                a list of int with a length of batch_size + 1
            cu_seqlens_k (List[int] | None): the global cumulative sequence lengths of the key tokens,
                a list of int with a length of batch_size + 1
                NOTE: if attn_type is `self_attn`, then cu_seqlens_k should be None to be equal to cu_seqlens_q,
                    while it should be given if attn_type is `cross_attn`
            attn_mask_type (AttnMaskType | List[AttnMaskType]): indicate the (i-th sample of) attn mask type
                e.g. 'full' or 'causal', and if it is a list, it should have the same length as cu_seqlens_q
                to indicate the attn mask type for each sample in the batch
            cp_rank (int): the rank of the current cp
            cp_size (int): the size of the cp
            cp_group_nccl (dist.ProcessGroup | None): the process group for nccl backend
            cp_group_gloo (dist.ProcessGroup | None): the process group for gloo backend
            chunk_size (int): the chunk size of any unit of tokens to be dispatched,
                which should be divisible by the total seqlen, as well as
                the number of chunks should be divisible by `cp_size`
            overlap_degree (int): the overlap degree of remote kv computation and communication
            shuffle_times (int): the maximum number of times to shuffle the tokens
            shuffle_timeout (int): the timeout milliseconds for the shuffling,
                NOTE: the shuffling will stop when either shuffle_times or shuffle_timeout is been reached
            shuffle_seed (int | None): the seed for the shuffling
            **kwargs: additional keyword arguments saved for future use

        Returns:
            DispatchMeta: the meta information of the dispatched query tokens for this cp rank
            DispatchMeta: the meta information of the dispatched key tokens for this cp rank
        """

        assert cp_group_nccl is not None, "cp_group_nccl is required"
        assert cp_group_gloo is not None, "cp_group_gloo is required"

        if attn_type is AttnType.SELF_ATTN:
            assert (
                cu_seqlens_k is None
            ), "cu_seqlens_k should be None when attn_type is `self_attn`"
            cu_seqlens_k = cu_seqlens_q
        elif attn_type is AttnType.CROSS_ATTN:
            assert (
                cu_seqlens_k is not None
            ), "cu_seqlens_k should be given when attn_type is `cross_attn`"

        assert cu_seqlens_q[0] == cu_seqlens_k[0] == 0 and len(cu_seqlens_k) == len(
            cu_seqlens_q
        ), (
            f"both cu_seqlens_q and cu_seqlens_k should start from 0 and share the same length, "
            f"but got {cu_seqlens_q} and {cu_seqlens_k}"
        )

        q_ranges = AttnRanges.from_cu_seqlens(cu_seqlens_q, as_cu_seqlens=True)
        k_ranges = AttnRanges.from_cu_seqlens(cu_seqlens_k, as_cu_seqlens=False)

        return self.compute_meta(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type=attn_type,
            attn_mask_type=attn_mask_type,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group_nccl=cp_group_nccl,
            cp_group_gloo=cp_group_gloo,
            chunk_size=chunk_size,
            overlap_degree=overlap_degree,
            shuffle_times=shuffle_times,
            shuffle_timeout=shuffle_timeout,
            shuffle_seed=shuffle_seed,
            **kwargs,
        )

    @nvtx.instrument_nvtx
    def compute_meta(
        self,
        q_ranges: RangesType,
        k_ranges: RangesType,
        attn_type: AttnType = AttnType.SELF_ATTN,
        attn_mask_type: AttnMaskType | List[AttnMaskType] = AttnMaskType.FULL,
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_group_nccl: dist.ProcessGroup | None = None,
        cp_group_gloo: dist.ProcessGroup | None = None,
        chunk_size: int = 1,
        overlap_degree: int = 1,
        shuffle_times: int = 0,  # NOTE: shut it down for a while
        shuffle_timeout: int = 10,
        shuffle_seed: int = 42,
        **kwargs,
    ) -> Tuple[DispatchMeta, DispatchMeta]:
        """Compute the dispatch meta information for this cp rank,
            based on some basic data packing structure like q_ranges and k_ranges

        Args:
            q_ranges (RangesType): the ranges of query tokens, a list of pairs of int, with a length of batch_size
            k_ranges (RangesType): the ranges of the key tokens, a list of pairs of int, with a length of batch_size
                NOTE: as for our settings:
                    * `q_ranges` should be formed consecutively, mutually excusively, completely just like `cu_seqlens_q`,
                        e.g. q_ranges = [(0, 4), (4, 7), (7, 12)] (the corresponding cu_seqlens_q is [0, 4, 7, 12])
                    * `k_ranges`, however, can be any arbitrary structure, e.g. k_ranges = [(2, 5), (3, 6), (7, 11)]
            attn_mask_type (AttnMaskType | List[AttnMaskType]): indicate the (i-th sample of) attn mask type
                e.g. 'full' or 'causal', and if it is a list, it should have the same length as cu_seqlens_q
                to indicate the attn mask type for each sample in the batch
            cp_rank (int): the rank of the current cp
            cp_size (int): the size of the cp
            cp_group_nccl (dist.ProcessGroup | None): the process group for nccl backend
            cp_group_gloo (dist.ProcessGroup | None): the process group for gloo backend
            chunk_size (int): the chunk size of any unit of tokens to be dispatched,
                which should be divisible by the total seqlen, as well as
                the number of chunks should be divisible by `cp_size`
            overlap_degree (int): the overlap degree of remote kv computation and communication
            shuffle_times (int): the maximum number of times to shuffle the tokens
            shuffle_timeout (int): the timeout milliseconds for the shuffling,
                NOTE: the shuffling will stop when either shuffle_times or shuffle_timeout is been reached
            shuffle_seed (int | None): the seed for the shuffling
            **kwargs: additional keyword arguments saved for future use

        Returns:
            DispatchMeta: the meta information of the dispatched query tokens for this cp rank
            DispatchMeta: the meta information of the dispatched key tokens for this cp rank
        """
        attn_mask_type = wrap_to_list(attn_mask_type)

        # FIXME: limitations to be improved in the future
        assert attn_type is AttnType.SELF_ATTN, "For now, only supports self-attention."
        assert is_list_all(
            attn_mask_type, AttnMaskType.FULL
        ), "Only supports all full attn mask for now."
        assert overlap_degree == 1, "For now, only supports overlap degree == 1."

        q_ranges = AttnRanges.from_ranges(q_ranges, as_cu_seqlens=True)
        k_ranges = AttnRanges.from_ranges(k_ranges, as_cu_seqlens=False)
        assert isinstance(q_ranges, AttnRanges)
        assert isinstance(k_ranges, AttnRanges)

        AttnRanges.check_valid_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            is_self_attn=attn_type is AttnType.SELF_ATTN,
        )

        if attn_type is AttnType.SELF_ATTN:
            meta_q, meta_k = self._compute_self_attn_meta(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                cp_rank=cp_rank,
                cp_size=cp_size,
                cp_group_nccl=cp_group_nccl,
                cp_group_gloo=cp_group_gloo,
                chunk_size=chunk_size,
                overlap_degree=overlap_degree,
                shuffle_times=shuffle_times,
                shuffle_timeout=shuffle_timeout,
                shuffle_seed=shuffle_seed,
                **kwargs,
            )
        elif attn_type is AttnType.CROSS_ATTN:
            raise NotImplementedError("Cross attention is not supported yet.")
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        return meta_q, meta_k

    @nvtx.instrument_nvtx
    def _compute_self_attn_meta(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: List[AttnMaskType],
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_group_nccl: dist.ProcessGroup | None = None,
        cp_group_gloo: dist.ProcessGroup | None = None,
        chunk_size: int = 1,
        overlap_degree: int = 1,
        shuffle_times: int = 100,
        shuffle_timeout: int = 10,
        shuffle_seed: int = 42,
        **kwargs,
    ) -> Tuple[DispatchMeta, DispatchMeta]:
        """Inner function to compute the dispatch meta information for self attention"""
        total_seqlen, batch_size = q_ranges.end, q_ranges.size

        assert (
            total_seqlen % chunk_size == 0
        ), f"The total seqlen ({total_seqlen}) should be divisible by chunk size ({chunk_size})."

        num_chunks = total_seqlen // chunk_size
        assert (
            num_chunks % cp_size == 0
        ), f"The number of chunks ({num_chunks}) should be divisible by cp size ({cp_size})."

        cu_seqlens = (
            q_ranges.to_cu_seqlens()
        )  # q_ranges can be transferred to cu_seqlens_q for sure
        seqlens = cu_seqlens2seqlens(cu_seqlens)

        (
            q_ranges_permed,
            k_ranges_permed,
            global_bucket,
            seqlens_perm_idxs,
            partitions_permed,
        ) = self._shuffle_compute_self_attn_areas_and_solve(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            cp_size=cp_size,
            chunk_size=chunk_size,
            overlap_degree=overlap_degree,
            shuffle_times=shuffle_times,
            shuffle_timeout=shuffle_timeout,
            shuffle_seed=shuffle_seed,
        )

        seqlens_unperm_idxs = perm_idxs2unperm_idxs(seqlens_perm_idxs)
        partitions_perm_idxs = flatten_nested_list(partitions_permed)  # type: ignore
        partitions_unperm_idxs = perm_idxs2unperm_idxs(partitions_perm_idxs)

        seqlens_permed = [seqlens[i] for i in seqlens_perm_idxs]
        cu_seqlens_permed = seqlens2cu_seqlens(seqlens_permed)

        (
            buckets_per_rank,
            host_qk_ranges_global_per_rank,
            host_qk_ranges_local_per_rank,
            host_req_k_ranges_global_per_rank,
            remote_k_ranges_global_per_rank,
            remote_k_ranges_local_per_rank,
            kv_transfer_table,
        ) = self._compute_kv_transfer_table(
            global_bucket=global_bucket,
            partitions=partitions_permed,
            cp_group=cp_group_gloo,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

        # TODO: compute the overlap split size for overlap degree > 1
        num_remote_tokens = remote_k_ranges_global_per_rank[cp_rank].seqlen
        overlap_split_size_list = [num_remote_tokens]

        (
            kv_input_split_size_list,
            kv_dst_indices_list,
            kv_output_split_size_list,
            kv_src_index_list,
        ) = self._compute_group_cast_args(
            kv_transfer_table=kv_transfer_table,
            cp_rank=cp_rank,
            cp_size=cp_size,
        )

        (
            local_attn_arg_q_ranges,
            local_attn_arg_k_ranges,
            local_attn_arg_is_causal_mapping,
            local_attn_arg_max_seqlen_q,
            local_attn_arg_max_seqlen_k,
            remote_attn_args_q_ranges_list,
            remote_attn_args_k_ranges_list,
            remote_attn_args_is_causal_mapping_list,
            remote_attn_args_max_seqlen_q_list,
            remote_attn_args_max_seqlen_k_list,
        ) = self._compute_attn_args(
            host_qk_ranges_global_for_this_rank=host_qk_ranges_global_per_rank[cp_rank],
            host_qk_ranges_local_for_this_rank=host_qk_ranges_local_per_rank[cp_rank],
            host_req_k_ranges_global_for_this_rank=host_req_k_ranges_global_per_rank[
                cp_rank
            ],
            remote_k_ranges_global_for_this_rank=remote_k_ranges_global_per_rank[
                cp_rank
            ],
            attn_mask_type=attn_mask_type,
            overlap_degree=overlap_degree,
        )

        common_meta_kwargs = dict(
            attn_type=AttnType.SELF_ATTN,
            attn_mask_type=attn_mask_type,
            batch_size=batch_size,
            total_seqlen=total_seqlen,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_group_nccl=cp_group_nccl,
            cp_group_gloo=cp_group_gloo,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            overlap_degree=overlap_degree,
            num_remote_tokens=num_remote_tokens,
            overlap_split_size_list=overlap_split_size_list,
            seqlens=seqlens,
            seqlens_permed=seqlens_permed,
            seqlens_perm_idxs=seqlens_perm_idxs,
            seqlens_unperm_idxs=seqlens_unperm_idxs,
            cu_seqlens=cu_seqlens,
            cu_seqlens_permed=cu_seqlens_permed,
            partitions_permed=partitions_permed,
            partitions_perm_idxs=partitions_perm_idxs,
            partitions_unperm_idxs=partitions_unperm_idxs,
            global_bucket=global_bucket,
            buckets_per_rank=buckets_per_rank,
            host_qk_ranges_global_per_rank=host_qk_ranges_global_per_rank,
            host_qk_ranges_local_per_rank=host_qk_ranges_local_per_rank,
            host_req_k_ranges_global_per_rank=host_req_k_ranges_global_per_rank,
            remote_k_ranges_global_per_rank=remote_k_ranges_global_per_rank,
            remote_k_ranges_local_per_rank=remote_k_ranges_local_per_rank,
            kv_transfer_table=kv_transfer_table,
            kv_input_split_size_list=kv_input_split_size_list,
            kv_output_split_size_list=kv_output_split_size_list,
            kv_dst_indices_list=kv_dst_indices_list,
            kv_src_index_list=kv_src_index_list,
            local_attn_arg_q_ranges=local_attn_arg_q_ranges,
            local_attn_arg_k_ranges=local_attn_arg_k_ranges,
            local_attn_arg_is_causal_mapping=local_attn_arg_is_causal_mapping,
            local_attn_arg_max_seqlen_q=local_attn_arg_max_seqlen_q,
            local_attn_arg_max_seqlen_k=local_attn_arg_max_seqlen_k,
            remote_attn_args_q_ranges_list=remote_attn_args_q_ranges_list,
            remote_attn_args_k_ranges_list=remote_attn_args_k_ranges_list,
            remote_attn_args_is_causal_mapping_list=remote_attn_args_is_causal_mapping_list,
            remote_attn_args_max_seqlen_q_list=remote_attn_args_max_seqlen_q_list,
            remote_attn_args_max_seqlen_k_list=remote_attn_args_max_seqlen_k_list,
        )

        meta_q = DispatchMeta(
            attn_role=AttnRole.QUERY,
            ranges=q_ranges,
            ranges_permed=q_ranges_permed,
            **common_meta_kwargs,  # type: ignore
        )
        meta_k = DispatchMeta(
            attn_role=AttnRole.KEY,
            ranges=k_ranges,
            ranges_permed=k_ranges_permed,
            **common_meta_kwargs,  # type: ignore
        )

        return meta_q, meta_k

    @nvtx.instrument_nvtx
    def _shuffle_compute_self_attn_areas_and_solve(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: List[AttnMaskType],
        cp_size: int = 1,
        chunk_size: int = 1,
        overlap_degree: int = 1,
        shuffle_times: int = 0,
        shuffle_timeout: int = 10,
        shuffle_seed: int = 42,
    ) -> Tuple[AttnRanges, AttnRanges, AttnBucket, List[int], List[List[int]]]:
        """A scheduler function, which executes the following steps for each shuffle iteration:
        1. shuffle the q and k ranges by samples, and check if this permutation is safe (i.e. valid) for ffa
        2. compute the self-attn areas for each chunk and return the global bucket containing all the chunks in turn
        3. solve the area-balancing problem by partitioning the chunks into cp_size local buckets
        3. update the best answer (the minimum of the maximum of workloads) and best partitions of chunks
        """
        cur_perm_idxs, best_perm_idxs = list(range(len(q_ranges))), None
        best_answer, best_partitions = float("inf"), None
        best_q_ranges_permed, best_k_ranges_permed = None, None
        best_global_bucket = None

        random.seed(shuffle_seed)
        shuffle_timeout_sec = shuffle_timeout / 1000
        start_time = time.time()
        for _ in range(shuffle_times + 1):
            (
                is_safe_to_permute,
                q_ranges,
                k_ranges,
            ) = self._safe_permute_qk_ranges_by_samples(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                perm_idxs=cur_perm_idxs,
            )

            if is_safe_to_permute:
                global_bucket: AttnBucket = self._compute_self_attn_areas(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    chunk_size=chunk_size,
                    overlap_degree=overlap_degree,
                    attn_mask_type=attn_mask_type,
                )
                attn_areas = global_bucket.areas

                answer, workloads, partitions = self.solve_func(
                    jobs=attn_areas,
                    k=cp_size,
                )

                if answer < best_answer:
                    best_answer = answer
                    best_perm_idxs = cur_perm_idxs.copy()
                    best_partitions = partitions
                    best_q_ranges_permed = q_ranges
                    best_k_ranges_permed = k_ranges
                    best_global_bucket = global_bucket

            elapsed_time = time.time() - start_time
            if elapsed_time > shuffle_timeout_sec:
                break

            random.shuffle(cur_perm_idxs)

        # since the order for any partition doesn't matter,
        # here we just keep it sorted ascendingly, like (0,5,4) -> (0,4,5)
        best_partitions = [sorted(p) for p in best_partitions]  # type: ignore

        return (  # type: ignore
            best_q_ranges_permed,
            best_k_ranges_permed,
            best_global_bucket,
            best_perm_idxs,
            best_partitions,
        )

    @nvtx.instrument_nvtx
    def _compute_self_attn_areas(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: List[AttnMaskType],
        chunk_size: int | None = None,
        overlap_degree: int = 1,
    ) -> AttnBucket:
        """Compute the self-attn areas, with constructing the global bucket,
        which is mainly consists of a list of all the chunks in ascending order, with a length of `cp_size`

        Args:
            q_ranges (AttnRanges): the query ranges
            k_ranges (AttnRanges): the key ranges
            attn_mask_type (List[AttnMaskType]): the attn mask type list
            chunk_size (int | None): the chunk size, which should be divisible by `cp_size`
            overlap_degree (int): the overlap degree of remote kv computation and communication

        Returns:
            AttnBucket: the global bucket
        """
        ts = q_ranges.end
        assert is_list_all(
            attn_mask_type, just_same=True
        ), "Only supports all full attn mask or all causal attn mask for now."
        mask_type = attn_mask_type[0]
        is_causal = mask_type == AttnMaskType.CAUSAL

        if chunk_size is None:
            chunk_size = ts
        num_chunks = ts // chunk_size

        global_bucket = AttnBucket()
        range_idx, seqi_mid = 0, 0

        for chunk_idx in range(num_chunks):  # for each chunk
            chunk: AttnChunk = AttnChunk(chunk_id=chunk_idx)
            cur_chunk_size = 0

            slice_idx = 0
            while cur_chunk_size < chunk_size:  # for each slice
                slice: AttnSlice = AttnSlice(slice_id=slice_idx, mask_type=mask_type)

                seqi_end = q_ranges[range_idx].end
                seqi_len_bottom = seqi_end - seqi_mid

                attn_len = k_ranges[range_idx].size
                attn_start, attn_end = (
                    k_ranges[range_idx].start,
                    k_ranges[range_idx].end,
                )

                exceed_size = seqi_len_bottom + cur_chunk_size - chunk_size

                q_range_start, q_range_end, k_range_start, k_range_end = (
                    None,
                    None,
                    None,
                    None,
                )
                # analyze this slice
                if (
                    exceed_size <= 0
                ):  # this bottom half of seqi should be all in this chunk
                    # set start and end for q_range of this slice
                    q_range_start, q_range_end = seqi_mid, seqi_end

                    # set start and end for k_range of this slice
                    k_range_start, k_range_end = attn_start, attn_end

                    # compuate areas
                    if is_causal:
                        if attn_len > seqi_len_bottom:  # the area of a trapezoid
                            slice.area = (
                                (2 * attn_len - seqi_len_bottom) * seqi_len_bottom // 2
                            )
                        else:  # the area of a triangle
                            slice.area = (1 + attn_len) * attn_len // 2
                    else:  # the area of a rectangle
                        slice.area = seqi_len_bottom * attn_len

                    # iterate to the next sample within the same chunk
                    range_idx += 1
                    seqi_mid = seqi_end
                    cur_chunk_size += seqi_len_bottom
                else:  # only the prefix of this bottom half of seqi should be in this chunk
                    # truncate the seqlen to the edge of chunk line
                    seqi_end_truncate = seqi_end - exceed_size
                    seqi_len_bottom_truncate = seqi_end_truncate - seqi_mid
                    attn_len_truncate = attn_len - exceed_size

                    # set start and end for q_range of this slice
                    q_range_start, q_range_end = seqi_mid, seqi_end_truncate

                    # compuate areas
                    if is_causal:
                        if attn_len > seqi_len_bottom:  # the area of a trapezoid
                            slice.area = (
                                (
                                    2 * (attn_len - seqi_len_bottom)
                                    + seqi_len_bottom_truncate
                                )
                                * seqi_len_bottom_truncate
                                // 2
                            )
                            # set start and end for k_range of this slice
                            k_range_start, k_range_end = (
                                attn_start,
                                attn_start + attn_len_truncate,
                            )
                        elif attn_len > exceed_size:  # the area of a triangle
                            slice.area = (
                                (1 + attn_len_truncate) * attn_len_truncate // 2
                            )
                            # set start and end for k_range of this slice
                            k_range_start, k_range_end = (
                                attn_start,
                                attn_start + attn_len_truncate,
                            )
                        else:  # no area to compute
                            slice.area = 0
                            # set start and end for k_range of this slice
                            k_range_start, k_range_end = attn_start, attn_start
                    else:  # the area of a rectangle
                        slice.area = seqi_len_bottom_truncate * attn_len
                        # set start and end for k_range of this slice
                        k_range_start, k_range_end = attn_start, attn_end

                    # iterate to next chunk within the same sample
                    seqi_mid = seqi_end_truncate
                    cur_chunk_size = chunk_size

                # set q_range, k_range for this slice
                slice.q_range = AttnRange(start=q_range_start, end=q_range_end)
                slice.k_range = AttnRange(start=k_range_start, end=k_range_end)

                # append this q slice to the current chunk
                chunk.q_slices.append(slice)

                slice_idx += 1

            global_bucket.q_chunks.append(chunk)

        return global_bucket

    @nvtx.instrument_nvtx
    def _compute_kv_transfer_table(
        self,
        global_bucket: AttnBucket,
        partitions: List[List[int]],
        cp_group: dist.ProcessGroup,
        cp_size: int,
        cp_rank: int,
    ) -> Tuple[
        List[AttnBucket],
        List[AttnRanges],
        List[AttnRanges],
        List[AttnRanges],
        List[AttnRanges],
        List[AttnRanges],
        KVTransferTable,
    ]:
        """Compute the kv transfer table, which mainly includes:
            1. kv_receive_table (List[AttnRanges]): a list of k_ranges, where table[ranki] indicates:
                the k_ranges that should be received from ranki to this rank
            2. kv_send_table (List[AttnRanges]): a list of k_ranges, where table[ranki] indicates:
                the k_ranges that should be sent to ranki from this rank

        Args:
            global_bucket (AttnBucket): the global bucket where all chunks are stored in ascending order
            partitions (List[List[int]]): the partitions of the chunks,
                where partitions[ranki] is a list of chunk ids indicating the chunks that belong to ranki, e.g.
                [[0, 4, 6], [1, 2, 5], [3, 7, 8], ...] => \
                    rank0 holds the chunks 0, 4, 6; rank1 holds the chunks 1, 2, 5; rank2 holds the chunks 3, 7, 8; ...
            cp_group (dist.ProcessGroup): the context parallel process group
            cp_rank (int): the cp rank
            cp_size (int): the cp size

        Returns:
            buckets_per_rank: List[AttnBucket]: a list of AttnBucket,
                where buckets[ranki] indicates the chunks that belong to ranki (global meta info)
            host_qk_ranges_global_per_rank: List[AttnRanges]: a list of AttnRanges,
                where ranges[ranki] indicates the qk ranges (with global indices) that stores on ranki (global meta info)
            host_qk_ranges_local_per_rank: List[AttnRanges]: a list of AttnRanges,
                where ranges[ranki] indicates the qk ranges (with local indices) that stores on ranki (global meta info)
            host_req_k_ranges_global_per_rank: List[AttnRanges]: a list of AttnRanges,
                where ranges[ranki] indicates the k ranges (with global indices) that
                    requests to be attended on ranki (global meta info)
            remote_k_ranges_global_per_rank: List[AttnRanges]: a list of AttnRanges,
                where ranges[ranki] indicates the k ranges (with global indices) that
                    ranki needs to fetch from remote (global meta info)
            remote_k_ranges_local_per_rank: List[AttnRanges]: a list of AttnRanges,
                where ranges[ranki] indicates the k ranges (with local indices in itself) that
                    ranki needs to fetch from remote (global meta info)
            kv_send_table_for_this_rank_global: List[AttnRanges]: a list of AttnRanges,
                where table[ranki] indicates the k ranges (with global indices) that
                    should be sent to ranki from this rank (local meta info)
            kv_receive_table_for_this_rank_global: List[AttnRanges]: a list of AttnRanges,
                where table[ranki] indicates the k ranges (with global indices) that
                    should be received from ranki to this rank (local meta info)
            kv_send_table_for_this_rank_local: List[AttnRanges]: a list of AttnRanges,
                where table[ranki] indicates the k ranges (with local indices) that
                    should be sent to ranki from this rank (local meta info)
            kv_receive_table_for_this_rank_local: List[AttnRanges]: a list of AttnRanges,
                where table[ranki] indicates the k ranges (with local indices) that
                    should be received from ranki to this rank (local meta info)
            kv_input_split_sizes: List[List[int]]: a list of List[int],
            kv_output_split_sizes: List[int]: a list of indices,

        """
        buckets_per_rank: List[AttnBucket] = [
            AttnBucket(
                cp_rank=rank,
                q_chunks=[global_bucket.q_chunks[chunk_id] for chunk_id in partition],
            )
            for rank, partition in enumerate(partitions)
        ]

        # host q, k ranges (flatten)
        host_qk_ranges_global_per_rank: List[AttnRanges] = [
            bucket.q_ranges for bucket in buckets_per_rank
        ]
        host_qk_ranges_local_per_rank: List[AttnRanges] = [
            host_global_ranges.to_local_ranges()
            for host_global_ranges in host_qk_ranges_global_per_rank
        ]

        # host k ranges requested / available to be attended
        host_req_k_ranges_global_per_rank: List[AttnRanges] = [
            bucket.k_ranges for bucket in buckets_per_rank
        ]

        # remote k ranges (flatten and aggregated)
        remote_k_ranges_global_per_rank: List[AttnRanges] = [
            bucket.remote_k_ranges for bucket in buckets_per_rank
        ]
        remote_k_ranges_local_per_rank: List[AttnRanges] = [
            remote_global_ranges.to_local_ranges()
            for remote_global_ranges in remote_k_ranges_global_per_rank
        ]

        kv_transfer_table = KVTransferTable(
            cp_group=cp_group,
            host_qk_ranges_global_per_rank=host_qk_ranges_global_per_rank,
            remote_k_ranges_global_per_rank=remote_k_ranges_global_per_rank,
        )

        with nvtx.add_nvtx_event("build_kv_receive_table"):
            # --------------      kv receive table      --------------#
            # for each remote_k_range for this rank, which other rank should it receive the kv from ?
            # table[i]: the k_ranges that should be received from ranki to this rank
            # thus: table[this_rank] should be empty
            kv_receive_table_for_this_rank_global: List[AttnRanges] = [
                AttnRanges() for _ in range(cp_size)
            ]
            host_qk_ranges_global_for_this_rank: AttnRanges = (
                host_qk_ranges_global_per_rank[cp_rank]
            )
            remote_k_ranges_global_for_this_rank: AttnRanges = (
                remote_k_ranges_global_per_rank[cp_rank]
            )

            if not remote_k_ranges_global_for_this_rank.is_empty():
                remote_start, remote_end = (
                    remote_k_ranges_global_for_this_rank.start,
                    remote_k_ranges_global_for_this_rank.end,
                )
                remote_ranges: NaiveRanges = remote_k_ranges_global_for_this_rank.ranges
                for rank, host_qk_ranges_global in enumerate(
                    host_qk_ranges_global_per_rank
                ):
                    if rank != cp_rank:  # skip this rank itself
                        overlap_ranges = find_overlap_ranges(  # 连着的
                            ranges1=remote_ranges,
                            ranges2=host_qk_ranges_global.ranges,
                            axis_start=min(remote_start, host_qk_ranges_global.start),
                            axis_end=max(remote_end, host_qk_ranges_global.end),
                        )

                        overlap_ranges_new = find_overlap_ranges_new(
                            ranges1=AttnRanges.from_ranges(remote_ranges),
                            ranges2=AttnRanges.from_ranges(
                                host_qk_ranges_global.ranges
                            ),
                        )
                        assert overlap_ranges_new == AttnRanges.from_ranges(
                            overlap_ranges
                        ), f"{overlap_ranges_new} != {overlap_ranges}"

                        for overlap_range in overlap_ranges:
                            kv_receive_table_for_this_rank_global[rank].append(
                                AttnRange.from_range(overlap_range), check=False
                            )

            kv_receive_table_for_this_rank_local: List[AttnRanges] = []
            for rank, remote_k_ranges in enumerate(
                kv_receive_table_for_this_rank_global
            ):
                remote_k_ranges_to_local = AttnRanges()
                for remote_k_range in remote_k_ranges:
                    remote_k_ranges_to_local.append(
                        remote_k_ranges_global_per_rank[cp_rank].make_range_local(
                            remote_k_range
                        )
                    )
                kv_receive_table_for_this_rank_local.append(remote_k_ranges_to_local)

            for i in range(cp_size):
                kv_transfer_table.get_ranges(i, cp_rank, "recv").extend(
                    kv_receive_table_for_this_rank_local[i]
                )
                kv_transfer_table.get_ranges(i, cp_rank, "global").extend(
                    kv_receive_table_for_this_rank_global[i]
                )

        with nvtx.add_nvtx_event("build_kv_send_table"):
            # --------------      kv send table      --------------#
            # for each other rank, which local_k_ranges it needs this rank to send ?
            # table[i]: the k_ranges that should be sent from this rank to ranki
            # thus: table[this_rank] should be empty
            kv_send_table_for_this_rank_global: List[AttnRanges] = [
                AttnRanges() for _ in range(cp_size)
            ]
            host_qk_ranges_global_for_this_rank: AttnRanges = (  # type: ignore
                host_qk_ranges_global_per_rank[cp_rank]
            )

            local_start, local_end = (
                host_qk_ranges_global_for_this_rank.start,
                host_qk_ranges_global_for_this_rank.end,
            )
            local_ranges: NaiveRanges = host_qk_ranges_global_for_this_rank.ranges
            for rank, remote_k_ranges in enumerate(remote_k_ranges_global_per_rank):
                if (
                    rank != cp_rank and not remote_k_ranges.is_empty()
                ):  # skip this rank itself, as well as the rank that does need any remote k_ranges
                    overlap_ranges = find_overlap_ranges(
                        ranges1=remote_k_ranges.ranges,
                        ranges2=local_ranges,
                        axis_start=min(local_start, remote_k_ranges.start),
                        axis_end=max(local_end, remote_k_ranges.end),
                    )

                    overlap_ranges_new = find_overlap_ranges_new(
                        ranges1=AttnRanges.from_ranges(remote_k_ranges.ranges),
                        ranges2=AttnRanges.from_ranges(local_ranges),
                    )
                    assert overlap_ranges_new == AttnRanges.from_ranges(
                        overlap_ranges
                    ), f"{overlap_ranges_new} != {overlap_ranges}"

                    for overlap_range in overlap_ranges:
                        kv_send_table_for_this_rank_global[rank].append(
                            AttnRange.from_range(overlap_range), check=False
                        )

            kv_send_table_for_this_rank_local: List[AttnRanges] = []
            for rank, remote_k_ranges in enumerate(kv_send_table_for_this_rank_global):
                remote_k_ranges_to_local = AttnRanges()
                for remote_k_range in remote_k_ranges:
                    remote_k_ranges_to_local.append(
                        host_qk_ranges_global_per_rank[cp_rank].make_range_local(
                            remote_k_range
                        )
                    )
                kv_send_table_for_this_rank_local.append(remote_k_ranges_to_local)

            for i in range(cp_size):
                kv_transfer_table.get_ranges(cp_rank, i, "send").extend(
                    kv_send_table_for_this_rank_local[i]
                )

        kv_transfer_table.correct()

        return (
            buckets_per_rank,
            host_qk_ranges_global_per_rank,
            host_qk_ranges_local_per_rank,
            host_req_k_ranges_global_per_rank,
            remote_k_ranges_global_per_rank,
            remote_k_ranges_local_per_rank,
            kv_transfer_table,
        )

    @nvtx.instrument_nvtx
    def _compute_group_cast_args(
        self,
        kv_transfer_table: KVTransferTable,
        cp_rank: int,
        cp_size: int,
    ):
        return kv_transfer_table.to_group_cast_args(cp_rank)

    @nvtx.instrument_nvtx
    def _compute_attn_args(
        self,
        host_qk_ranges_global_for_this_rank: AttnRanges,
        host_qk_ranges_local_for_this_rank: AttnRanges,
        host_req_k_ranges_global_for_this_rank: AttnRanges,
        remote_k_ranges_global_for_this_rank: AttnRanges,
        attn_mask_type: List[AttnMaskType],
        overlap_degree: int,
    ):
        assert is_list_all(
            attn_mask_type, just_same=True
        ), "Only supports all full attn mask or all causal attn mask for now."
        is_causal = attn_mask_type[0] == AttnMaskType.CAUSAL

        # init local attn args
        local_attn_arg_q_ranges = AttnRanges()
        local_attn_arg_k_ranges = AttnRanges()
        local_attn_arg_is_causal_mapping: List[bool] = []
        local_attn_arg_max_seqlen_q = local_attn_arg_max_seqlen_k = 0

        # init remote attn args
        remote_attn_args_q_ranges_list: List[AttnRanges] = [
            AttnRanges() for _ in range(overlap_degree)
        ]
        remote_attn_args_k_ranges_list: List[AttnRanges] = [
            AttnRanges() for _ in range(overlap_degree)
        ]
        remote_attn_args_is_causal_mapping_list: List[List[bool]] = [
            [] for _ in range(overlap_degree)
        ]
        remote_attn_args_max_seqlen_q_list: List[int] = [
            0 for _ in range(overlap_degree)
        ]
        remote_attn_args_max_seqlen_k_list: List[int] = [
            0 for _ in range(overlap_degree)
        ]

        if is_causal:
            raise NotImplementedError("Causal attention is not supported yet.")
        else:
            has_remote_k_ranges = not remote_k_ranges_global_for_this_rank.is_empty()
            # init remote k ranges if needed
            if has_remote_k_ranges:
                (
                    remote_k_ranges_global,
                    remote_k_ranges_start_global,
                    remote_k_ranges_end_global,
                ) = (
                    remote_k_ranges_global_for_this_rank.ranges,
                    remote_k_ranges_global_for_this_rank.start,
                    remote_k_ranges_global_for_this_rank.end,
                )

            # FIXME: there might be a long traversal
            for host_qk_range_local, host_req_k_range_global in zip(
                host_qk_ranges_local_for_this_rank,
                host_req_k_ranges_global_for_this_rank,
            ):
                (
                    host_req_k_range_global,
                    host_req_k_range_global_start,
                    host_req_k_range_global_end,
                ) = (
                    host_req_k_range_global.range,  # type: ignore
                    host_req_k_range_global.start,
                    host_req_k_range_global.end,
                )
                if has_remote_k_ranges:
                    axis_start = min(
                        host_req_k_range_global_start, remote_k_ranges_start_global
                    )
                    axis_end = max(
                        host_req_k_range_global_end, remote_k_ranges_end_global
                    )

                    host_req_k_avail_sub_ranges_global: NaiveRanges = find_hole_ranges(
                        all_ones_ranges=[host_req_k_range_global],  # type: ignore
                        all_zeros_ranges=remote_k_ranges_global,
                        axis_start=axis_start,
                        axis_end=axis_end,
                    )
                    host_req_k_unavail_sub_ranges_global: NaiveRanges = (
                        find_overlap_ranges(
                            ranges1=[host_req_k_range_global],  # type: ignore
                            ranges2=remote_k_ranges_global,
                            axis_start=axis_start,
                            axis_end=axis_end,
                        )
                    )

                    overlap_ranges_new = find_overlap_ranges_new(
                        ranges1=AttnRanges.from_ranges([host_req_k_range_global]),
                        ranges2=AttnRanges.from_ranges(remote_k_ranges_global),
                    )
                    assert overlap_ranges_new == AttnRanges.from_ranges(
                        host_req_k_unavail_sub_ranges_global
                    ), f"{overlap_ranges_new} != {host_req_k_unavail_sub_ranges_global}"

                else:  # no remote k ranges, so the whole host req k range is available
                    host_req_k_avail_sub_ranges_global = [host_req_k_range_global]  # type: ignore
                    host_req_k_unavail_sub_ranges_global = []

                # set local attn args for this (q, k) range pair, if available
                # NOTE: if this req k range is totally unavailable, 'find_hole_ranges' will return []
                if (
                    host_req_k_avail_sub_ranges_global
                ):  # this (sub) req k range is available on the host
                    host_req_k_avail_sub_ranges_global = AttnRanges.from_ranges(  # type: ignore
                        host_req_k_avail_sub_ranges_global,
                        as_cu_seqlens=False,
                    )
                    host_req_k_avail_sub_ranges_local = host_qk_ranges_global_for_this_rank.make_ranges_local(
                        host_req_k_avail_sub_ranges_global  # type: ignore
                    )
                    host_req_k_avail_sub_ranges_local_merged = (
                        host_req_k_avail_sub_ranges_local.merge()
                    )

                    # NOTE: this's supposed to never happen
                    # since we assure the consistence during shuffling
                    if host_req_k_avail_sub_ranges_local_merged.size > 1:
                        raise ValueError(
                            "The ffa does NOT support multiple k ranges for any single q range."
                        )

                    # append the merged available sub req k ranges to local attn args
                    local_attn_arg_q_ranges.append(host_qk_range_local, check=False)
                    local_attn_arg_k_ranges.append(
                        host_req_k_avail_sub_ranges_local_merged[0], check=False
                    )

                # set remote attn args for this (q, k) range pair, if unavailable
                # NOTE: if this req k range is totally available, 'find_overlap_ranges' will return []
                if host_req_k_unavail_sub_ranges_global:
                    host_req_k_unavail_sub_ranges_global = AttnRanges.from_ranges(  # type: ignore
                        host_req_k_unavail_sub_ranges_global,
                        as_cu_seqlens=False,
                    )
                    host_req_k_unavail_sub_ranges_local = remote_k_ranges_global_for_this_rank.make_ranges_local(
                        host_req_k_unavail_sub_ranges_global  # type: ignore
                    )
                    host_req_k_unavail_sub_ranges_local_merged = (
                        host_req_k_unavail_sub_ranges_local.merge()
                    )

                    # NOTE: this's supposed to never happen
                    # since we assure the consistence during shuffling
                    if host_req_k_unavail_sub_ranges_local_merged.size > 1:
                        raise ValueError(
                            "The ffa does NOT support multiple k ranges for any single q range."
                        )

                    # append the merged unavailable sub req k ranges to remote attn args
                    remote_attn_args_q_ranges_list[0].append(
                        host_qk_range_local, check=False
                    )
                    remote_attn_args_k_ranges_list[0].append(
                        host_req_k_unavail_sub_ranges_local_merged[0], check=False
                    )

            local_attn_arg_is_causal_mapping = [
                is_causal
            ] * local_attn_arg_q_ranges.size
            remote_attn_args_is_causal_mapping_list = [
                [is_causal] * remote_attn_args_q_ranges.size
                for remote_attn_args_q_ranges in remote_attn_args_q_ranges_list
            ]

        local_attn_arg_max_seqlen_q = local_attn_arg_q_ranges.max_seqlen
        local_attn_arg_max_seqlen_k = local_attn_arg_k_ranges.max_seqlen
        remote_attn_args_max_seqlen_q_list = [
            remote_attn_args_q_ranges.max_seqlen
            for remote_attn_args_q_ranges in remote_attn_args_q_ranges_list
        ]
        remote_attn_args_max_seqlen_k_list = [
            remote_attn_args_k_ranges.max_seqlen
            for remote_attn_args_k_ranges in remote_attn_args_k_ranges_list
        ]

        return (
            local_attn_arg_q_ranges,
            local_attn_arg_k_ranges,
            local_attn_arg_is_causal_mapping,
            local_attn_arg_max_seqlen_q,
            local_attn_arg_max_seqlen_k,
            remote_attn_args_q_ranges_list,
            remote_attn_args_k_ranges_list,
            remote_attn_args_is_causal_mapping_list,
            remote_attn_args_max_seqlen_q_list,
            remote_attn_args_max_seqlen_k_list,
        )

    @nvtx.instrument_nvtx
    def _safe_permute_qk_ranges_by_samples(
        self,
        q_ranges: "AttnRanges",
        k_ranges: "AttnRanges",
        perm_idxs: List[int],
    ) -> Tuple[bool, "AttnRanges", "AttnRanges"]:
        """Safely permute q_ranges and k_ranges following the samples permutation indices 'perm_idxs'

        NOTE:
            1. now, ffa does NOT support multiple k_ranges for any single q_range, so, some permutation may be invalid.
            2. this function assumes the input q_ranges and k_ranges are valid already
        """
        is_safe = True

        q_ranges = q_ranges.ranges  # type: ignore
        k_ranges = k_ranges.ranges  # type: ignore

        q_ranges_permed = [q_ranges[i] for i in perm_idxs]
        k_ranges_permed = [k_ranges[i] for i in perm_idxs]

        token_perm_idxs = [0] * q_ranges[-1][1]  # type: ignore
        idx = 0
        for s, e in q_ranges_permed:  # type: ignore
            for j in range(s, e):  # type: ignore
                token_perm_idxs[j] = idx
                idx += 1

        q_ranges_new = [  # type: ignore
            (token_perm_idxs[s], token_perm_idxs[e - 1] + 1) for s, e in q_ranges_permed  # type: ignore
        ]

        k_ranges_new = []
        for s, e in k_ranges_permed:  # type: ignore
            k_idxs = [token_perm_idxs[j] for j in range(s, e)]  # type: ignore
            min_k_idx, max_k_idx = min(k_idxs), max(k_idxs)
            if max_k_idx - min_k_idx != len(k_idxs) - 1:
                is_safe = False
                break
            k_ranges_new.append((min_k_idx, max_k_idx + 1))

        if is_safe:
            q_ranges = AttnRanges.from_ranges(q_ranges_new, as_cu_seqlens=True)
            k_ranges = AttnRanges.from_ranges(k_ranges_new, as_cu_seqlens=False)
        else:
            q_ranges = AttnRanges.from_ranges(q_ranges, as_cu_seqlens=True)
            k_ranges = AttnRanges.from_ranges(k_ranges, as_cu_seqlens=False)

        return is_safe, q_ranges, k_ranges
