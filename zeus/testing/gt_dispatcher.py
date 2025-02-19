from typing import List

import torch.nn as nn

from zeus.common.enum import AttnMaskType, DispatchAlgorithm
from zeus.common.mask import AttnMask
from zeus.common.range import AttnRange
from zeus.common.ranges import AttnRanges
from zeus.meta.container import AttnBucket, AttnChunk, AttnSlice


class GroundTruthDispatcher(nn.Module):
    """Balance dispatching tokens towards each cp rank along the sequence dimension for distributed attention
    where the "balance" has several folds of meaning:
        1. The number of tokens in each cp rank should be exactly balanced, i.e. equal to each other
        2. The computation cost, i.e. the area of the attn_mask matrix, in each cp rank should be roughly balanced
        3. The locality of the dispatched tokens in each cp rank should be maximized, i.e.

    NOTE: this is the ground-truth implementation of the dispatcher,
        which overwrites all of the intrinsic dispatching logics in a naive and inefficient way,
            so as to ONLY be used in testing, instead of production
    """

    def __init__(
        self,
        alg: DispatchAlgorithm = DispatchAlgorithm.MIN_HEAP,
        **alg_kwargs,
    ) -> None:
        super().__init__()

        self._self_attn_mask: AttnMask = None  # type: ignore
        self._cross_attn_mask: AttnMask = None  # type: ignore

        self._chunk_masks: List[AttnMask] = []  # type: ignore

    def _compute_self_attn_areas(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: List[AttnMaskType],
        chunk_size: int | None = None,
    ) -> AttnBucket:
        """Compute the self-attn areas, with constructing the global bucket,
        which is mainly consists of a list of all the chunks in ascending order, with a length of `cp_size`

        Args:
            q_ranges (AttnRanges): the query ranges
            k_ranges (AttnRanges): the key ranges
            attn_mask_type (List[AttnMaskType]): the attn mask type list
            chunk_size (int | None): the chunk size, which should be divisible by `cp_size`

        Returns:
            AttnBucket: the global bucket
        """

        ts = q_ranges.end
        if chunk_size is None:
            chunk_size = ts
        num_chunks = ts // chunk_size
        one_row_range = AttnRange(start=0, end=ts)

        # build the global mask
        self._self_attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=q_ranges.end,
            total_seqlen_k=q_ranges.end,  # NOTE: self-attn uses the end of sq
        )

        # build the global bucket
        global_bucket = AttnBucket()
        for chunk_idx in range(num_chunks):  # for each chunk
            chunk_start, chunk_end = (
                chunk_idx * chunk_size,
                (chunk_idx + 1) * chunk_size,
            )
            chunk = AttnChunk(chunk_id=chunk_idx)
            chunk_mask = self._self_attn_mask.make_sub_mask(
                q_range=AttnRange(
                    start=chunk_start,
                    end=chunk_end,
                ),
                k_range=one_row_range,
            )
            self._chunk_masks.append(chunk_mask)
            for slice_idx, (q_range, k_range, mask_type) in enumerate(
                chunk_mask.tuples()
            ):
                slice = AttnSlice(
                    slice_id=slice_idx,
                    q_range=q_range.offset(chunk_start),
                    k_range=k_range,
                    mask_type=mask_type,
                )

                # HACK: 后面会将计算面积的逻辑封装在AttnSlice中并且area只读, 这里保留直接设置area的功能
                slice.area = chunk_mask.calc_sub_area(
                    q_range=q_range,
                    k_range=k_range,
                )

                chunk.q_slices.append(slice)

            global_bucket.q_chunks.append(chunk)

        return global_bucket
