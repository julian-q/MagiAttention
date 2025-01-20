import unittest
from unittest import TestCase

from zeus.common.enum import AttnMaskType
from zeus.common.mask import AttnMask
from zeus.common.ranges import AttnRanges
from zeus.testing.gt_dispatcher import GroundTruthDispatcher


class TestGroundTruthDispatcher(TestCase):
    def test_compute_self_attn_areas(self):
        # --------------      init sample meta      -------------- #

        q_ranges = AttnRanges.from_ranges(
            [
                (0, 6),
                (6, 9),
                (9, 12),
                (12, 16),
            ]
        )

        k_ranges = AttnRanges.from_ranges(
            [
                (0, 4),
                (4, 12),
                (12, 15),
                (1, 13),
            ]
        )

        attn_mask_type = [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
        ]

        chunk_size = 4
        overlap_degree = 1

        # --------------      init attn mask       -------------- #

        attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=q_ranges.end,
            total_seqlen_k=q_ranges.end,  # use the end of sq
        )

        # --------------      init gt dispatcher       -------------- #

        gt_dispatcher = GroundTruthDispatcher()
        global_bucket = gt_dispatcher._compute_self_attn_areas(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            chunk_size=chunk_size,
            overlap_degree=overlap_degree,
        )

        self.assertEqual(global_bucket.area, attn_mask.area)

        self_attn_mask = gt_dispatcher._self_attn_mask
        self.assertEqual(self_attn_mask, attn_mask)

        for chunk_mask, chunk in zip(
            gt_dispatcher._chunk_masks, global_bucket.q_chunks
        ):
            self.assertEqual(chunk_mask.area, chunk.area)


if __name__ == "__main__":
    unittest.main()
