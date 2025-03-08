import unittest
from unittest import TestCase

from zeus.common.range import AttnRange
from zeus.common.ranges import AttnRanges, is_valid_cu_seqlens


class TestAttnRanges(TestCase):
    def test_make_ranges_local(self):
        # ---------    case1: w/o truncate     --------- #
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_ranges = AttnRanges.from_ranges([(2, 8), (25, 32), (42, 45)])
        local_ranges = ranges.make_ranges_local(other_ranges)
        self.assertEqual(
            local_ranges, AttnRanges.from_ranges([(2, 8), (15, 22), (27, 30)])
        )

    def test_make_ranges_local_raises(self):
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_ranges = AttnRanges.from_ranges([(2, 8), (25, 36), (42, 45)])
        with self.assertRaisesRegex(
            ValueError,
            "The attn_range [25, 36) is not in the (even merged) attn_ranges [[0, 10), [20, 35), [40, 50)]",
        ):
            ranges.make_ranges_local(other_ranges)

    def test_make_range_local(self):
        # ---------    case1: w/o truncate     --------- #
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_range = AttnRange(23, 32)
        local_range = ranges.make_range_local(other_range)
        self.assertEqual(local_range, AttnRange(13, 22))

    def test_make_range_local_raises(self):
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])

        # 测试完全在ranges外的情况
        invalid_range = AttnRange(60, 70)
        with self.assertRaisesRegex(
            ValueError,
            "The attn_range [60, 70) is not in the (even merged) attn_ranges [[0, 10), [20, 35), [40, 50)]",
        ):
            ranges.make_range_local(invalid_range)

        # 测试部分重叠但不完全包含的情况
        invalid_range = AttnRange(5, 25)
        with self.assertRaisesRegex(
            ValueError,
            "The attn_range [5, 25) is not in the (even merged) attn_ranges [[0, 10), [20, 35), [40, 50)]",
        ):
            ranges.make_range_local(invalid_range)

        # 测试跨越多个ranges的情况
        invalid_range = AttnRange(25, 45)
        with self.assertRaisesRegex(
            ValueError,
            "The attn_range [25, 45) is not in the (even merged) attn_ranges [[0, 10), [20, 35), [40, 50)]",
        ):
            ranges.make_range_local(invalid_range)

    def test_find_hole_range(self):
        ranges1_list = [
            [(0, 10), (20, 30), (40, 50)],
            [
                [0, 28800],
                [0, 28800],
                [0, 28800],
                [0, 57600],
                [0, 57600],
                [0, 57600],
                [0, 57600],
                [0, 86400],
                [0, 86400],
                [0, 86400],
                [0, 115200],
                [0, 115200],
                [0, 115200],
                [0, 115200],
                [115200, 144000],
                [115200, 144000],
                [115200, 144000],
                [115200, 144000],
                [144000, 172800],
                [144000, 172800],
                [144000, 172800],
                [144000, 172800],
                [172800, 201600],
                [172800, 201600],
                [172800, 201600],
                [201600, 230400],
                [201600, 230400],
                [201600, 230400],
                [201600, 230400],
                [201600, 230400],
                [230400, 234040],
                [230400, 234040],
                [234040, 237680],
            ],
        ]

        ranges2_list = [
            [(5, 25), (35, 45)],
            [
                [4096, 6144],
                [12288, 14336],
                [20480, 22528],
                [30720, 32768],
                [38912, 40960],
                [49152, 51200],
                [57344, 57600],
                [57600, 59392],
                [77824, 79872],
                [83968, 86016],
                [92160, 94208],
                [94208, 96256],
                [104448, 106496],
                [112640, 114688],
                [118784, 120832],
                [120832, 122880],
                [131072, 133120],
                [139264, 141312],
                [147456, 149504],
                [149504, 151552],
                [157696, 159744],
                [165888, 167936],
                [182272, 184320],
                [186368, 188416],
                [194560, 196608],
                [202752, 204800],
                [210944, 212992],
                [219136, 221184],
                [227328, 229376],
                [229376, 230400],
                [230400, 231424],
                [233472, 234040],
                [234040, 235520],
            ],
        ]
        ans_list = [
            [(0, 5), (25, 30), (45, 50)],
            [
                [0, 4096],
                [6144, 12288],
                [14336, 20480],
                [22528, 30720],
                [32768, 38912],
                [40960, 49152],
                [51200, 57344],
                [59392, 77824],
                [79872, 83968],
                [86016, 92160],
                [96256, 104448],
                [106496, 112640],
                [114688, 118784],
                [122880, 131072],
                [133120, 139264],
                [141312, 147456],
                [151552, 157696],
                [159744, 165888],
                [167936, 182272],
                [184320, 186368],
                [188416, 194560],
                [196608, 202752],
                [204800, 210944],
                [212992, 219136],
                [221184, 227328],
                [231424, 233472],
                [235520, 237680],
            ],
        ]
        for ranges1_list, ranges2_list, ans_list in zip(
            ranges1_list, ranges2_list, ans_list
        ):
            ranges1 = AttnRanges.from_ranges(ranges1_list)
            ranges2 = AttnRanges.from_ranges(ranges2_list)
            hole_ranges = ranges1.find_hole_ranges(ranges2)
            self.assertEqual(hole_ranges, AttnRanges.from_ranges(ans_list))

    def test_find_overlap_range(self):
        ranges1_list = [
            [(0, 10), (20, 30), (40, 50)],
        ]
        ranges2_list = [
            [(5, 25), (35, 45)],
        ]
        ans_list = [
            [(5, 10), (20, 25), (40, 45)],
        ]
        for ranges1_list, ranges2_list, ans_list in zip(
            ranges1_list, ranges2_list, ans_list
        ):
            ranges1 = AttnRanges.from_ranges(ranges1_list)
            ranges2 = AttnRanges.from_ranges(ranges2_list)
            overlap_ranges = ranges1.find_overlap_ranges(ranges2)
            self.assertEqual(overlap_ranges, AttnRanges.from_ranges(ans_list))

    def test_from_cu_seqlens(self):
        cu_seqlens = [0, 10, 20, 30, 40, 50]
        seq_len = 50
        ranges = AttnRanges.from_cu_seqlens(cu_seqlens, seq_len)
        self.assertEqual(
            ranges,
            AttnRanges.from_ranges([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]),
        )

    def test_sort(self):
        ranges = AttnRanges.from_ranges(
            [(0, 10), (30, 35), (20, 30), (10, 25), (40, 50)]
        )
        self.assertFalse(ranges.is_sorted())
        sorted_ranges = ranges.sort()
        self.assertEqual(
            sorted_ranges,
            AttnRanges.from_ranges([(0, 10), (10, 25), (20, 30), (30, 35), (40, 50)]),
        )
        self.assertTrue(sorted_ranges.is_sorted())

    def test_merge(self):
        ranges = AttnRanges.from_ranges(
            [(0, 10), (10, 20), (20, 30), (30, 35), (40, 50)]
        )
        self.assertFalse(ranges.is_merged())
        merged_ranges = ranges.merge()
        self.assertEqual(merged_ranges, AttnRanges.from_ranges([(0, 35), (40, 50)]))
        self.assertTrue(merged_ranges.is_merged())

    def test_non_overlap(self):
        attn_ranges = AttnRanges.from_ranges([(8, 14), (5, 10)])
        self.assertFalse(attn_ranges.is_non_overlap())

        attn_ranges = AttnRanges.from_ranges([(8, 14), (14, 15)])
        self.assertTrue(attn_ranges.is_non_overlap())

        attn_ranges = AttnRanges.from_ranges([(8, 14), (3, 7)])
        self.assertTrue(attn_ranges.is_non_overlap())

    def test_chunk(self):
        # ----    case0: raise assert error when the ranges is overlapped   --- #

        attn_ranges = AttnRanges.from_ranges([(8, 14), (5, 10)])

        with self.assertRaises(
            AssertionError,
            msg="attn_ranges should be non-overlapped before chunking",
        ):
            attn_ranges.chunk(chunk_size=8)

        # ---------    case1: a single range     --------- #
        attn_ranges = AttnRanges.from_ranges([(5, 10)])

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [attn_ranges],  # itself all in one chunk
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(5, 9)]),
                AttnRanges.from_ranges([(9, 10)]),
            ],  # split itself into two chunks
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=1),
            [
                AttnRanges.from_ranges([(5, 6)]),
                AttnRanges.from_ranges([(6, 7)]),
                AttnRanges.from_ranges([(7, 8)]),
                AttnRanges.from_ranges([(8, 9)]),
                AttnRanges.from_ranges([(9, 10)]),
            ],  # split itself into several chunks
        )

        # ---------    case2: several small ranges     --------- #
        attn_ranges = AttnRanges.from_ranges([(2, 4), (6, 9), (10, 12), (14, 17)])

        # chunk with a xlarge chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=16),
            [attn_ranges],  # itself all in one chunk
        )

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [
                AttnRanges.from_ranges([(2, 4), (6, 9), (10, 12), (14, 15)]),
                AttnRanges.from_ranges([(15, 17)]),
            ],
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(2, 4), (6, 8)]),
                AttnRanges.from_ranges([(8, 9), (10, 12), (14, 15)]),
                AttnRanges.from_ranges([(15, 17)]),
            ],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=2),
            [
                AttnRanges.from_ranges([(2, 4)]),
                AttnRanges.from_ranges([(6, 8)]),
                AttnRanges.from_ranges([(8, 9), (10, 11)]),
                AttnRanges.from_ranges([(11, 12), (14, 15)]),
                AttnRanges.from_ranges([(15, 17)]),
            ],
        )

        # -----    case3: several small ranges + one long range   ------ #
        attn_ranges = AttnRanges.from_ranges([(3, 6), (8, 9), (10, 12), (14, 21)])

        # chunk with a xlarge chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=16),
            [attn_ranges],  # itself all in one chunk
        )

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [
                AttnRanges.from_ranges([(3, 6), (8, 9), (10, 12), (14, 16)]),
                AttnRanges.from_ranges([(16, 21)]),
            ],
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(3, 6), (8, 9)]),
                AttnRanges.from_ranges([(10, 12), (14, 16)]),
                AttnRanges.from_ranges([(16, 20)]),
                AttnRanges.from_ranges([(20, 21)]),
            ],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=2),
            [
                AttnRanges.from_ranges([(3, 5)]),
                AttnRanges.from_ranges([(5, 6), (8, 9)]),
                AttnRanges.from_ranges([(10, 12)]),
                AttnRanges.from_ranges([(14, 16)]),
                AttnRanges.from_ranges([(16, 18)]),
                AttnRanges.from_ranges([(18, 20)]),
                AttnRanges.from_ranges([(20, 21)]),
            ],
        )

        # -----    case4: one long range + several small ranges   ------ #
        attn_ranges = AttnRanges.from_ranges([(1, 2), (4, 12), (13, 15), (16, 19)])

        # chunk with a xlarge chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=16),
            [attn_ranges],  # itself all in one chunk
        )

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [
                AttnRanges.from_ranges([(1, 2), (4, 11)]),
                AttnRanges.from_ranges([(11, 12), (13, 15), (16, 19)]),
            ],
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(1, 2), (4, 7)]),
                AttnRanges.from_ranges([(7, 11)]),
                AttnRanges.from_ranges([(11, 12), (13, 15), (16, 17)]),
                AttnRanges.from_ranges([(17, 19)]),
            ],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=2),
            [
                AttnRanges.from_ranges([(1, 2), (4, 5)]),
                AttnRanges.from_ranges([(5, 7)]),
                AttnRanges.from_ranges([(7, 9)]),
                AttnRanges.from_ranges([(9, 11)]),
                AttnRanges.from_ranges([(11, 12), (13, 14)]),
                AttnRanges.from_ranges([(14, 15), (16, 17)]),
                AttnRanges.from_ranges([(17, 19)]),
            ],
        )

    def test_truncate(self):
        attn_ranges = AttnRanges.from_ranges([(9, 15), (20, 30), (30, 35), (40, 50)])

        # ---------    case1: w/o truncate     --------- #
        trunc_start, trunc_end = None, None
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_ranges, attn_ranges)

        # ---------    case2: with dummy truncate     --------- #
        trunc_start, trunc_end = 0, 57
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_ranges, attn_ranges)

        # ---------    case3: with left truncate     --------- #
        trunc_start, trunc_end = 25, None
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(
            trunc_ranges, AttnRanges.from_ranges([(25, 30), (30, 35), (40, 50)])
        )

        # ---------    case4: with right truncate     --------- #
        trunc_start, trunc_end = None, 31
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(
            trunc_ranges, AttnRanges.from_ranges([(9, 15), (20, 30), (30, 31)])
        )

        # ---------    case5: with left+right truncate     --------- #
        trunc_start, trunc_end = 25, 31
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_ranges, AttnRanges.from_ranges([(25, 30), (30, 31)]))

        # -----    case6: with left+right truncate but too left   ---- #
        trunc_start, trunc_end = 0, 5
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_ranges.is_empty())

        # -----    case7: with left+right truncate but too right   ---- #
        trunc_start, trunc_end = 53, 64
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_ranges.is_empty())

        # -----    case8: with left+right truncate but falling into a hole   ---- #
        trunc_start, trunc_end = 36, 39
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_ranges.is_empty())

    def test_is_valid_cu_seqlens(self):
        # NOTE: this test func also tests 'check_valid_cu_seqlens' implicitly

        # ---------    empty cu_seqlens always True     --------- #
        self.assertTrue(is_valid_cu_seqlens([], 0))
        self.assertTrue(is_valid_cu_seqlens([], 5))

        # ---------    valid cu_seqlens     --------- #
        self.assertTrue(is_valid_cu_seqlens([0, 23, 49, 58, 89], 89))
        self.assertTrue(is_valid_cu_seqlens([0, 89], 89))
        self.assertTrue(is_valid_cu_seqlens([0], 0))

        # ---------    invalid cu_seqlens w/o starting from 0     --------- #
        self.assertFalse(is_valid_cu_seqlens([23, 49, 58, 89], 89))

        # ---------    invalid cu_seqlens w/o monotonically increasing     --------- #
        self.assertFalse(is_valid_cu_seqlens([0, 50, 49, 58, 89], 89))
        self.assertFalse(is_valid_cu_seqlens([0, 49, 49, 58, 89], 89))

        # ---------    invalid cu_seqlens w/o ending at seq_len     --------- #
        self.assertFalse(is_valid_cu_seqlens([0, 23, 49, 58, 89], 90))


if __name__ == "__main__":
    unittest.main()
