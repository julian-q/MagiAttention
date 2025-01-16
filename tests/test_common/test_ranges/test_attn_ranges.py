import unittest
from unittest import TestCase

from zeus.common.ranges import AttnRange, AttnRanges


class TestFindHoleRanges(TestCase):
    def test_make_ranges_local(self):
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

    def test_make_range_lcoal(self):
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_range = AttnRange(3, 9)
        local_range = ranges.make_range_local(other_range)
        self.assertEqual(local_range, AttnRange(3, 9))

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


if __name__ == "__main__":
    unittest.main()
