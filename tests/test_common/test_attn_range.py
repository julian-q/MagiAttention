import unittest
from unittest import TestCase

from dffa.common.range import AttnRange, RangeError


class TestAttnRange(TestCase):
    def test_simple_properties(self):
        # ---------    init an attn range     --------- #

        attn_range = AttnRange(0, 10)
        self.assertEqual(attn_range.start, 0)
        self.assertEqual(attn_range.end, 10)
        self.assertEqual(attn_range.seqlen, 10)
        self.assertEqual(attn_range.seqlen, 10)
        self.assertEqual(len(attn_range), 10)
        self.assertFalse(attn_range.is_empty())

        # ---------    change its start     --------- #

        with self.assertRaises(
            RangeError,
            msg="Against the rule: 'start >= 0'",
        ):
            attn_range.start = -1

        with self.assertRaises(
            RangeError,
            msg="Against the rule: 'start <= end'",
        ):
            attn_range.start = 11

        attn_range.start = 4
        self.assertEqual(attn_range.start, 4)
        self.assertEqual(attn_range.end, 10)
        self.assertEqual(attn_range.seqlen, 6)
        self.assertEqual(attn_range.seqlen, 6)
        self.assertEqual(len(attn_range), 6)

        # ---------    change its end     --------- #

        with self.assertRaises(
            RangeError,
            msg="Against the rule: 'start <= end'",
        ):
            attn_range.end = 3

        attn_range.end = 12
        self.assertEqual(attn_range.start, 4)
        self.assertEqual(attn_range.end, 12)
        self.assertEqual(attn_range.seqlen, 8)
        self.assertEqual(attn_range.seqlen, 8)
        self.assertEqual(len(attn_range), 8)

        # ---------    test empty range     --------- #

        attn_range.start = 5
        attn_range.end = 5
        self.assertEqual(attn_range.start, 5)
        self.assertEqual(attn_range.end, 5)
        self.assertEqual(attn_range.seqlen, 0)
        self.assertEqual(attn_range.seqlen, 0)
        self.assertEqual(len(attn_range), 0)
        self.assertTrue(attn_range.is_empty())

        # ---------    test read-only properties     --------- #

        with self.assertRaises(
            AttributeError,
            msg="The 'size' property is read-only",
        ):
            attn_range.seqlen = 3

        with self.assertRaises(
            AttributeError,
            msg="The 'seqlen' property is read-only",
        ):
            attn_range.seqlen = 3

        # ---------    test range equal with some other simple APIs    --------- #
        attn_range2 = AttnRange(7, 9)
        self.assertNotEqual(attn_range, attn_range2)

        attn_range3 = AttnRange(0, 0)
        self.assertTrue(attn_range3.is_empty())
        self.assertNotEqual(attn_range, attn_range3)  # both empty, but not equal

        attn_range4 = attn_range3.offset(5)
        self.assertTrue(attn_range4.is_empty())
        self.assertEqual(attn_range, attn_range4)

        naive_attn_range4 = attn_range4.to_naive_range()
        self.assertEqual(naive_attn_range4, (5, 5))
        self.assertNotEqual(
            attn_range, naive_attn_range4
        )  # the same content, but not the same type
        attn_range4_from_naive = AttnRange.from_range(
            naive_attn_range4
        )  # another contructor, from naive range
        self.assertEqual(attn_range, attn_range4_from_naive)

    def test_set_ops(self):
        # ---------    init several attn ranges     --------- #

        # the attn ranges below can be mapped into the axis as below:
        #      |---------------------| r1:(2,9)
        #               |-------| r2:(4,7)
        # |----| r3:(0,2)
        #                              |-------| r4:(10,13)
        #                         |---------| r5:(8,12)

        attn_range1 = AttnRange(2, 9)
        attn_range2 = AttnRange.from_range((4, 7))
        attn_range3 = AttnRange(0, 2)
        attn_range4 = AttnRange.from_range((10, 13))
        attn_range5 = AttnRange(8, 12)

        # ---------    test is subrange of     --------- #

        self.assertTrue(attn_range2.is_subrange_of(attn_range1))
        self.assertFalse(attn_range3.is_subrange_of(attn_range1))
        self.assertFalse(attn_range5.is_subrange_of(attn_range1))

        # ---------    test intersect     --------- #

        self.assertEqual(attn_range1.intersect(attn_range2), attn_range2)
        self.assertTrue(attn_range1.intersect(attn_range4).is_empty())
        self.assertEqual(attn_range1.intersect(attn_range5), AttnRange(8, 9))
        self.assertEqual(attn_range4.intersect(attn_range5), AttnRange(10, 12))
        self.assertTrue(attn_range4.is_overlap_with(attn_range5))
        self.assertFalse(attn_range2.is_overlap_with(attn_range5))

        # ---------    test diff by     --------- #

        # case1: two disjoint ranges
        attn_diff_ranges_13 = attn_range1.diff_by(attn_range3)
        self.assertEqual(len(attn_diff_ranges_13), 1)
        self.assertEqual(attn_diff_ranges_13[0], attn_range3)
        attn_diff_ranges_31 = attn_range3.diff_by(attn_range1)
        self.assertEqual(len(attn_diff_ranges_31), 1)
        self.assertEqual(attn_diff_ranges_31[0], attn_range1)

        # case2: range and its sub range
        self.assertTrue(attn_range1.diff_by(attn_range2) == [])
        attn_diff_ranges_21 = attn_range2.diff_by(attn_range1)
        self.assertEqual(len(attn_diff_ranges_21), 2)
        self.assertEqual(attn_diff_ranges_21[0], AttnRange(2, 4))
        self.assertEqual(attn_diff_ranges_21[1], AttnRange(7, 9))

        # case3: overlapped ranges but not case2
        attn_diff_ranges_15 = attn_range1.diff_by(attn_range5)
        self.assertEqual(len(attn_diff_ranges_15), 1)
        self.assertEqual(attn_diff_ranges_15[0], AttnRange(9, 12))
        attn_diff_ranges_51 = attn_range5.diff_by(attn_range1)
        self.assertEqual(len(attn_diff_ranges_51), 1)
        self.assertEqual(attn_diff_ranges_51[0], AttnRange(2, 8))

    def test_truncate(self):
        attn_range = AttnRange(9, 15)

        # ---------    case1: w/o truncate     --------- #
        trunc_start, trunc_end = None, None
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, attn_range)

        # ---------    case2: with dummy truncate     --------- #
        trunc_start, trunc_end = 0, 20
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, attn_range)

        # ---------    case3: with left truncate     --------- #
        trunc_start, trunc_end = 11, None
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRange(11, 15))

        # ---------    case4: with right truncate     --------- #
        trunc_start, trunc_end = None, 13
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRange(9, 13))

        # ---------    case5: with left+right truncate     --------- #
        trunc_start, trunc_end = 11, 13
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRange(11, 13))

        # -----    case6: with left+right truncate but too left   ---- #
        trunc_start, trunc_end = 1, 7
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_range.is_empty())

        # -----    case7: with left+right truncate but too right   ---- #
        trunc_start, trunc_end = 17, 23
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_range.is_empty())


if __name__ == "__main__":
    unittest.main()
