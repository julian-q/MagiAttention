import torch
import torch.distributed
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from zeus.comm.primitive import group_cast_collective, group_reduce_collective
from zeus.comm.primitive._all_gather_v import (
    _get_dims_as_trans_with_dim0,
    _trans_with_dim0,
)
from zeus.comm.primitive._group_collective import unpermute_tensor
from zeus.testing.dist_common import DistTestBase, with_comms


class TestMultiCastCollective(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_naive_group_cast_like_a2a(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        input_tensor = torch.tensor(
            [self.rank] * self.world_size, dtype=dtype, device=device
        )
        output_tensor = torch.tensor([-1] * self.world_size, dtype=dtype, device=device)
        input_split_size_list = [1] * self.world_size
        output_split_size_list = [1] * self.world_size
        dst_indices_list = [[i] for i in range(self.world_size)]
        # dst_indices_list:
        # r0, r1, r2, r3: [[0],
        #                  [1],
        #                  [2],
        #                  [3]]
        src_index_list = [i for i in range(self.world_size)]
        # src_index_list:
        # r0, r1, r2, r3: [0, 1, 2, 3]

        work, post_process_fn = group_cast_collective(
            input=input_tensor,
            output=output_tensor,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=self.process_group,
            async_op=True,
        )

        work.wait()
        output_tensor = post_process_fn(output_tensor)

        expected_output = torch.tensor(
            [i for i in range(self.world_size)], dtype=dtype, device=device
        )
        # expected_output:
        # r0, r1, r2, r3: [0, 1, 2, 3]
        assert torch.equal(output_tensor, expected_output)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_naive_group_reduce(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        # Do group-reduce collective as following:
        # r0: input: [0, 0, 0, 0] output: [0, 0, 0, 0] ---> [0, 1, 2, 3]
        # r1: input: [1, 1, 1, 1] output: [1, 1, 1, 1] ---> [1, 2, 3, 4]
        # r2: input: [2, 2, 2, 2] output: [2, 2, 2, 2] ---> [2, 3, 4, 5]
        # r3: input: [3, 3, 3, 3] output: [3, 3, 3, 3] ---> [3, 4, 5, 6]
        expected_tensor_per_rank = [
            torch.tensor([0, 1, 2, 3], dtype=dtype, device=device),
            torch.tensor([1, 2, 3, 4], dtype=dtype, device=device),
            torch.tensor([2, 3, 4, 5], dtype=dtype, device=device),
            torch.tensor([3, 4, 5, 6], dtype=dtype, device=device),
        ]

        input_tensor = torch.tensor(
            [self.rank] * self.world_size, dtype=dtype, device=device
        )
        output_tensor = torch.tensor(
            [self.rank] * self.world_size, dtype=dtype, device=device
        )
        input_split_size_list = [1] * self.world_size
        output_split_size_list = [1] * self.world_size
        src_indices_list = [[i] for i in range(self.world_size)]
        # src_indices_list:
        # r0, r1, r2, r3: [[0],
        #                  [1],
        #                  [2],
        #                  [3]]
        dst_index_list = [i for i in range(self.world_size)]
        # dst_index_list:
        # r0, r1, r2, r3: [0, 1, 2, 3]

        work, post_process_fn = group_reduce_collective(
            input=input_tensor,
            output=output_tensor,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_index_list=dst_index_list,
            src_indices_list=src_indices_list,
            group=self.process_group,
            async_op=True,
        )

        work.wait()
        output_tensor = post_process_fn(output_tensor)

        assert torch.equal(output_tensor, expected_tensor_per_rank[self.rank])

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_group_cast_collective(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        # Do multi-cast collective as following:
        # r0: [0, 1, 2, 3] -------> [5, 9, 13]
        # r1: [4, 5, 6, 7] -------> [0, 1, 10, 11, 2, 12, 13]
        # r2: [8, 9, 10, 11] -----> [2, 3, 6, 7, 14, 15]
        # r3: [12, 13, 14, 15] ---> [4, 5, 8, 9]
        expected_tensor_per_rank = [
            torch.tensor([5, 9, 13], dtype=dtype, device=device),
            torch.tensor([0, 1, 10, 11, 2, 12, 13], dtype=dtype, device=device),
            torch.tensor([2, 3, 6, 7, 14, 15], dtype=dtype, device=device),
            torch.tensor([4, 5, 8, 9], dtype=dtype, device=device),
        ]
        input_split_size_list_per_rank = [[2, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2]]
        output_split_size_list_per_rank = [
            [1, 1, 1],
            [2, 2, 1, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
        ]
        dst_indices_list_per_rank = [
            [[1], [1, 2], [2]],
            [[3], [0, 3], [2]],
            [[3], [0, 3], [1]],
            [[1], [0, 1], [2]],
        ]
        src_index_list_per_rank = [[1, 2, 3], [0, 2, 0, 3], [0, 0, 1, 3], [1, 1, 2, 2]]
        input_tensor = torch.tensor(
            [
                i
                for i in range(
                    self.world_size * self.rank, self.world_size * (self.rank + 1)
                )
            ],
            dtype=dtype,
            device=device,
        )
        output_tensor = torch.full_like(
            expected_tensor_per_rank[self.rank], -1, dtype=dtype, device=device
        )
        input_split_size_list = input_split_size_list_per_rank[self.rank]
        output_split_size_list = output_split_size_list_per_rank[self.rank]
        dst_indices_list = dst_indices_list_per_rank[self.rank]
        src_index_list = src_index_list_per_rank[self.rank]

        expected_tensor = expected_tensor_per_rank[self.rank]

        work, post_process_fn = group_cast_collective(
            input=input_tensor,
            output=output_tensor,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=self.process_group,
            async_op=True,
        )
        work.wait()
        output_tensor = post_process_fn(output_tensor)
        assert torch.equal(output_tensor, expected_tensor)

    def test_trans_with_dim0(self):
        # ---------    high-dim tensor     --------- #

        x = torch.arange(2 * 3 * 4).reshape(2, 3, 4).contiguous()

        y0 = _trans_with_dim0(x, dim=0)
        y0_ = _trans_with_dim0(x, dim=-3)
        self.assertTrue(y0.is_contiguous())
        self.assertTrue(torch.equal(y0, x))
        self.assertTrue(torch.equal(y0_, y0))
        self.assertTrue(y0.data_ptr() == x.data_ptr())  # same storage

        y1 = _trans_with_dim0(x, dim=1)
        y1_ = _trans_with_dim0(x, dim=-2)
        self.assertTrue(y1.is_contiguous())
        self.assertTrue(torch.equal(y1, x.transpose(0, 1)))
        self.assertTrue(torch.equal(y1_, y1))
        self.assertFalse(y1.data_ptr() == x.data_ptr())  # different storage

        y2 = _trans_with_dim0(x, dim=2)
        y2_ = _trans_with_dim0(x, dim=-1)
        self.assertTrue(y2.is_contiguous())
        self.assertTrue(torch.equal(y2, x.transpose(0, 2)))
        self.assertTrue(torch.equal(y2_, y2))
        self.assertFalse(y2.data_ptr() == x.data_ptr())  # different storage

        # ---------    1-dim tensor     --------- #

        x = torch.arange(
            12,
        )

        y0 = _trans_with_dim0(x, dim=0)
        self.assertTrue(y0.is_contiguous())
        self.assertTrue(torch.equal(y0, x))
        self.assertTrue(y0.data_ptr() == x.data_ptr())  # same storage

        y1 = _trans_with_dim0(x, dim=-1)
        self.assertTrue(y1.is_contiguous())
        self.assertTrue(torch.equal(y1, x))
        self.assertTrue(y1.data_ptr() == x.data_ptr())  # same storage

    def test_get_dims_as_trans_with_dim0(self):
        # ---------    high-dim shape     --------- #

        x_shape = [2, 3, 4, 5]

        this_dim, other_dims = _get_dims_as_trans_with_dim0(x_shape, dim=0)
        self.assertEqual(this_dim, 2)
        self.assertEqual(other_dims, [3, 4, 5])

        this_dim, other_dims = _get_dims_as_trans_with_dim0(x_shape, dim=1)
        self.assertEqual(this_dim, 3)
        self.assertEqual(other_dims, [2, 4, 5])

        this_dim, other_dims = _get_dims_as_trans_with_dim0(x_shape, dim=2)
        self.assertEqual(this_dim, 4)
        self.assertEqual(other_dims, [3, 2, 5])

        this_dim, other_dims = _get_dims_as_trans_with_dim0(x_shape, dim=3)
        this_dim_, other_dims_ = _get_dims_as_trans_with_dim0(x_shape, dim=-1)
        self.assertEqual(this_dim, 5)
        self.assertEqual(other_dims, [3, 4, 2])
        self.assertEqual(this_dim_, this_dim)
        self.assertEqual(other_dims_, other_dims)

        # ---------    1-dim shape     --------- #

        x_shape = [12]

        this_dim, other_dims = _get_dims_as_trans_with_dim0(x_shape, dim=0)
        self.assertEqual(this_dim, 12)
        self.assertEqual(other_dims, [])

        this_dim, other_dims = _get_dims_as_trans_with_dim0(x_shape, dim=-1)
        self.assertEqual(this_dim, 12)
        self.assertEqual(other_dims, [])

        # ---------    invalid dim     --------- #

        x_shape = [2, 3, 4]

        with self.assertRaises(
            AssertionError,
            msg="dim should be in [0, len(x_shape) - 1) or -1",
        ):
            _get_dims_as_trans_with_dim0(x_shape, dim=4)

        with self.assertRaises(
            AssertionError,
            msg="dim should be in [0, len(x_shape) - 1) or -1",
        ):
            _get_dims_as_trans_with_dim0(x_shape, dim=-2)

    def test_unpermute_tensor(self):
        # NOTE: this test func also tests 'safe_cat' implicitly

        # ---------    normal unperm idxs     --------- #
        x = torch.randn(6, 3, 4)

        unperm_idxs1 = [0, 2, 1]
        split_sizes1 = [2, 1, 3]
        y1 = unpermute_tensor(
            x, unpermute_index_list=unperm_idxs1, tensor_size_list=split_sizes1
        )
        self.assertTrue(y1.is_contiguous())
        self.assertTrue(
            torch.equal(
                y1,
                torch.cat(
                    [  # split_sizes_after_unperm = (2,3,1)
                        x[:2],
                        x[-3:],
                        x[2:3],
                    ]
                ),
            )
        )

        unperm_idxs2 = [2, 1, 0]
        split_sizes2 = [2, 3, 1]
        y2 = unpermute_tensor(
            x, unpermute_index_list=unperm_idxs2, tensor_size_list=split_sizes2
        )
        self.assertTrue(y2.is_contiguous())
        self.assertTrue(
            torch.equal(
                y2,
                torch.cat(
                    [  # split_sizes_after_unperm = (1,3,2)
                        x[-1:],
                        x[2:-1],
                        x[:2],
                    ]
                ),
            )
        )

        unperm_idxs3 = [2, 0, 1]
        split_sizes3 = [3, 1, 2]
        y3 = unpermute_tensor(
            x, unpermute_index_list=unperm_idxs3, tensor_size_list=split_sizes3
        )
        self.assertTrue(y3.is_contiguous())
        self.assertTrue(
            torch.equal(
                y3,
                torch.cat(
                    [  # split_sizes_after_unperm = (2,3,1)
                        x[-2:],
                        x[:3],
                        x[3:4],
                    ]
                ),
            )
        )

        # ---------    empty unperm idxs     --------- #

        x = torch.randn(6, 3, 4)
        torch.manual_seed(42)
        emp = torch.empty(0, 3, 4)

        torch.manual_seed(42)
        y = unpermute_tensor(x, unpermute_index_list=[], tensor_size_list=[1, 2, 3])
        self.assertTrue(y.is_contiguous())
        self.assertTrue(torch.equal(y, emp))


if __name__ == "__main__":
    run_tests()
