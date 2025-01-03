import torch
import torch.distributed
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from zeus.comm.primitive import multi_cast_collective
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
    def test_naive_a2a(self):
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

        work = multi_cast_collective(
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
        expected_output = torch.tensor(
            [i for i in range(self.world_size)], dtype=dtype, device=device
        )
        # expected_output:
        # r0, r1, r2, r3: [0, 1, 2, 3]
        assert torch.equal(output_tensor, expected_output)

    @skip_if_lt_x_gpu(4)
    @with_comms
    def test_multi_cast_collective(self):
        dtype = torch.int32
        device = torch.cuda.current_device()

        # Do multi-cast collective as following:
        # r0: [0, 1, 2, 3] -------> [5, 9, 13]
        # r1: [4, 5, 6, 7] -------> [0, 1, 2, 10, 11, 12, 13]
        # r2: [8, 9, 10, 11] -----> [2, 3, 6, 7, 14, 15]
        # r3: [12, 13, 14, 15] ---> [4, 5, 8, 9]
        expected_tensor_per_rank = [
            torch.tensor([5, 9, 13], dtype=dtype, device=device),
            torch.tensor([0, 1, 2, 10, 11, 12, 13], dtype=dtype, device=device),
            torch.tensor([2, 3, 6, 7, 14, 15], dtype=dtype, device=device),
            torch.tensor([4, 5, 8, 9], dtype=dtype, device=device),
        ]
        input_split_size_list_per_rank = [[2, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2]]
        output_split_size_list_per_rank = [
            [1, 1, 1],
            [2, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
        ]
        dst_indices_list_per_rank = [
            [[1], [1, 2], [2]],
            [[3], [0, 3], [2]],
            [[3], [0, 3], [1]],
            [[1], [0, 1], [2]],
        ]
        src_index_list_per_rank = [[1, 2, 3], [0, 0, 2, 3], [0, 0, 1, 3], [1, 1, 2, 2]]
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

        work = multi_cast_collective(
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

        assert torch.equal(output_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
