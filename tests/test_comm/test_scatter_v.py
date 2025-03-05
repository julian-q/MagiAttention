import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from zeus.testing.dist_common import DistTestBase


class TestScatterV(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    def test_scatter_v(self):
        # TODO: add unitest for scatter_v
        pass


if __name__ == "__main__":
    run_tests()
