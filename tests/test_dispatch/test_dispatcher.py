import torch
import torch.distributed
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from zeus.common.enum import AttnMaskType, AttnType
from zeus.dispatch import SequenceDispatcher
from zeus.testing.dist_common import DistTestBase, with_comms

WORLD_SIZE = 4
SEED = 42


class TestDispatcher(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return WORLD_SIZE

    @property
    def seed(self) -> int:
        return SEED

    @skip_if_lt_x_gpu(WORLD_SIZE)
    @with_comms
    def test_dispatch_and_undispatch(self):
        # --------------      setup       --------------#

        rank = self.rank
        cp_size = self.world_size
        world_group = self.process_group
        manual_seed = self.seed
        device = torch.cuda.current_device()
        torch.manual_seed(manual_seed)

        world_group_gloo = dist.new_group(ranks=list(range(cp_size)), backend="gloo")

        # --------------      init sample meta      --------------#

        q_ranges = [
            (0, 1),
            (1, 5),
            (5, 12),
            (12, 16),
        ]

        k_ranges = [
            (0, 1),
            (1, 4),
            (5, 10),
            (12, 13),
        ]

        attn_type = AttnType.SELF_ATTN
        attn_mask_type = [AttnMaskType.FULL for _ in range(len(q_ranges))]

        chunk_size = 4
        overlap_degree = 1

        seq_dim = 0

        shuffle_times = 100
        shuffle_timeout = 10
        shuffle_seed = 42

        # --------------      init global data       --------------#

        global_q = torch.arange(q_ranges[-1][1] * 2).view(-1, 2).to(device)  # (sq, 2)
        global_k = global_q * -1  # (sq, 2), due to self-attn

        # --------------      init dispatcher       --------------#

        dispatcher = SequenceDispatcher()

        # --------------      compute meta       --------------#

        meta_q, meta_k = dispatcher.compute_meta(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type=attn_type,
            attn_mask_type=attn_mask_type,
            cp_rank=rank,
            cp_size=cp_size,
            cp_group_nccl=world_group,
            cp_group_gloo=world_group_gloo,
            chunk_size=chunk_size,
            overlap_degree=overlap_degree,
            shuffle_times=shuffle_times,
            shuffle_timeout=shuffle_timeout,
            shuffle_seed=shuffle_seed,
        )

        # --------------      dispatch data       --------------#

        local_q = dispatcher.dispatch(
            x_global=global_q,
            meta=meta_q,
            seq_dim=seq_dim,
        )

        local_k = dispatcher.dispatch(
            x_global=global_k,
            meta=meta_k,
            seq_dim=seq_dim,
        )

        # --------------      undispatch data       --------------#

        global_q_und = dispatcher.undispatch(
            x_local=local_q,
            meta=meta_q,
            seq_dim=seq_dim,
        )

        global_k_und = dispatcher.undispatch(
            x_local=local_k,
            meta=meta_k,
            seq_dim=seq_dim,
        )

        # --------------      check       --------------#

        assert torch.equal(global_q_und, global_q)
        assert torch.equal(global_k_und, global_k)


if __name__ == "__main__":
    run_tests()
