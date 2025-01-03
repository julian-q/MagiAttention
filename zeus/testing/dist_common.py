from functools import wraps
from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase

DEVICE_TYPE = (
    "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
)
PG_BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"

NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())


class DistTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        return PG_BACKEND

    def init_pg(self) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            raise RuntimeError(
                f"nccl backend requires {self.world_size} GPUs, but only {torch.cuda.device_count()} are available"
            )

        if self.backend not in ["nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl"]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",  # noqa
        )

        # set device for nccl pg for collectives
        if "nccl" in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


TestFunc = Callable[..., Any]


# wrapper to initialize comms (processgroup)
def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(
        self, *args: Tuple[object], **kwargs: Dict[str, Any]  # type: ignore[misc]
    ) -> None:
        # if backend not specified, and cuda available, then use nccl, else gloo
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        self.init_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

    return wrapper
