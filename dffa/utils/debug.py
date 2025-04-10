import os

import debugpy
import torch.distributed as dist


def debugpy_listen():  # pragma: no cover
    ENABLE_REMOTE_DEBUG = os.environ.get("ENABLE_REMOTE_DEBUG", "false").lower()
    if ENABLE_REMOTE_DEBUG != "false":
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        debug_ranks = []
        if ENABLE_REMOTE_DEBUG == "true":
            debug_ranks = [0]
        elif ENABLE_REMOTE_DEBUG == "all":
            debug_ranks = [i for i in range(world_size)]
        else:
            debug_ranks = [int(i) for i in ENABLE_REMOTE_DEBUG.split(",")]

        if rank in debug_ranks:
            debug_port = 37777 + int(rank)
            print(f"[rank {rank}] Starting remote debug on port {debug_port}")
            debugpy.listen(("127.0.0.1", debug_port))
            debugpy.wait_for_client()
            print(f"[rank {rank}] Remote debug attached")
