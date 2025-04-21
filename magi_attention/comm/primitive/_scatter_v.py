# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist

__all__ = ["scatter_v"]


def scatter_v(
    input: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int = 0,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    rank = dist.get_rank(group)

    if split_sizes is None:
        input_split = torch.chunk(input, chunks=dist.get_world_size(group), dim=dim)
    else:
        input_split = torch.split(input, split_sizes, dim=dim)

    return input_split[rank]
