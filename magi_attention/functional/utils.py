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


def safe_subtract(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Safely subtracts two tensors,
    where the subtraction results of two -inf will be set to -inf.
    """

    eq = (a == b) & (a == float("-inf"))
    sub = a - b
    sub = torch.where(eq, torch.fill(sub, float("-inf")), sub)

    return sub


def softmax_bwd(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Standard backward func for `out = softmax(inp)`"""

    diag_out = torch.diag_embed(out)
    outer_out = torch.einsum("...ij, ...ik -> ...ijk", out, out)

    dinp = torch.einsum("...ij, ...ijk -> ...ik", dout, diag_out - outer_out)

    return dinp
