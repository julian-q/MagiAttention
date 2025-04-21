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

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta._calc_dispatch_meta import _calc_self_attn_areas


def calculate_attn_flops(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    num_heads_q: int,
    head_dim: int,
) -> dict[str, float]:
    attn_area = _calc_self_attn_areas(
        q_ranges,
        k_ranges,
        attn_mask_type,
        num_chunks=1,
        chunk_size=total_seqlen_q,
    ).area

    flops_fwd = 4 * attn_area * num_heads_q * head_dim
    flops_bwd = flops_fwd * 2.5  # 2.0(bwd) + 0.5(recompute)
    flops_1f1b = flops_fwd + flops_bwd

    return {
        "fwd": flops_fwd,
        "bwd": flops_bwd,
        "1f1b": flops_1f1b,
    }
