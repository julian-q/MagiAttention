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

from ._calc_attn_meta import calc_attn_meta_from_dispatch_meta
from ._calc_dispatch_meta import calc_dispatch_meta_from_qk_ranges

__all__ = [
    "calc_dispatch_meta_from_qk_ranges",
    "calc_attn_meta_from_dispatch_meta",
]
