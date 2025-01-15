from enum import Enum


class AttnType(Enum):
    SELF_ATTN = "self_attn"
    CROSS_ATTN = "cross_attn"


class AttnRole(Enum):
    QUERY = "query"
    KEY = "key"
    VALUE = "value"


class AttnMaskType(Enum):
    FULL = "full"
    CAUSAL = "causal"  # NOTE: this causal mask aligns to the bottom-right corner if it's not square
