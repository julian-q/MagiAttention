from enum import Enum


class AttnType(Enum):
    """The enum used to specify the type of attention calculation we support"""

    SELF_ATTN = "self_attn"
    CROSS_ATTN = "cross_attn"


class AttnRole(Enum):
    """The enum used to specify the tensor role in attention"""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"


class AttnMaskType(Enum):
    """The enum used to specify the unit type of attention mask we support"""

    FULL = "full"
    CAUSAL = "causal"  # NOTE: The causal mask aligns to the bottom-right corner if it's not square


class AttnOverlapMode(Enum):
    """The enum used to specify the overlap mode for multi-stage overlapping"""

    STATIC = "static"
    DYNAMIC = "dynamic"


class DispatchAlgType(Enum):
    """The enum used to specify the algorithm type for load-balanced dispatching"""

    LOWER_BOUND = "lower_bound"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BINARY_SEARCH = "binary_search"
    MIN_HEAP = "min_heap"


class OverlapAlgType(Enum):
    """The enum used to specify the algorithm type for multi-stage overlapping"""

    UNIFORM = "uniform"
    GREEDY = "greedy"
