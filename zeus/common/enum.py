from enum import Enum


class AttnType(Enum):
    """This enum is used to specify the type of attention calculation we support"""

    SELF_ATTN = "self_attn"
    CROSS_ATTN = "cross_attn"


class AttnRole(Enum):
    """This enum is used to specify the tensor role in attention"""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"


class AttnMaskType(Enum):
    """This enum is used to specify the unit type of attention mask we support"""

    FULL = "full"
    CAUSAL = "causal"  # NOTE: this causal mask aligns to the bottom-right corner if it's not square


class AttnOverlapMode(Enum):
    """This enum is used to specify the overlap mode for multi-stage overlapping"""

    STATIC = "static"
    DYNAMIC = "dynamic"


class DispatchAlgorithm(Enum):
    """This enum is used to specify the algorithm for balanced dispatching"""

    LOWER_BOUND = "lower_bound"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BINARY_SEARCH = "binary_search"
    MIN_HEAP = "min_heap"


class OverlapAlgorithm(Enum):
    """This enum is used to specify the algorithm for multi-stage overlapping"""

    UNIFORM = "uniform"
    GREEDY = "greedy"
