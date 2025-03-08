import contextlib
from functools import wraps
from typing import Any, Callable, TypeVar, cast

import torch

# fixed the mypy type check missing bug
# when a func is wrapped
# issue: https://stackoverflow.com/questions/65621789/mypy-untyped-decorator-makes-function-my-method-untyped
F = TypeVar("F", bound=Callable[..., Any])


@contextlib.contextmanager
def add_nvtx_event(event_name):
    """
    Context manager to add an NVTX event around a code block.

    Args:
    - event_name: The name of the event to be recorded.
    """
    torch.cuda.nvtx.range_push(event_name)
    yield
    torch.cuda.nvtx.range_pop()


def instrument_nvtx(func: F) -> F:
    """
    Decorator that records an NVTX range for the duration of the function call.

    Args:
    - func: The function to be decorated.

    Returns:
    - Wrapped function that is now being profiled.
    """

    @wraps(func)
    def wrapped_fn(*args, **kwargs):
        with add_nvtx_event(func.__qualname__):
            ret_val = func(*args, **kwargs)
        return ret_val

    return cast(F, wrapped_fn)
