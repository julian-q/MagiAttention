import contextlib

import torch


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


def instrument_nvtx(func):
    """
    Decorator that records an NVTX range for the duration of the function call.

    Args:
    - func: The function to be decorated.

    Returns:
    - Wrapped function that is now being profiled.
    """

    def wrapped_fn(*args, **kwargs):
        with add_nvtx_event(func.__qualname__):
            ret_val = func(*args, **kwargs)
        return ret_val

    return wrapped_fn
