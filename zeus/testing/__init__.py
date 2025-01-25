from functools import partial
from typing import Any, Callable

from . import dist_common
from .gt_dispatcher import GroundTruthDispatcher
from .precision import assert_close, get_mask_from_ranges, torch_attn_ref

__all__ = [
    "dist_common",
    "GroundTruthDispatcher",
    "assert_close",
    "get_mask_from_ranges",
    "torch_attn_ref",
    "parameterize",
]


def parameterize(argument: str, values: list[Any]) -> Callable:
    """
    This function is to simulate the same behavior as pytest.mark.parameterize. As
    we want to avoid the number of distributed network initialization, we need to have
    this extra decorator on the function launched by torch.multiprocessing.

    If a function is wrapped with this wrapper, non-parametrized arguments must be keyword arguments,
    positional arguments are not allowed.

    Usage::

        # Example 1:
        @parameterize('person', ['xavier', 'davis'])
        def say_something(person, msg):
            print(f'{person}: {msg}')

        say_something(msg='hello')

        # This will generate output:
        # > xavier: hello
        # > davis: hello

        # Example 2:
        @parameterize('person', ['xavier', 'davis'])
        @parameterize('msg', ['hello', 'bye', 'stop'])
        def say_something(person, msg):
            print(f'{person}: {msg}')

        say_something()

        # This will generate output:
        # > xavier: hello
        # > xavier: bye
        # > xavier: stop
        # > davis: hello
        # > davis: bye
        # > davis: stop

    Args:
        argument (str): the name of the argument to parameterize
        values (list[Any]): a list of values to iterate for this argument
    """

    def _wrapper(func):
        def _execute_function_by_param(*args, **kwargs):
            for val in values:
                arg_map = {argument: val}
                partial_func = partial(func, **arg_map)
                partial_func(*args, **kwargs)

        return _execute_function_by_param

    return _wrapper
