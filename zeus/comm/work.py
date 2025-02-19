from dataclasses import dataclass, field
from typing import Any, Callable

from torch.distributed import Work


@dataclass
class WorkWithPostProcessFn:
    """A dist work with a post process func to call after waiting
    HACK: this is a hack way to apply async op
        that the user needs to call post process manually after waiting
    NOTE: this is a disposable object,
        once the work + post process fn are both done,
        the user are not supposed to use this object anymore
    """

    work: Work | None = None
    post_process_fn: Callable = field(
        default_factory=lambda: lambda *args, **kwargs: None
    )
    sync: bool = False

    def __post_init__(self):
        # the flag to note if
        # the optional work + post process fn are both done
        # to avoid repeatedly calling post process fn
        self._work_done = False

        # if sync mode, the given work needs to wait immediately
        if self.sync and self.work is not None:
            self.work.wait()
            self.work = None

    def wait_post_process(self, *args, **kwargs) -> Any:
        if self._work_done:
            raise RuntimeError("Work has already been done.")

        if self.work is not None:
            self.work.wait()
            self.work = None

        ret = self.post_process_fn(*args, **kwargs)

        self._work_done = True

        return ret
