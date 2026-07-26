from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Callable, ParamSpec, TypeVar


Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
WORKER_COUNT_ENV = "CONTRAST_FRAME_WORKERS"


def available_parallelism() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def default_worker_count() -> int:
    configured = os.environ.get(WORKER_COUNT_ENV)
    if configured is not None:
        try:
            worker_count = int(configured)
        except ValueError as exc:
            raise ValueError(f"{WORKER_COUNT_ENV} must be an integer") from exc
        if worker_count < 1:
            raise ValueError(f"{WORKER_COUNT_ENV} must be at least one")
        return worker_count
    return max(1, (available_parallelism() + 1) // 2)


class AdaptiveFrameExecutor:
    """A lazily growing, bounded pool for compute work from any frame stage."""

    def __init__(self, max_workers: int | None = None, max_pending: int | None = None) -> None:
        self.max_workers = default_worker_count() if max_workers is None else max_workers
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self.max_pending = self.max_workers * 2 if max_pending is None else max_pending
        if self.max_pending < self.max_workers:
            raise ValueError("max_pending must be at least max_workers")
        self._capacity = BoundedSemaphore(self.max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="frame-worker",
        )

    def submit(
        self,
        function: Callable[Parameters, Result],
        /,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> Future[Result]:
        self._capacity.acquire()
        try:
            future = self._executor.submit(function, *args, **kwargs)
        except BaseException:
            self._capacity.release()
            raise
        future.add_done_callback(lambda _: self._capacity.release())
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> AdaptiveFrameExecutor:
        return self

    def __exit__(self, _exc_type: object, exc_value: object, _traceback: object) -> None:
        self.shutdown(cancel_futures=exc_value is not None)