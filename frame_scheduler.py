from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import count
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Callable, ParamSpec, TypeVar


Parameters = ParamSpec("Parameters")
Result = TypeVar("Result")
WORKER_COUNT_ENV = "CONTRAST_FRAME_WORKERS"
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")


def available_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return tuple(range(max(1, os.cpu_count() or 1)))


def available_parallelism() -> int:
    return len(available_cpu_ids())


def performance_cpu_ids(sysfs_root: Path = CPU_SYSFS_ROOT) -> tuple[int, ...]:
    frequencies: dict[int, int] = {}
    for cpu_id in available_cpu_ids():
        frequency_path = sysfs_root / f"cpu{cpu_id}" / "cpufreq" / "cpuinfo_max_freq"
        try:
            frequencies[cpu_id] = int(frequency_path.read_text().strip())
        except (OSError, ValueError):
            return available_cpu_ids()
    maximum_frequency = max(frequencies.values())
    return tuple(cpu_id for cpu_id, frequency in frequencies.items() if frequency == maximum_frequency)


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
    return len(performance_cpu_ids())


class AdaptiveFrameExecutor:
    """A lazily growing, bounded pool for compute work from any frame stage."""

    def __init__(self, max_workers: int | None = None, max_pending: int | None = None) -> None:
        self.max_workers = default_worker_count() if max_workers is None else max_workers
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self.max_pending = self.max_workers * 2 if max_pending is None else max_pending
        if self.max_pending < self.max_workers:
            raise ValueError("max_pending must be at least max_workers")
        performance_cpus = performance_cpu_ids()
        self.cpu_affinity = performance_cpus if self.max_workers <= len(performance_cpus) else available_cpu_ids()
        self._capacity = BoundedSemaphore(self.max_pending)
        affinity_counter = count()
        affinity_lock = Lock()

        def initialize_worker() -> None:
            with affinity_lock:
                cpu_id = self.cpu_affinity[next(affinity_counter) % len(self.cpu_affinity)]
            try:
                os.sched_setaffinity(0, {cpu_id})
            except (AttributeError, OSError):
                pass

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="frame-worker",
            initializer=initialize_worker,
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