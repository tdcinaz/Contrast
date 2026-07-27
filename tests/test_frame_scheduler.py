from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread, current_thread
from unittest.mock import patch

from frame_scheduler import AdaptiveFrameExecutor, default_worker_count, performance_cpu_ids


class AdaptiveFrameExecutorTests(unittest.TestCase):
    def test_scales_workers_under_backlog(self) -> None:
        release = Event()
        all_started = Event()
        lock = Lock()
        thread_names: set[str] = set()

        def work(value: int) -> int:
            with lock:
                thread_names.add(current_thread().name)
                if len(thread_names) == 4:
                    all_started.set()
            release.wait(2.0)
            return value * 2

        with AdaptiveFrameExecutor(max_workers=4, max_pending=8) as executor:
            futures = [executor.submit(work, value) for value in range(4)]
            self.assertTrue(all_started.wait(2.0))
            release.set()
            self.assertEqual([future.result() for future in futures], [0, 2, 4, 6])

        self.assertEqual(len(thread_names), 4)

    def test_blocks_submission_at_pending_limit(self) -> None:
        release = Event()
        first_started = Event()
        second_submitted = Event()

        def blocked_work() -> None:
            first_started.set()
            release.wait(2.0)

        with AdaptiveFrameExecutor(max_workers=1, max_pending=1) as executor:
            first = executor.submit(blocked_work)
            self.assertTrue(first_started.wait(2.0))

            def submit_second() -> None:
                executor.submit(lambda: None)
                second_submitted.set()

            submitter = Thread(target=submit_second)
            submitter.start()
            self.assertFalse(second_submitted.wait(0.1))
            release.set()
            first.result()
            self.assertTrue(second_submitted.wait(2.0))
            submitter.join()

    def test_default_uses_performance_cpu_tier(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("frame_scheduler.performance_cpu_ids", return_value=tuple(range(10))):
            self.assertEqual(default_worker_count(), 10)

    def test_selects_highest_frequency_cpu_tier(self) -> None:
        with TemporaryDirectory() as directory, patch("frame_scheduler.available_cpu_ids", return_value=(0, 1, 2, 3)):
            root = Path(directory)
            for cpu_id, frequency in enumerate((2_800_000, 3_900_000, 2_800_000, 3_900_000)):
                frequency_path = root / f"cpu{cpu_id}" / "cpufreq" / "cpuinfo_max_freq"
                frequency_path.parent.mkdir(parents=True)
                frequency_path.write_text(str(frequency))
            self.assertEqual(performance_cpu_ids(root), (1, 3))

    def test_environment_override_and_invalid_limits(self) -> None:
        with patch.dict(os.environ, {"CONTRAST_FRAME_WORKERS": "3"}):
            self.assertEqual(default_worker_count(), 3)
        with self.assertRaises(ValueError):
            AdaptiveFrameExecutor(max_workers=0)
        with self.assertRaises(ValueError):
            AdaptiveFrameExecutor(max_workers=2, max_pending=1)


if __name__ == "__main__":
    unittest.main()