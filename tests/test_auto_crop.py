from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QRect

import main
from main import (
    _adjust_auto_crop_square,
    _detect_aligned_field_crop,
    _detect_pillarbox_crop,
    detect_fluoroscope_crop_from_frames,
    detect_pre_injection_trim_start,
    VideoPanel,
)


class AutoCropTests(unittest.TestCase):
    def test_adjusted_crop_stays_square_centered_and_in_frame(self) -> None:
        crop = QRect(120, 40, 160, 160)

        self.assertEqual(_adjust_auto_crop_square(crop, 400, 240, 32).getRect(), (104, 24, 192, 192))
        self.assertEqual(_adjust_auto_crop_square(crop, 400, 240, -32).getRect(), (136, 56, 128, 128))
        self.assertEqual(_adjust_auto_crop_square(crop, 400, 240, 512).getRect(), (88, 8, 224, 224))

    def test_detects_aligned_square_inside_circular_field(self) -> None:
        frames: list[np.ndarray] = []
        for level in (150, 155, 160, 165, 170, 175):
            frame = np.zeros((240, 400), dtype=np.uint8)
            cv2.circle(frame, (200, 120), 110, level, thickness=-1)
            frames.append(frame)

        crop = _detect_aligned_field_crop(frames)

        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.width(), crop.height())
        self.assertEqual(crop.width() % 32, 0)
        self.assertEqual(crop.getRect(), (120, 40, 160, 160))

    def test_rejects_rectangular_content_as_fluoroscope_field(self) -> None:
        frames = [np.pad(np.full((200, 240), 160, dtype=np.uint8), ((0, 0), (80, 80))) for _ in range(6)]

        self.assertIsNone(_detect_aligned_field_crop(frames))
        self.assertEqual(_detect_pillarbox_crop(frames, 400, 200).getRect(), (78, 0, 244, 200))

    def test_detects_one_fixed_crop_from_live_camera_samples(self) -> None:
        frames: list[np.ndarray] = []
        for level in (150, 155, 160, 165, 170, 175):
            frame = np.zeros((240, 400), dtype=np.uint8)
            cv2.circle(frame, (200, 120), 110, level, thickness=-1)
            frames.append(frame)

        crop = detect_fluoroscope_crop_from_frames(frames, 400, 240)

        self.assertEqual(crop.getRect(), (120, 40, 160, 160))

    def test_live_source_caches_one_crop_and_disables_temporal_alignment(self) -> None:
        class LiveSource:
            _full_frame_rect = VideoPanel._full_frame_rect

            def __init__(self) -> None:
                self.path = Path("camera-loop.mov")
                self.info = SimpleNamespace(width=400, height=240)
                self.live_input = True
                self._auto_crop_rect_cache = None
                self._trim_start_cache = {}

        source = LiveSource()
        expected_crop = QRect(120, 40, 160, 160)
        with patch.object(main, "detect_fluoroscope_crop", return_value=expected_crop) as detect_crop:
            first = VideoPanel.calculate_source_pipeline(source, True, True)
            second = VideoPanel.calculate_source_pipeline(source, True, True)

        self.assertEqual(first.crop_rect.getRect(), expected_crop.getRect())
        self.assertEqual(second.crop_rect.getRect(), expected_crop.getRect())
        self.assertEqual(first.configuration, (True, False, 0))
        self.assertEqual(second.configuration, (True, False, 0))
        detect_crop.assert_called_once()

    def test_detects_trim_start_half_second_before_contrast_onset(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(60):
            frame = np.full((240, 320), 150 + index // 2, dtype=np.uint8)
            if index >= 30:
                cv2.circle(frame, (160, 120), 52, 85 + index // 2, thickness=-1)
            else:
                cv2.circle(frame, (160, 120), 52, 148 + index // 2, thickness=-1)
            frames.append(frame)

        trim_start = detect_pre_injection_trim_start(frames, fps=10.0)

        self.assertGreaterEqual(trim_start, 24)
        self.assertLessEqual(trim_start, 26)


if __name__ == "__main__":
    unittest.main()