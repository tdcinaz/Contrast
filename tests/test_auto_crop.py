from __future__ import annotations

import unittest

import cv2
import numpy as np

from main import _detect_aligned_field_crop, _detect_pillarbox_crop, detect_pre_injection_trim_start


class AutoCropTests(unittest.TestCase):
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