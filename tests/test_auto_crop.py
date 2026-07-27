from __future__ import annotations

import unittest

import cv2
import numpy as np

from main import _detect_aligned_field_crop, _detect_pillarbox_crop


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


if __name__ == "__main__":
    unittest.main()