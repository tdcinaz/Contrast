from __future__ import annotations

import unittest
from concurrent.futures import Future
from queue import SimpleQueue
from types import SimpleNamespace

import cv2
import numpy as np

from main import ContrastWindow, overlay_segmentation_mask, segment_dark_contrast


class SegmentationTests(unittest.TestCase):
    def test_preserves_component_brightness_and_removes_small_components(self) -> None:
        frame = np.full((160, 160), 180, dtype=np.uint8)
        cv2.circle(frame, (55, 80), 18, 45, thickness=-1)
        cv2.circle(frame, (105, 80), 18, 95, thickness=-1)
        cv2.circle(frame, (20, 20), 2, 60, thickness=-1)

        mask = segment_dark_contrast(
            frame,
            block_size=51,
            sensitivity=7.0,
            level_tolerance=0,
            minimum_area=80,
        )

        self.assertEqual(mask[80, 55], 45)
        self.assertEqual(mask[80, 105], 95)
        self.assertEqual(mask[20, 20], 0)
        self.assertEqual(mask.dtype, np.uint8)

    def test_groups_component_levels_within_brightness_tolerance(self) -> None:
        frame = np.full((180, 220), 180, dtype=np.uint8)
        cv2.circle(frame, (35, 90), 15, 45, thickness=-1)
        cv2.circle(frame, (85, 90), 15, 52, thickness=-1)
        cv2.circle(frame, (135, 90), 15, 59, thickness=-1)
        cv2.circle(frame, (185, 90), 15, 95, thickness=-1)

        component_map = segment_dark_contrast(
            frame,
            block_size=41,
            sensitivity=7.0,
            level_tolerance=10,
            minimum_area=80,
        )

        self.assertEqual(component_map[90, 35], component_map[90, 85])
        self.assertNotEqual(component_map[90, 85], component_map[90, 135])
        self.assertNotEqual(component_map[90, 135], component_map[90, 185])

    def test_overlay_changes_only_masked_pixels_without_mutating_input(self) -> None:
        frame = np.full((4, 4, 3), 100, dtype=np.uint8)
        original = frame.copy()
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 45
        mask[2, 2] = 95

        overlaid = overlay_segmentation_mask(frame, mask, opacity=0.5)

        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(overlaid[0, 0], original[0, 0])
        self.assertFalse(np.array_equal(overlaid[1, 1], overlaid[2, 2]))

    def test_poll_attaches_mask_before_pipeline_future_completes(self) -> None:
        encoded_mask = np.array([1, 2, 3], dtype=np.uint8)
        seeks: list[int] = []
        panel = SimpleNamespace(
            enhanced_frames=[np.array([4, 5, 6], dtype=np.uint8)],
            segmentation_masks=[],
            seek=seeks.append,
        )
        mask_events: SimpleQueue[tuple[int, int, int, np.ndarray]] = SimpleQueue()
        mask_events.put((7, 0, 0, encoded_mask))
        pending_future: Future[bool] = Future()
        playback_limits: list[int] = []
        window = SimpleNamespace(
            _segmentation_mask_events=mask_events,
            _enhancement_frame_events=SimpleQueue(),
            _enhancement_generation=7,
            _enhancement_active_request=None,
            _enhancement_future=pending_future,
            panels=[panel],
            current_frame_index=0,
            _set_playback_limit=playback_limits.append,
        )

        ContrastWindow._poll_enhancement(window)

        self.assertFalse(pending_future.done())
        self.assertIs(panel.segmentation_masks[0], encoded_mask)
        self.assertEqual(seeks, [0])
        self.assertEqual(playback_limits, [0])


if __name__ == "__main__":
    unittest.main()