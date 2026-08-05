from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtCore import QRect

import main
from main import (
    _adjust_auto_crop_square,
    _detect_aligned_field_crop,
    _detect_pillarbox_crop,
    detect_stationary_metal_mask,
    detect_fluoroscope_crop_from_frames,
    detect_pre_injection_trim_start,
    SourcePipelineState,
    VideoPanel,
    align_source_gain_states,
)


class AutoCropTests(unittest.TestCase):
    def test_stationary_metal_needle_is_detected_and_replaced_from_local_context(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(5):
            frame = np.full((180, 240), 160, dtype=np.uint8)
            cv2.rectangle(frame, (12, 28), (34, 152), 12, thickness=-1)
            cv2.circle(frame, (120 + index * 8, 90), 20, 25, thickness=-1)
            frames.append(frame)

        mask = detect_stationary_metal_mask(frames)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertGreater(np.count_nonzero(mask), 2_000)
        self.assertEqual(mask[90, 120], 0)
        filtered = VideoPanel._remove_stationary_metal(SimpleNamespace(metal_needle_mask=mask), frames[0])
        masked = mask > 0
        self.assertGreater(float(np.mean(filtered[masked])), float(np.mean(frames[0][masked])) + 100.0)

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

    def test_detects_dark_standby_field_with_bright_label(self) -> None:
        frames: list[np.ndarray] = []
        for level in (42, 45, 48, 51, 54, 57):
            frame = np.full((240, 400), level, dtype=np.uint8)
            cv2.circle(frame, (200, 120), 110, 0, thickness=-1)
            cv2.putText(frame, "Please Initiate SCAN", (112, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 170, 2)
            frames.append(frame)

        crop = _detect_aligned_field_crop(frames)

        self.assertIsNotNone(crop)
        assert crop is not None
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
        self.assertEqual(first.configuration, (True, False, False, False, 0, 0, 0.0, 0.0, 0.0, False))
        self.assertEqual(second.configuration, (True, False, False, False, 0, 0, 0.0, 0.0, 0.0, False))
        detect_crop.assert_called_once()

    def test_temporal_offsets_adjust_detected_trim_without_changing_cache_key(self) -> None:
        class TemporalSource:
            _full_frame_rect = VideoPanel._full_frame_rect
            _crop_rect_key = VideoPanel._crop_rect_key

            def __init__(self) -> None:
                self.info = SimpleNamespace(width=400, height=240, fps=10.0, frame_count=100)
                self.live_input = False
                self._auto_crop_rect_cache = None
                self._trim_start_cache = {}

            def _sample_cropped_gray_frames(self, crop_rect, progress_callback):  # noqa: ANN001
                return [np.zeros((240, 400), dtype=np.uint8)] * 3

        source = TemporalSource()
        with patch.object(main, "detect_pre_injection_trim_start", return_value=30):
            state = VideoPanel.calculate_source_pipeline(
                source,
                False,
                True,
                temporal_trim_offset_seconds=0.20,
                temporal_end_trim_seconds=1.25,
                comparison_sync_offset_seconds=0.30,
            )

        self.assertEqual(state.trim_start, 35)
        self.assertEqual(state.trim_end, 12)
        self.assertEqual(state.detected_trim_start, 30)
        self.assertEqual(state.configuration, (False, True, False, False, 0, 0, 0.20, 1.25, 0.30, False))

    def test_gain_alignment_raises_only_the_lower_contrast_response(self) -> None:
        common = dict(
            crop_rect=QRect(0, 0, 4, 4),
            trim_start=0,
            auto_crop_rect=None,
            trim_cache_key=None,
            detected_trim_start=None,
            background_reference=None,
            configuration=(False, False, True, False, 0, 0, 0.0, 0.0),
        )

        states = align_source_gain_states(
            [
                SourcePipelineState(**common, gain_alignment_baseline=80.0, gain_alignment_span=20.0),
                SourcePipelineState(**common, gain_alignment_baseline=120.0, gain_alignment_span=30.0),
            ]
        )

        self.assertEqual(states[0].gain_multiplier, 1.5)
        self.assertEqual(states[0].gain_offset, 0.0)
        self.assertEqual(states[1].gain_multiplier, 1.0)

    def test_gain_alignment_maps_baseline_and_contrast_span_together(self) -> None:
        common = dict(
            crop_rect=QRect(0, 0, 4, 4),
            trim_start=0,
            auto_crop_rect=None,
            trim_cache_key=None,
            detected_trim_start=None,
            background_reference=None,
            configuration=(False, False, True, False, 0, 0, 0.0, 0.0),
        )
        states = align_source_gain_states(
            [
                SourcePipelineState(**common, gain_alignment_baseline=120.0, gain_alignment_span=20.0),
                SourcePipelineState(**common, gain_alignment_baseline=150.0, gain_alignment_span=30.0),
            ]
        )

        self.assertEqual(states[0].gain_multiplier, 1.5)
        self.assertEqual(states[0].gain_offset, -30.0)
        self.assertEqual(120.0 * states[0].gain_multiplier + states[0].gain_offset, 150.0)

    def test_background_source_setting_change_invalidates_panel_cache(self) -> None:
        class SourcePanel:
            apply_source_pipeline_state = VideoPanel.apply_source_pipeline_state

            def __init__(self) -> None:
                self.source_pipeline_configuration = (False, False, False, False, 0, 0, 0.0, 0.0)
                self.background_subtraction_enabled = False
                self.background_level = 0
                self.background_reference = None
                self.source_gain_multiplier = 1.0
                self.source_gain_offset = 0.0
                self._auto_crop_rect_cache = None
                self._trim_start_cache = {}
                self.info = SimpleNamespace(frame_count=20)
                self.crop_rect = QRect(0, 0, 400, 240)
                self.trim_start_frame = 0
                self.trim_frame_count = 20
                self.clear_enhancement_cache = Mock()
                self.set_trim_window = Mock()
                self._activate_stage_roi_selection = Mock()

        panel = SourcePanel()
        state = main.SourcePipelineState(
            crop_rect=QRect(0, 0, 400, 240),
            trim_start=0,
            auto_crop_rect=None,
            trim_cache_key=None,
            detected_trim_start=None,
            background_reference=np.full((240, 400), 160, dtype=np.uint8),
            configuration=(False, False, False, True, 48, 0, 0.0, 0.0),
            trim_end=4,
        )

        self.assertTrue(panel.apply_source_pipeline_state(state))
        self.assertTrue(panel.background_subtraction_enabled)
        self.assertEqual(panel.background_level, 48)
        self.assertIsNotNone(panel.background_reference)
        panel.set_trim_window.assert_called_once_with(0, 16)

    def test_dsa_acquires_mask_before_injection_without_temporal_trimming(self) -> None:
        class Source:
            _full_frame_rect = VideoPanel._full_frame_rect
            calculate_source_pipeline = VideoPanel.calculate_source_pipeline

            def __init__(self) -> None:
                self.info = SimpleNamespace(width=400, height=240, fps=10.0, frame_count=100)
                self.live_input = False
                self._auto_crop_rect_cache = None
                self._trim_start_cache = {}
                self._sample_cropped_gray_frames = Mock(return_value=[np.zeros((240, 400), dtype=np.uint8)] * 100)
                self._acquire_dsa_mask = Mock(return_value=np.full((240, 400), 160, dtype=np.uint8))

        source = Source()
        with patch.object(main, "detect_pre_injection_trim_start", return_value=25):
            state = source.calculate_source_pipeline(False, False, background_subtraction_enabled=True)

        self.assertEqual(state.trim_start, 0)
        source._acquire_dsa_mask.assert_called_once()
        self.assertEqual(source._acquire_dsa_mask.call_args.args[1], 30)
        self.assertIsNotNone(state.background_reference)

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