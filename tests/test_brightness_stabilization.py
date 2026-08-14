from __future__ import annotations

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from contrast_pipeline import EnhancementParameters, EnhancementStages, PipelineStage
from main import (
    VideoPanel,
    align_frame_intensity,
    estimate_camera_translations,
    estimate_intensity_corrections,
    intensity_alignment_parameters,
    measure_roi_needle_baselines,
    stabilize_frame_intensity,
    stabilize_frame_translation,
)


class BrightnessStabilizationTests(unittest.TestCase):
    def _build_frames(self) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[:96, :128]
        base = 48.0 + 0.9 * xx + 0.55 * yy
        contrast_mask = (xx - 64) ** 2 + (yy - 48) ** 2 <= 16**2
        stable_mask = ~contrast_mask
        gains = np.asarray([0.88, 1.08, 0.94, 1.12, 0.91, 1.04, 0.86, 1.10, 0.97, 1.06] * 3)
        offsets = np.asarray([-8, 6, 3, -5, 9, -3, 5, -7, 1, 4] * 3)
        contrast_drop = np.linspace(0.0, 28.0, gains.size)
        frames: list[np.ndarray] = []

        for gain, offset, drop in zip(gains, offsets, contrast_drop):
            scene = base.copy()
            scene[contrast_mask] -= drop
            frames.append(np.clip(scene * gain + offset, 0, 255).astype(np.uint8))

        return frames, base, stable_mask, contrast_drop

    def test_stabilization_removes_affine_gain_and_brightness_jitter(self) -> None:
        frames, base, stable_mask, _ = self._build_frames()
        gains, offsets = estimate_intensity_corrections(frames, analysis_size=96)
        stabilized = [
            stabilize_frame_intensity(frame, gain, offset)
            for frame, gain, offset in zip(frames, gains, offsets)
        ]

        raw_maps = np.asarray([np.polyfit(frame[stable_mask], base[stable_mask], 1) for frame in frames])
        stabilized_maps = np.asarray(
            [np.polyfit(frame[stable_mask], base[stable_mask], 1) for frame in stabilized]
        )

        self.assertGreater(float(np.std(raw_maps[:, 0])), 0.05)
        self.assertLess(float(np.std(stabilized_maps[:, 0])), 0.005)
        self.assertGreater(float(np.std(raw_maps[:, 1])), 3.0)
        self.assertLess(float(np.std(stabilized_maps[:, 1])), 0.75)

    def test_stabilization_preserves_changing_dark_contrast(self) -> None:
        frames, base, _, contrast_drop = self._build_frames()
        yy, xx = np.mgrid[:96, :128]
        contrast_mask = (xx - 64) ** 2 + (yy - 48) ** 2 <= 16**2
        gains, offsets = estimate_intensity_corrections(frames, analysis_size=96)
        stabilized = [
            stabilize_frame_intensity(frame, gain, offset)
            for frame, gain, offset in zip(frames, gains, offsets)
        ]

        measured_trace = np.asarray(
            [float(np.median(frame[contrast_mask] - base[contrast_mask])) for frame in stabilized]
        )
        relative_trace_error = (measured_trace - measured_trace[0]) + contrast_drop
        self.assertLess(float(np.max(np.abs(relative_trace_error))), 0.75)

    def test_camera_motion_stabilization_corrects_abrupt_small_translation(self) -> None:
        rng = np.random.default_rng(7)
        texture = rng.normal(128.0, 24.0, (128, 160)).astype(np.float32)
        base = cv2.GaussianBlur(texture, (0, 0), sigmaX=1.2)
        cv2.line(base, (18, 20), (140, 104), 205.0, thickness=3)
        cv2.circle(base, (112, 44), 18, 62.0, thickness=3)
        base = np.clip(base, 0, 255).astype(np.uint8)
        observed_shifts = [(0.0, 0.0)] * 6 + [(5.0, -3.0)] * 8 + [(0.0, 0.0)] * 4
        frames: list[np.ndarray] = []
        for index, (shift_x, shift_y) in enumerate(observed_shifts):
            transform = np.asarray([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
            frame = cv2.warpAffine(base, transform, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
            cv2.circle(frame, (48 + index * 2, 78), 9, max(20, 90 - index * 3), thickness=-1)
            frames.append(frame)

        translations = estimate_camera_translations(frames, max_shift=8.0, analysis_size=160)
        stabilized = [
            stabilize_frame_translation(frame, translation)
            for frame, translation in zip(frames, translations)
        ]

        np.testing.assert_allclose(translations[8], (-5.0, 3.0), atol=0.35)
        np.testing.assert_allclose(translations[-1], (0.0, 0.0), atol=0.35)
        reference_edges = cv2.Laplacian(stabilized[0], cv2.CV_32F)[12:-12, 12:-12]
        shifted_edges = cv2.Laplacian(stabilized[8], cv2.CV_32F)[12:-12, 12:-12]
        self.assertLess(float(np.mean(np.abs(reference_edges - shifted_edges))), 13.0)

    def test_camera_motion_stage_runs_in_video_scheduler(self) -> None:
        rng = np.random.default_rng(11)
        base = cv2.GaussianBlur(
            rng.integers(40, 220, size=(96, 128), dtype=np.uint8),
            (0, 0),
            sigmaX=1.0,
        )
        frames = [base.copy() for _ in range(4)]
        transform = np.asarray([[1.0, 0.0, 4.0], [0.0, 1.0, -2.0]], dtype=np.float32)
        frames.extend(
            cv2.warpAffine(base, transform, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
            for _ in range(6)
        )
        panel = VideoPanel.__new__(VideoPanel)
        panel.path = "synthetic.avi"
        panel.info = SimpleNamespace(fps=15.0)
        panel.trim_frame_count = len(frames)
        panel.target_median = 128.0
        panel.camera_view_mask = np.ones(base.shape, dtype=bool)
        panel.source_gray_frames = frames
        panel.stage_frame_cache = {}
        panel.encoded_frame_cache = {}
        panel.roi_region_mask_cache = {}
        panel.roi_selection_cache = {}
        panel.temporal_change_map_cache = {}
        panel.active_sequence_key = None
        panel.inactive_sequence_key = None
        panel.stage_duration_per_frame = {}
        panel.enhanced_frames = None
        panel.roi_region_masks = None
        panel.stage_roi_selection = None
        panel.display = SimpleNamespace(set_roi=lambda *_args: None)
        parameters = EnhancementParameters()
        stages = EnhancementStages(
            instances=(PipelineStage("camera_motion_stabilization", True, parameters),)
        )

        self.assertTrue(panel.prepare_enhanced_frames(stages=stages, parameters=parameters))

        stabilized = next(iter(panel.stage_frame_cache.values()))
        self.assertEqual(len(stabilized), len(frames))
        difference = np.abs(stabilized[0][8:-8, 8:-8].astype(np.int16) - stabilized[-1][8:-8, 8:-8])
        self.assertLess(float(np.mean(difference)), 2.0)

    def test_narrow_roi_and_needle_range_is_expanded_then_offset(self) -> None:
        gain, offset = intensity_alignment_parameters(100.0, 50.0, 170.0, 70.0)
        frame = np.asarray([[50, 75, 100]], dtype=np.uint8)

        aligned = align_frame_intensity(frame, gain, offset)

        self.assertAlmostEqual(gain, 2.0)
        self.assertAlmostEqual(offset, -30.0)
        np.testing.assert_array_equal(aligned, np.asarray([[70, 120, 170]], dtype=np.uint8))

    def test_roi_and_needle_anchors_must_be_distinct(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            intensity_alignment_parameters(100.0, 100.0, 120.0, 40.0)

    def test_measures_pre_injection_roi_and_needle_baselines(self) -> None:
        frames = [np.full((80, 120), 150, dtype=np.uint8) for _ in range(8)]
        roi_mask = np.zeros((80, 120), dtype=np.uint8)
        cv2.circle(roi_mask, (80, 40), 12, 255, thickness=-1)
        for frame in frames:
            frame[10:70, 8:13] = 40

        roi_level, needle_level, needle_mask = measure_roi_needle_baselines(frames, 10.0, roi_mask)

        self.assertAlmostEqual(roi_level, 150.0)
        self.assertAlmostEqual(needle_level, 40.0)
        self.assertGreater(int(np.count_nonzero(needle_mask)), 200)


if __name__ == "__main__":
    unittest.main()
