from __future__ import annotations

import unittest
from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QFrame

import main
from main import (
    ContrastWindow,
    EnhancementParameters,
    EnhancementRequest,
    EnhancementStages,
    MODE_LIVE,
    VideoPanel,
    analyze_gray_frames,
    compute_temporal_change_map,
    detect_aneurysm_roi,
    overlay_segmentation_mask,
    segment_dark_contrast,
    segment_temporal_change_map,
    segment_temporal_change_contrast,
)


class SegmentationTests(unittest.TestCase):
    def test_current_source_pipeline_skips_source_preparation(self) -> None:
        request = EnhancementRequest(
            generation=1,
            mode="classical",
            model_label="Classical",
            stages=EnhancementStages(),
            parameters=EnhancementParameters(),
            noise_sigma=10,
            batch_size=4,
            precision="fp16",
            auto_crop=True,
            temporal_alignment=True,
            source_pipeline_current=True,
        )
        window = SimpleNamespace(panels=[])

        self.assertTrue(ContrastWindow._run_enhancement_request(window, request, Event()))

    def test_rebuild_marks_applied_source_pipeline_current(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window._add_pipeline_stage("auto_crop")
        window._add_pipeline_stage("temporal_alignment")
        for drawer in window.pipeline_stage_drawers:
            drawer.enable_button.blockSignals(True)
            drawer.enable_button.setChecked(True)
            drawer.enable_button.blockSignals(False)
        for panel in window.panels:
            panel.source_pipeline_configuration = (True, True)

        with patch.object(window, "_start_enhancement_request") as start_request:
            window.rebuild_enhancement_pipeline()

        request = start_request.call_args.args[0]
        self.assertTrue(request.source_pipeline_current)

    def test_disabled_pipeline_stage_changes_do_not_rebuild(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        with patch.object(window, "rebuild_enhancement_pipeline") as rebuild:
            first_drawer = window._add_pipeline_stage("local_contrast")
            window._duplicate_pipeline_stage(first_drawer)
            window._delete_pipeline_stage(window.live_pipeline_stage_drawers[-1])
            rebuild.assert_not_called()

            first_drawer.enable_button.setChecked(True)
            rebuild.assert_called_once()

    def test_config_round_trip_restores_duplicate_stage_settings(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        first_drawer = window._add_pipeline_stage("local_contrast")
        first_drawer.findChild(QDoubleSpinBox, "claheClipLimit").setValue(2.5)
        window._duplicate_pipeline_stage(first_drawer)
        second_drawer = window.live_pipeline_stage_drawers[-1]
        second_drawer.findChild(QDoubleSpinBox, "claheClipLimit").setValue(4.0)
        first_drawer.enable_button.blockSignals(True)
        first_drawer.enable_button.setChecked(True)
        first_drawer.enable_button.blockSignals(False)
        window.compare_view_check.setChecked(False)
        window.speed_slider.setValue(175)
        window.threshold_spin.setValue(0.35)

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.json"
            config_path.write_text(main.json.dumps(window._config_data()))
            with patch.object(main, "CONFIG_DIRECTORY", Path(directory)), patch.object(
                main,
                "RECENT_CONFIG_FILE",
                Path(directory) / "recent.json",
            ), patch.object(window, "on_pipeline_stages_changed"):
                self.assertTrue(window._load_config_file(config_path, show_error=False))

        self.assertEqual([drawer.stage_key for drawer in window.source_pipeline_stage_drawers], ["auto_crop", "temporal_alignment"])
        self.assertEqual([drawer.stage_key for drawer in window.live_pipeline_stage_drawers], ["brightness_stabilization", "local_contrast", "local_contrast"])
        self.assertTrue(window.live_pipeline_stage_drawers[1].enable_button.isChecked())
        self.assertFalse(window.live_pipeline_stage_drawers[2].enable_button.isChecked())
        self.assertEqual(window.live_pipeline_stage_drawers[1].findChild(QDoubleSpinBox, "claheClipLimit").value(), 2.5)
        self.assertEqual(window.live_pipeline_stage_drawers[2].findChild(QDoubleSpinBox, "claheClipLimit").value(), 4.0)
        self.assertFalse(window.compare_view_check.isChecked())
        self.assertEqual(window.speed_slider.value(), 175)
        self.assertEqual(window.threshold_spin.value(), 0.35)

    def test_source_and_live_pipeline_defaults_are_separate(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        self.assertEqual([drawer.stage_key for drawer in window.source_pipeline_stage_drawers], ["auto_crop", "temporal_alignment"])
        self.assertEqual([drawer.stage_key for drawer in window.live_pipeline_stage_drawers], ["brightness_stabilization"])

    def test_video_selection_queues_pipeline_without_sync_source_work(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        class FakeVideoPanel(QFrame):
            def __init__(self, label, color, path):  # noqa: ANN001
                super().__init__()
                self.label = label
                self.roiChanged = SimpleNamespace(connect=lambda callback: None)
                self.info = SimpleNamespace(fps=30.0)
                self.playback_frame_count = 1
                self.trim_frame_count = 1

            def set_comparison(self, enabled, frame_index):  # noqa: ANN001
                pass

            def seek(self, frame_index):  # noqa: ANN001
                pass

        with patch.object(main, "VideoPanel", FakeVideoPanel), patch.object(
            window, "_apply_source_pipeline_stages"
        ) as apply_source, patch.object(window, "rebuild_enhancement_pipeline") as rebuild:
            window._set_video_panels([Path("selected.mov")])

        apply_source.assert_not_called()
        rebuild.assert_called_once()
        self.assertTrue(all(drawer.enable_button.isChecked() for drawer in window.pipeline_stage_drawers))

        window._reorder_pipeline_stage_by_key("auto_crop", "brightness_stabilization")
        self.assertEqual([drawer.stage_key for drawer in window.source_pipeline_stage_drawers], ["auto_crop", "temporal_alignment"])
        self.assertEqual([drawer.stage_key for drawer in window.live_pipeline_stage_drawers], ["brightness_stabilization"])

    def test_live_mode_disables_temporal_stages_without_precomputing_video(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        class FakeLiveVideoPanel(QFrame):
            def __init__(self, label, color, path, live_input=False):  # noqa: ANN001
                super().__init__()
                self.label = label
                self.live_input = live_input
                self.roiChanged = SimpleNamespace(connect=lambda callback: None)
                self.info = SimpleNamespace(fps=30.0)
                self.playback_frame_count = 1
                self.trim_frame_count = 1

            def set_comparison(self, enabled, frame_index):  # noqa: ANN001
                pass

            def seek(self, frame_index):  # noqa: ANN001
                pass

        with patch.object(main, "VideoPanel", FakeLiveVideoPanel), patch.object(
            window, "_apply_source_pipeline_stages"
        ) as apply_source, patch.object(window, "_render_live_frame") as render_live, patch.object(
            window, "rebuild_enhancement_pipeline"
        ) as rebuild:
            window._set_video_panels([Path("camera-loop.mov")], live_input=True)

        self.assertEqual(window.active_mode, MODE_LIVE)
        self.assertTrue(window.panels[0].live_input)
        self.assertTrue(apply_source.called)
        self.assertTrue(render_live.called)
        rebuild.assert_not_called()
        for stage_key in ("temporal_alignment", "brightness_stabilization"):
            drawer = window._stage_drawers(stage_key)[0]
            self.assertFalse(drawer.enable_button.isEnabled())
            self.assertFalse(drawer.enable_button.isChecked())

    def test_aneurysm_detection_rejects_video_without_temporal_darkening(self) -> None:
        frames = [np.full((120, 160), 175, dtype=np.uint8) for _ in range(12)]
        for frame in frames:
            cv2.circle(frame, (80, 60), 22, 95, thickness=-1)

        self.assertIsNone(detect_aneurysm_roi(frames, fps=10.0))

    def test_detects_circular_contrast_filling_region_over_vessel_and_static_shapes(self) -> None:
        frames: list[np.ndarray] = []
        for step in range(16):
            frame = np.full((180, 240), 180 + step // 4, dtype=np.uint8)
            cv2.circle(frame, (65, 55), 17, 95 + step // 4, thickness=-1)
            cv2.rectangle(frame, (125, 82), (225, 94), 180 - step * 5, thickness=-1)
            cv2.circle(frame, (82, 125), 24, 180 - step * 6, thickness=-1)
            frames.append(frame)

        roi = detect_aneurysm_roi(frames, fps=10.0)

        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertTrue(roi.contains(82, 125))
        self.assertFalse(roi.contains(65, 55))
        self.assertFalse(roi.contains(180, 88))
        self.assertLess(abs(roi.width() - roi.height()), 12)

    def test_analysis_uses_roi_mask_instead_of_full_bounding_box(self) -> None:
        detection_frames: list[np.ndarray] = []
        for index in range(8):
            frame = np.full((96, 96), 180, dtype=np.uint8)
            level = 178 if index < 3 else 60
            cv2.circle(frame, (48, 48), 14, level, thickness=-1)
            detection_frames.append(frame)

        roi = detect_aneurysm_roi(detection_frames, fps=10.0)

        self.assertIsNotNone(roi)
        assert roi is not None
        analysis_frame = np.full((96, 96), 180, dtype=np.uint8)
        cv2.circle(analysis_frame, (48, 48), 14, 60, thickness=-1)
        result = analyze_gray_frames(
            "Pre-deployment",
            Path("synthetic.mov"),
            10.0,
            roi.rect,
            roi.mask,
            [analysis_frame],
            0.5,
            False,
        )

        full_box_mean = float(np.mean(analysis_frame[roi.rect.y() : roi.rect.y() + roi.rect.height(), roi.rect.x() : roi.rect.x() + roi.rect.width()]))
        self.assertGreaterEqual(result.mean_intensity[0], 60.0)
        self.assertLess(result.mean_intensity[0], full_box_mean - 5.0)

    def test_detected_roi_mask_is_softened_and_expanded_when_enabled(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(8):
            frame = np.full((96, 96), 180, dtype=np.uint8)
            level = 178 if index < 3 else 60
            cv2.circle(frame, (48, 48), 14, level, thickness=-1)
            frames.append(frame)

        raw_roi = detect_aneurysm_roi(frames, fps=10.0)
        roi = detect_aneurysm_roi(frames, fps=10.0, soften_mask=True, soften_radius_ratio=0.12, soften_threshold=0.10)

        self.assertIsNotNone(raw_roi)
        self.assertIsNotNone(roi)
        assert raw_roi is not None
        assert roi is not None
        self.assertGreater(int(np.count_nonzero(roi.mask)), int(np.count_nonzero(raw_roi.mask)))

    def test_analysis_requirement_fails_without_upstream_roi_stage(self) -> None:
        window = SimpleNamespace(
            roi_stage_check=SimpleNamespace(isChecked=lambda: False),
            panels=[],
            _missing_stage_roi_labels=lambda: [],
        )

        failure = ContrastWindow._analysis_requirement_failure(window)

        self.assertEqual(failure, "ROI residence analysis failed: enable upstream aneurysm ROI extraction.")

    def test_analysis_requirement_fails_when_roi_stage_has_no_mask(self) -> None:
        panel = SimpleNamespace(label="Pre-deployment", has_stage_roi_mask=lambda: False)
        window = SimpleNamespace(
            roi_stage_check=SimpleNamespace(isChecked=lambda: True),
            panels=[panel],
            _missing_stage_roi_labels=lambda: ["Pre-deployment"],
        )

        failure = ContrastWindow._analysis_requirement_failure(window)

        self.assertEqual(
            failure,
            "ROI residence analysis failed: ROI extraction did not produce masks for Pre-deployment.",
        )

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

    def test_temporal_change_segmentation_keeps_only_regions_with_large_full_video_change(self) -> None:
        frames: list[np.ndarray] = []
        for step in range(12):
            frame = np.full((120, 160), 180, dtype=np.uint8)
            cv2.circle(frame, (48, 60), 16, 180 - step * 8, thickness=-1)
            cv2.circle(frame, (112, 60), 16, 120, thickness=-1)
            frames.append(frame)

        change_map = segment_temporal_change_contrast(
            frames,
            change_threshold=20.0,
            level_tolerance=0,
            minimum_area=80,
            smoothing_window=21,
        )

        self.assertGreater(change_map[60, 48], 0)
        self.assertEqual(change_map[60, 112], 0)
        self.assertEqual(change_map.dtype, np.uint8)

    def test_temporal_change_map_cache_reuses_raw_measurement(self) -> None:
        panel = VideoPanel.__new__(VideoPanel)
        panel.temporal_change_map_cache = {}

        frames: list[np.ndarray] = []
        for step in range(8):
            frame = np.full((96, 96), 160, dtype=np.uint8)
            cv2.circle(frame, (32, 48), 12, 160 - step * 10, thickness=-1)
            cv2.circle(frame, (64, 48), 12, 110, thickness=-1)
            frames.append(frame)

        input_prefix: tuple[tuple[str, tuple[object, ...]], ...] = (("gain_stabilization", (128.0, 0.7, 1.45)),)

        cached = panel.temporal_change_map_cache.get(input_prefix)
        if cached is None:
            cached = compute_temporal_change_map(frames)
            panel.temporal_change_map_cache[input_prefix] = cached

        mask_a = segment_temporal_change_map(
            panel.temporal_change_map_cache[input_prefix],
            change_threshold=12.0,
            level_tolerance=0,
            minimum_area=40,
            smoothing_window=11,
        )
        cache_ref = panel.temporal_change_map_cache[input_prefix]

        # Simulate a second run with only segmentation parameters changed.
        cached_again = panel.temporal_change_map_cache.get(input_prefix)
        if cached_again is None:
            cached_again = compute_temporal_change_map(frames)
            panel.temporal_change_map_cache[input_prefix] = cached_again

        mask_b = segment_temporal_change_map(
            panel.temporal_change_map_cache[input_prefix],
            change_threshold=24.0,
            level_tolerance=0,
            minimum_area=40,
            smoothing_window=11,
        )

        self.assertIs(cache_ref, panel.temporal_change_map_cache[input_prefix])
        self.assertGreater(np.count_nonzero(mask_a), np.count_nonzero(mask_b))

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
            _source_pipeline_events=SimpleQueue(),
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