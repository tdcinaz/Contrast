from __future__ import annotations

import math
import unittest
from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue
from threading import Event
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QFrame, QSpinBox

import main
from main import (
    ContrastWindow,
    EnhancementParameters,
    EnhancementRequest,
    EnhancementStages,
    MODE_COMPARISON,
    MODE_LIVE,
    VideoPanel,
    analyze_gray_frames,
    average_frame_brightness,
    compute_temporal_change_map,
    compute_temporal_change_summary,
    detect_aneurysm_roi,
    build_analysis_result,
    normalize_analysis_results,
    overlay_roi_regions,
    fit_circle_to_convex_hull,
    roi_selection_from_mask,
    segment_temporal_change_map,
    segment_temporal_change_contrast,
    temporal_change_heatmap_peaks,
)


class SegmentationTests(unittest.TestCase):
    def test_manual_roi_selection_bypasses_auto_detection_and_has_a_distinct_cache_token(self) -> None:
        automatic = EnhancementParameters()
        manual = EnhancementParameters(roi_mode="manual", roi_manual_circle=(24, 30, 12))

        self.assertNotEqual(
            main.BUILTIN_STAGES.require("roi_extraction").cache_token(automatic),
            main.BUILTIN_STAGES.require("roi_extraction").cache_token(manual),
        )
        with patch.object(main, "detect_aneurysm_roi") as detect:
            selection = main.extract_aneurysm_roi([np.zeros((64, 64), dtype=np.uint8)], 10.0, manual)

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.rect, QRect(12, 18, 25, 25))
        self.assertTrue(selection.mask[12, 12])
        self.assertFalse(selection.mask[0, 0])
        detect.assert_not_called()

    def test_comparison_roi_modes_produce_independent_panel_parameters(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window._manual_roi_circles = [(14, 18, 10), None]
        window.roi_mode_combos[0].setCurrentIndex(window.roi_mode_combos[0].findData("manual"))
        window.roi_mode_combos[1].setCurrentIndex(window.roi_mode_combos[1].findData("auto"))

        pre = window._roi_parameters_for_panel(0)
        post = window._roi_parameters_for_panel(1)

        self.assertEqual(pre.roi_mode, "manual")
        self.assertEqual(pre.roi_manual_circle, (14, 18, 10))
        self.assertEqual(post.roi_mode, "auto")
        self.assertIsNone(post.roi_manual_circle)

    def test_manual_roi_drag_uses_press_position_as_circle_center(self) -> None:
        QApplication.instance() or QApplication([])
        display = main.VideoDisplay("Video", QColor("#ffffff"))
        self.addCleanup(display.close)
        display.set_comparison_enabled(False)
        display.resize(200, 200)
        display.set_frame(np.zeros((100, 100, 3), dtype=np.uint8))
        display.show()
        QApplication.processEvents()

        circles: list[tuple[int, int, int]] = []
        display.roiDrawn.connect(circles.append)
        center = display._display_rect.center()
        end = center + main.QPoint(30, 0)
        expected_center = display._display_to_frame_point(center)
        expected_end = display._display_to_frame_point(end)
        expected_radius = round(math.hypot(expected_end.x() - expected_center.x(), expected_end.y() - expected_center.y()))

        QTest.mousePress(display, main.Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(display, end)
        QTest.mouseRelease(display, main.Qt.MouseButton.LeftButton, pos=end)

        self.assertEqual(circles, [(expected_center.x(), expected_center.y(), expected_radius)])
        mask = display.roi_mask()
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertTrue(mask[expected_radius, expected_radius])
        self.assertFalse(mask[0, 0])

    def test_play_restarts_file_playback_at_the_final_frame(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        panel = SimpleNamespace(enhance_display=False, seek=Mock())
        window.panels = [panel]
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.max_frame = 3
        window.current_frame_index = 3

        window.play()

        self.assertEqual(window.current_frame_index, 0)
        panel.seek.assert_called_once_with(0)
        self.assertTrue(window.is_playing)

    def test_loop_rewinds_without_pausing_at_the_final_frame(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        panel = SimpleNamespace(enhance_display=False, seek=Mock())
        window.panels = [panel]
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.max_frame = 3
        window.current_frame_index = 3
        window.is_playing = True
        window.loop_check.setChecked(True)

        window.advance_frame()

        self.assertEqual(window.current_frame_index, 0)
        panel.seek.assert_called_once_with(0)
        self.assertTrue(window.is_playing)

    def test_non_looping_playback_pauses_at_the_final_frame(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.panels = [SimpleNamespace(enhance_display=False, seek=Mock())]
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.max_frame = 3
        window.current_frame_index = 3
        window.is_playing = True

        window.advance_frame()

        self.assertFalse(window.is_playing)

    def test_config_serializes_loop_preference(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        self.assertFalse(window._config_data()["view"]["loop"])
        window.loop_check.setChecked(True)

        self.assertTrue(window._config_data()["view"]["loop"])

    def test_comparison_report_title_uses_shared_device_and_test_identifier(self) -> None:
        self.assertEqual(
            main.comparison_report_title([Path("CArm_Study42_pre_01.avi"), Path("CArm_Study42_post_02.mov")]),
            "CArm Study42 - Contrast Residence Comparison",
        )
        self.assertEqual(
            main.comparison_report_title([Path("pre.mp4"), Path("post.mp4")]),
            "Contrast ROI Residence Comparison",
        )

    def test_comparison_pdf_report_writes_enhanced_images_and_residence_curves(self) -> None:
        QApplication.instance() or QApplication([])
        frame = np.full((40, 60), 120, dtype=np.uint8)
        encoded = cv2.imencode(".png", frame)[1]
        mask_image = np.zeros((12, 16), dtype=np.uint8)
        cv2.circle(mask_image, (8, 6), 5, 1, thickness=-1)
        mask = mask_image.astype(bool)
        panels = [
            SimpleNamespace(
                label="Pre-deployment",
                color=QColor("#38bdf8"),
                path=Path("pre.mp4"),
                enhanced_frames=[encoded],
                current_frame=None,
                roi=lambda: QRect(10, 8, 16, 12),
                roi_mask=lambda: mask,
            ),
            SimpleNamespace(
                label="Post-deployment",
                color=QColor("#f97316"),
                path=Path("post.mp4"),
                enhanced_frames=[encoded],
                current_frame=None,
                roi=lambda: QRect(10, 8, 16, 12),
                roi_mask=lambda: mask,
            ),
        ]
        results = {
            panel.label: SimpleNamespace(
                time=np.asarray([0.0, 0.5, 1.0]),
                normalized_signal=np.asarray([0.0, 1.0, 0.2]),
                mean_intensity=np.asarray([120.0, 92.0, 115.0]),
            )
            for panel in panels
        }
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "comparison.pdf"
            self.assertTrue(main.render_comparison_report(report_path, panels, results, 0))
            self.assertTrue(report_path.read_bytes().startswith(b"%PDF"))
            self.assertGreater(report_path.stat().st_size, 1_000)

    def test_reordered_roi_analysis_drawer_reflows_when_toggled(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.resize(500, 500)
        window.show()
        app.processEvents()

        gain_drawer = window._add_pipeline_stage("gain_stabilization")
        analysis_drawer = window._add_pipeline_stage("roi_residence_analysis", after=gain_drawer)
        window._reorder_pipeline_stage_by_key("roi_residence_analysis", "gain_stabilization")
        app.processEvents()
        collapsed_height = analysis_drawer.height()

        analysis_drawer.expand_button.setChecked(True)
        app.processEvents()
        expanded_height = analysis_drawer.height()

        analysis_drawer.expand_button.setChecked(False)
        app.processEvents()

        self.assertEqual(window.live_pipeline_stage_drawers[1], analysis_drawer)
        self.assertGreater(expanded_height, collapsed_height)
        self.assertEqual(analysis_drawer.height(), collapsed_height)

    def test_comparison_sync_control_is_only_visible_in_comparison_mode(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        self.assertFalse(window.comparison_sync_offset_row.isVisible())
        window.active_mode = MODE_COMPARISON
        window._update_temporal_alignment_controls()

        self.assertTrue(window.comparison_sync_offset_row.isHidden() is False)
    def test_average_frame_brightness_uses_each_full_frame_mean(self) -> None:
        frames = [
            np.asarray([[0, 2], [4, 6]], dtype=np.uint8),
            np.asarray([[10, 12], [14, 16]], dtype=np.uint8),
        ]

        np.testing.assert_array_equal(average_frame_brightness(frames), np.asarray([3.0, 13.0]))

    def test_temporal_change_summary_accumulates_change_and_normalizes_rate(self) -> None:
        frames = [
            np.asarray([[0, 10], [20, 30]], dtype=np.uint8),
            np.asarray([[4, 8], [20, 36]], dtype=np.uint8),
            np.asarray([[10, 8], [15, 42]], dtype=np.uint8),
        ]

        total, rate = compute_temporal_change_summary(frames, fps=2.0)

        np.testing.assert_array_equal(total, np.asarray([[10.0, 2.0], [5.0, 12.0]]))
        np.testing.assert_array_equal(rate, np.asarray([[10.0, 2.0], [5.0, 12.0]]))

    def test_temporal_change_heatmap_uses_shared_comparison_peaks(self) -> None:
        results = {
            "Pre-deployment": (np.asarray([[3.0, 8.0]]), np.asarray([[1.5, 4.0]])),
            "Post-deployment": (np.asarray([[5.0, 12.0]]), np.asarray([[2.0, 6.0]])),
        }

        cumulative_peak, rate_peak = temporal_change_heatmap_peaks(results)

        self.assertEqual(cumulative_peak, 12.0)
        self.assertEqual(rate_peak, 6.0)

    def test_comparison_heatmap_views_share_rendering_peaks(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.active_mode = MODE_COMPARISON
        panels = [
            SimpleNamespace(label="Pre-deployment", set_temporal_change_heatmap=lambda _heatmap: None),
            SimpleNamespace(label="Post-deployment", set_temporal_change_heatmap=lambda _heatmap: None),
        ]
        window.panels = panels
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.temporal_change_results = {
            "Pre-deployment": (np.asarray([[3.0]]), np.asarray([[1.5]])),
            "Post-deployment": (np.asarray([[12.0]]), np.asarray([[6.0]])),
        }

        with patch.object(main, "render_temporal_change_heatmap", return_value=np.zeros((1, 1, 3), dtype=np.uint8)) as render:
            window.refresh_temporal_change_views()

        self.assertEqual(render.call_count, 2)
        for call in render.call_args_list:
            self.assertEqual(call.args[2:], (12.0, 6.0))

    def test_temporal_change_view_is_mutually_exclusive_with_source_view(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.temporal_change_results = {"video": (np.zeros((2, 2)), np.zeros((2, 2)))}
        window.temporal_change_view_check.setEnabled(True)

        window.temporal_change_view_check.setChecked(True)

        self.assertTrue(window.temporal_change_view_check.isChecked())
        self.assertFalse(window.compare_view_check.isChecked())

        window.compare_view_check.setChecked(True)

        self.assertTrue(window.compare_view_check.isChecked())
        self.assertFalse(window.temporal_change_view_check.isChecked())

    def test_frame_brightness_analysis_uses_one_plot_per_comparison_video(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.panels = [
            SimpleNamespace(label="Pre-deployment", color=QColor("#38bdf8")),
            SimpleNamespace(label="Post-deployment", color=QColor("#f97316")),
        ]
        window.frame_brightness_results = {
            "Pre-deployment": (np.asarray([0.0, 1.0]), np.asarray([80.0, 81.0]), np.asarray([90.0, 91.0])),
            "Post-deployment": (np.asarray([0.0, 1.0]), np.asarray([82.0, 83.0]), np.asarray([92.0, 93.0])),
        }

        window.refresh_frame_brightness_plot()

        self.assertEqual(set(window.frame_brightness_plots), {"Pre-deployment", "Post-deployment"})
        self.assertEqual(window.frame_brightness_layout.count(), 2)

    def test_current_source_pipeline_skips_source_preparation(self) -> None:
        request = EnhancementRequest(
            generation=1,
            mode="ffdnet-ngc",
            model_label="NGC FFDNet (Docker)",
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
            panel.source_pipeline_configuration = (True, True, False, False, 0, 0, 0.0, 0.0)

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
        window.temporal_trim_offset_spin.setValue(-0.25)
        window.comparison_sync_offset_spin.setValue(0.12)

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.json"
            config_path.write_text(main.json.dumps(window._config_data()))
            with patch.object(main, "CONFIG_DIRECTORY", Path(directory)), patch.object(
                main,
                "RECENT_CONFIG_FILE",
                Path(directory) / "recent.json",
            ), patch.object(window, "on_pipeline_stages_changed"):
                self.assertTrue(window._load_config_file(config_path, show_error=False))

        self.assertEqual(
            [drawer.stage_key for drawer in window.source_pipeline_stage_drawers],
            ["auto_crop", "temporal_alignment", "background_subtraction"],
        )
        self.assertTrue(all(not drawer.isHidden() for drawer in window.source_pipeline_stage_drawers))
        self.assertEqual([drawer.stage_key for drawer in window.live_pipeline_stage_drawers], ["brightness_stabilization", "local_contrast", "local_contrast"])
        self.assertTrue(window.live_pipeline_stage_drawers[1].enable_button.isChecked())
        self.assertFalse(window.live_pipeline_stage_drawers[2].enable_button.isChecked())
        self.assertEqual(window.live_pipeline_stage_drawers[1].findChild(QDoubleSpinBox, "claheClipLimit").value(), 2.5)
        self.assertEqual(window.live_pipeline_stage_drawers[2].findChild(QDoubleSpinBox, "claheClipLimit").value(), 4.0)
        self.assertFalse(window.compare_view_check.isChecked())
        self.assertEqual(window.speed_slider.value(), 175)
        self.assertEqual(window.threshold_spin.value(), 0.35)
        self.assertEqual(window.temporal_trim_offset_spin.value(), -0.25)
        self.assertEqual(window.comparison_sync_offset_spin.value(), 0.12)

    def test_source_and_live_pipeline_defaults_are_separate(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        self.assertEqual(
            [drawer.stage_key for drawer in window.source_pipeline_stage_drawers],
            ["auto_crop", "temporal_alignment", "background_subtraction"],
        )
        self.assertEqual([drawer.stage_key for drawer in window.live_pipeline_stage_drawers], ["brightness_stabilization"])

    def test_default_pipeline_settings_restore_stages_without_video_paths(self) -> None:
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            settings_path = Path(directory) / "default_pipeline.json"
            with patch.object(main, "DEFAULT_PIPELINE_SETTINGS_FILE", settings_path):
                window = ContrastWindow()
                self.addCleanup(window.close)
                auto_crop = window._add_pipeline_stage("auto_crop")
                auto_crop.findChild(QSpinBox, "autoCropSizeOffset").setValue(64)
                brightness = window._add_pipeline_stage("brightness_stabilization")
                brightness.enable_button.setChecked(False)

                window.save_default_pipeline_settings()

                saved = main.json.loads(settings_path.read_text())
                self.assertEqual(set(saved), {"version", "pipeline"})
                self.assertNotIn("videos", saved)

                restored_window = ContrastWindow()
                self.addCleanup(restored_window.close)

        self.assertEqual([drawer.stage_key for drawer in restored_window.source_pipeline_stage_drawers], ["auto_crop"])
        self.assertEqual([drawer.stage_key for drawer in restored_window.live_pipeline_stage_drawers], ["brightness_stabilization"])
        self.assertEqual(
            restored_window.source_pipeline_stage_drawers[0].findChild(QSpinBox, "autoCropSizeOffset").value(),
            64,
        )
        self.assertFalse(restored_window.live_pipeline_stage_drawers[0].enable_button.isChecked())

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
        self.assertTrue(
            all(
                drawer.enable_button.isChecked()
                for drawer in window.pipeline_stage_drawers
                if drawer.stage_key != "background_subtraction"
            )
        )
        self.assertFalse(window.background_subtraction_stage_check.isChecked())

        window._reorder_pipeline_stage_by_key("auto_crop", "brightness_stabilization")
        self.assertEqual(
            [drawer.stage_key for drawer in window.source_pipeline_stage_drawers],
            ["auto_crop", "temporal_alignment", "background_subtraction"],
        )
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

    def test_aneurysm_detection_ignores_late_video_distractors(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(160):
            frame = np.full((120, 160), 180, dtype=np.uint8)
            if 6 <= index < 60:
                cv2.circle(frame, (48, 60), 18, 60, thickness=-1)
            if index >= 110:
                cv2.circle(frame, (112, 60), 24, 20, thickness=-1)
            frames.append(frame)

        roi = detect_aneurysm_roi(frames, fps=10.0)

        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertTrue(roi.contains(48, 60))
        self.assertFalse(roi.contains(112, 60))

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

    def test_analysis_normalizes_multiple_videos_to_the_shared_peak(self) -> None:
        roi = main.QRect(0, 0, 1, 1)
        pre = build_analysis_result(
            "Pre-deployment",
            Path("pre.mov"),
            1.0,
            roi,
            np.asarray([100.0, 100.0, 60.0, 100.0]),
            np.asarray([], dtype=float),
            0.20,
            False,
        )
        post = build_analysis_result(
            "Post-deployment",
            Path("post.mov"),
            1.0,
            roi,
            np.asarray([100.0, 100.0, 20.0, 100.0]),
            np.asarray([], dtype=float),
            0.20,
            False,
        )

        results = normalize_analysis_results({pre.label: pre, post.label: post}, 0.20)

        np.testing.assert_allclose(results[pre.label].normalized_signal, [0.0, 0.0, 0.5, 0.0])
        np.testing.assert_allclose(results[post.label].normalized_signal, [0.0, 0.0, 1.0, 0.0])

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

    def test_overlay_changes_only_masked_pixels_without_mutating_input(self) -> None:
        frame = np.full((4, 4, 3), 100, dtype=np.uint8)
        original = frame.copy()
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = main.ROI_VESSEL_LABEL
        mask[2, 2] = main.ROI_ANEURYSM_LABEL

        overlaid = overlay_roi_regions(frame, mask, opacity=0.5)

        np.testing.assert_array_equal(frame, original)
        np.testing.assert_array_equal(overlaid[0, 0], original[0, 0])
        self.assertFalse(np.array_equal(overlaid[1, 1], overlaid[2, 2]))

        colored = overlay_roi_regions(frame, mask, aneurysm_color=(0, 128, 255), opacity=1.0)
        np.testing.assert_array_equal(colored[1, 1], np.array([0, 0, 255], dtype=np.uint8))
        np.testing.assert_array_equal(colored[2, 2], np.array([0, 128, 255], dtype=np.uint8))

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

    def test_temporal_change_segmentation_labels_aneurysm_and_vessels_separately(self) -> None:
        change = np.zeros((160, 200), dtype=np.float32)
        cv2.rectangle(change, (20, 75), (150, 85), 30, thickness=-1)
        cv2.circle(change, (155, 80), 24, 30, thickness=-1)
        aneurysm = np.zeros(change.shape, dtype=bool)
        cv2.circle(aneurysm, (155, 80), 20, 1, thickness=-1)

        mask = segment_temporal_change_map(
            change,
            change_threshold=12.0,
            level_tolerance=12,
            minimum_area=80,
            smoothing_window=11,
            aneurysm_mask=aneurysm,
        )

        self.assertEqual(mask[80, 50], main.ROI_VESSEL_LABEL)
        self.assertEqual(mask[80, 155], main.ROI_ANEURYSM_LABEL)
        self.assertEqual(mask[20, 20], 0)
        overlay = overlay_roi_regions(np.full((160, 200, 3), 100, dtype=np.uint8), mask)
        self.assertFalse(np.array_equal(overlay[80, 50], overlay[80, 155]))

    def test_temporal_change_segmentation_fills_the_aneurysm_convex_hull(self) -> None:
        change = np.zeros((120, 160), dtype=np.float32)
        cv2.rectangle(change, (20, 45), (140, 75), 30, thickness=-1)
        aneurysm = np.zeros(change.shape, dtype=bool)
        aneurysm[50:71, 100:106] = True
        aneurysm[65:71, 100:131] = True

        mask = segment_temporal_change_map(
            change,
            change_threshold=12.0,
            level_tolerance=12,
            minimum_area=80,
            smoothing_window=11,
            aneurysm_mask=aneurysm,
        )

        self.assertEqual(mask[60, 110], main.ROI_ANEURYSM_LABEL)
        self.assertEqual(mask[60, 40], main.ROI_VESSEL_LABEL)

    def test_roi_selection_uses_the_final_aneurysm_region_perimeter(self) -> None:
        region = np.zeros((120, 160), dtype=bool)
        region[50:71, 100:106] = True
        region[65:71, 100:131] = True
        points = np.column_stack(np.nonzero(region)[::-1]).astype(np.int32)
        cv2.fillConvexPoly(region, cv2.convexHull(points), 1)

        selection = roi_selection_from_mask(region)

        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual(selection.rect.getRect(), (100, 50, 31, 21))
        self.assertTrue(selection.mask[10, 10])
        self.assertFalse(selection.mask[0, 30])

    def test_circle_fit_refines_the_aneurysm_convex_hull(self) -> None:
        region = np.zeros((120, 160), dtype=bool)
        cv2.circle(region, (90, 60), 20, 1, thickness=-1)
        cv2.rectangle(region, (105, 55), (125, 65), 1, thickness=-1)

        circle = fit_circle_to_convex_hull(region)

        self.assertTrue(circle[60, 90])
        self.assertFalse(circle[60, 125])

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

    def test_frame_cache_excludes_roi_and_keeps_one_undo_branch(self) -> None:
        panel = VideoPanel.__new__(VideoPanel)
        gain = ("gain_stabilization", (128.0, 0.7, 1.45))
        roi = ("roi_extraction", (False, 0.12, 0.1))
        contrast_a = ("local_contrast", (1.0, 6))
        contrast_b = ("local_contrast", (2.0, 6))
        smoothing = ("final_smoothing", (0.55,))
        first = (gain, roi, contrast_a)
        second = (gain, roi, contrast_b)
        third = (gain, roi, smoothing)
        panel.stage_frame_cache = {
            (gain,): [np.zeros((1, 1), dtype=np.uint8)],
            (gain, contrast_a): [np.zeros((1, 1), dtype=np.uint8)],
            (gain, contrast_b): [np.zeros((1, 1), dtype=np.uint8)],
            (gain, smoothing): [np.zeros((1, 1), dtype=np.uint8)],
        }
        panel.encoded_frame_cache = {
            (gain, contrast_a): [],
            (gain, contrast_b): [],
            (gain, smoothing): [],
        }
        panel.roi_region_mask_cache = {}
        panel.roi_selection_cache = {(gain, roi): None}
        panel.temporal_change_map_cache = {}
        panel.active_sequence_key = first
        panel.inactive_sequence_key = second

        self.assertEqual(panel._frame_sequence_key(first), (gain, contrast_a))

        panel._begin_cache_branch(third)

        self.assertIsNone(panel.inactive_sequence_key)
        self.assertEqual(set(panel.encoded_frame_cache), {(gain, contrast_a)})
        self.assertEqual(set(panel.stage_frame_cache), {(gain,), (gain, contrast_a)})

        panel.encoded_frame_cache[(gain, smoothing)] = []
        panel.stage_frame_cache[(gain, smoothing)] = [np.zeros((1, 1), dtype=np.uint8)]
        panel._activate_cache_branch(third)

        self.assertEqual(panel.active_sequence_key, third)
        self.assertEqual(panel.inactive_sequence_key, first)
        self.assertEqual(set(panel.encoded_frame_cache), {(gain, contrast_a), (gain, smoothing)})

    def test_prepare_reuses_upstream_frames_and_roi_does_not_rebuild_downstream(self) -> None:
        panel = VideoPanel.__new__(VideoPanel)
        panel.path = Path("synthetic.mov")
        panel.info = SimpleNamespace(fps=10.0)
        panel.trim_frame_count = 3
        panel.target_median = 128.0
        panel.source_gray_frames = [np.full((16, 16), level, dtype=np.uint8) for level in (80, 90, 100)]
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

        first_parameters = EnhancementParameters(
            gain_use_auto_target=False,
            gain_target_median=120,
            clahe_clip_limit=1.0,
        )
        second_parameters = EnhancementParameters(
            gain_use_auto_target=False,
            gain_target_median=120,
            clahe_clip_limit=2.0,
        )
        first_stages = EnhancementStages(
            instances=(
                main.PipelineStage("gain_stabilization", True, first_parameters),
                main.PipelineStage("local_contrast", True, first_parameters),
            )
        )
        second_stages = EnhancementStages(
            instances=(
                main.PipelineStage("gain_stabilization", True, second_parameters),
                main.PipelineStage("local_contrast", True, second_parameters),
            )
        )

        with patch.object(panel, "_apply_frame_stage", wraps=panel._apply_frame_stage) as apply_stage:
            self.assertTrue(panel.prepare_enhanced_frames(stages=first_stages, parameters=first_parameters))
            self.assertTrue(panel.prepare_enhanced_frames(stages=second_stages, parameters=second_parameters))

            applied_stage_keys = [call.args[0] for call in apply_stage.call_args_list]
            self.assertEqual(applied_stage_keys.count("gain_stabilization"), 3)
            self.assertEqual(applied_stage_keys.count("local_contrast"), 6)
            calls_before_roi = apply_stage.call_count
            encoded_before_roi = panel.enhanced_frames

            roi_stages = EnhancementStages(
                instances=(
                    main.PipelineStage("gain_stabilization", True, second_parameters),
                    main.PipelineStage("roi_extraction", True, second_parameters),
                    main.PipelineStage("local_contrast", True, second_parameters),
                )
            )
            with patch.object(main, "detect_aneurysm_roi", return_value=None) as detect_roi:
                self.assertTrue(panel.prepare_enhanced_frames(stages=roi_stages, parameters=second_parameters))

            self.assertEqual(apply_stage.call_count, calls_before_roi)
            self.assertIs(panel.enhanced_frames, encoded_before_roi)
            detect_roi.assert_called_once()

    def test_poll_attaches_mask_before_pipeline_future_completes(self) -> None:
        encoded_mask = np.array([1, 2, 3], dtype=np.uint8)
        seeks: list[int] = []
        panel = SimpleNamespace(
            enhanced_frames=[np.array([4, 5, 6], dtype=np.uint8)],
            roi_region_masks=[],
            seek=seeks.append,
        )
        mask_events: SimpleQueue[tuple[int, int, int, np.ndarray]] = SimpleQueue()
        mask_events.put((7, 0, 0, encoded_mask))
        pending_future: Future[bool] = Future()
        playback_limits: list[int] = []
        window = SimpleNamespace(
            _roi_region_mask_events=mask_events,
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
        self.assertIs(panel.roi_region_masks[0], encoded_mask)
        self.assertEqual(seeks, [0])
        self.assertEqual(playback_limits, [0])


if __name__ == "__main__":
    unittest.main()