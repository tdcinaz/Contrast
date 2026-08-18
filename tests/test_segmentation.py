from __future__ import annotations

import math
import unittest
from concurrent.futures import Future
from pathlib import Path
from queue import SimpleQueue
from threading import Event, current_thread
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QFrame, QSpinBox

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
    baseline_to_apex_normalized_signal,
    normalize_analysis_results,
    overlay_roi_regions,
    fit_circle_to_convex_hull,
    masked_average_brightness,
    needle_average_brightness,
    roi_selection_from_mask,
    segment_pre_injection_needle,
    segment_temporal_change_map,
    segment_temporal_change_contrast,
    temporal_derivative,
    temporal_change_heatmap_peaks,
)


class SegmentationTests(unittest.TestCase):
    def test_background_roi_uses_a_150_pixel_box_and_mean_brightness(self) -> None:
        frame = np.full((200, 200), 100, dtype=np.uint8)
        frame[25:175, 25:175] = 40

        trace = main.background_roi_average_brightness([frame], 100, 100)

        self.assertEqual(main.BACKGROUND_ROI_SIZE, 150)
        np.testing.assert_allclose(trace, [40.0])

    def test_shift_left_click_emits_background_roi_center_without_replacing_aneurysm_roi(self) -> None:
        QApplication.instance() or QApplication([])
        display = main.VideoDisplay("Video", QColor("#ffffff"))
        self.addCleanup(display.close)
        display.set_comparison_enabled(False)
        display.resize(300, 300)
        display.set_frame(np.zeros((200, 200, 3), dtype=np.uint8))
        display.show()
        QApplication.processEvents()

        centers: list[tuple[int, int]] = []
        display.backgroundRoiDrawn.connect(centers.append)
        center = display._display_rect.center()
        expected_center = display._display_to_frame_point(center)

        QTest.mouseClick(
            display,
            main.Qt.MouseButton.LeftButton,
            main.Qt.KeyboardModifier.ShiftModifier,
            center,
        )

        self.assertEqual(centers, [(expected_center.x(), expected_center.y())])
        self.assertIsNone(display.roi())

    def test_parent_vessel_roi_uses_a_rotated_50_pixel_box_for_darkest_decile_and_average(self) -> None:
        frame = np.full((100, 100), 180, dtype=np.uint8)
        frame[25:31, 25:75] = 12

        trace = main.parent_vessel_dark_median([frame], 50, 50, 30.0)
        average_brightness = main.parent_vessel_average_brightness([frame], 50, 50, 30.0)
        corners = cv2.boxPoints(((50.0, 50.0), (50.0, 50.0), 30.0)).round().astype(np.int32)
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners, 1)

        self.assertEqual(main.PARENT_VESSEL_ROI_SIZE, 50)
        np.testing.assert_allclose(trace, [12.0])
        np.testing.assert_allclose(average_brightness, [np.mean(frame[mask.astype(bool)])])

    def test_right_click_emits_parent_vessel_roi_center_without_replacing_aneurysm_roi(self) -> None:
        QApplication.instance() or QApplication([])
        display = main.VideoDisplay("Video", QColor("#ffffff"))
        self.addCleanup(display.close)
        display.set_comparison_enabled(False)
        display.resize(300, 300)
        display.set_frame(np.zeros((200, 200, 3), dtype=np.uint8))
        display.show()
        QApplication.processEvents()

        centers: list[tuple[int, int]] = []
        display.parentVesselRoiDrawn.connect(centers.append)
        center = display._display_rect.center()
        expected_center = display._display_to_frame_point(center)

        QTest.mouseClick(display, main.Qt.MouseButton.RightButton, pos=center)

        self.assertEqual(centers, [(expected_center.x(), expected_center.y())])
        self.assertIsNone(display.roi())

    def test_background_roi_and_temporal_change_results_keep_distinct_shapes(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        frames = [np.full((60, 80), 170, dtype=np.uint8) for _ in range(10)]
        panel = SimpleNamespace(
            label="Video",
            color=QColor("#38bdf8"),
            path=Path("video.avi"),
            info=SimpleNamespace(fps=4.0),
            source_gray_frames=frames,
            camera_view_mask=None,
            enhance_display=False,
            roi=Mock(return_value=None),
            roi_mask=Mock(return_value=None),
            encoded_analysis_frames=Mock(return_value=None),
            _sequence_key=Mock(return_value=()),
            _frame_sequence_key=Mock(return_value=()),
            set_temporal_change_heatmap=Mock(),
        )
        window.panels = [panel]
        self.addCleanup(lambda: setattr(window, "panels", []))
        window._background_rois = [(40, 30)]

        with patch.object(window, "_update_stage_statuses"):
            self.assertTrue(
                window._start_pipeline_analysis(
                    roi_residence=False,
                    frame_brightness=False,
                    needle_segmentation=False,
                    parent_vessel_roi=False,
                    background_roi=True,
                    temporal_change=True,
                )
            )
            future = window._analysis_future
            self.assertIsNotNone(future)
            assert future is not None
            future.result(timeout=5.0)
            window._poll_analysis()

        background_time, background_brightness = window.background_roi_results["Video"]
        self.assertEqual(background_brightness.shape, (len(frames),))
        np.testing.assert_allclose(background_time, np.arange(len(frames), dtype=float) / 4.0)
        burden, peak = window.temporal_change_results["Video"]
        self.assertEqual(burden.shape, frames[0].shape)
        self.assertEqual(peak.shape, frames[0].shape)
        panel.set_temporal_change_heatmap.assert_called_once()
        heatmap = panel.set_temporal_change_heatmap.call_args.args[0]
        self.assertEqual(heatmap.shape, (*frames[0].shape, 3))

    def test_parent_vessel_plot_marks_each_curve_minimum(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.panels = [
            SimpleNamespace(label="Pre-deployment", color=QColor("#38bdf8")),
            SimpleNamespace(label="Post-deployment", color=QColor("#f97316")),
        ]
        self.addCleanup(lambda: setattr(window, "panels", []))
        time = np.asarray([0.0, 0.5, 1.0])
        window.parent_vessel_roi_results = {
            "Pre-deployment": (time, np.asarray([102.0, 80.0, np.nan])),
            "Post-deployment": (time, np.asarray([95.0, 74.0, 88.0])),
        }

        window.refresh_parent_vessel_roi_plot()

        minimum_lines = [
            item
            for item in window.parent_vessel_roi_plot.plotItem.items
            if isinstance(item, main.pg.InfiniteLine)
        ]
        self.assertCountEqual([float(line.value()) for line in minimum_lines], [80.0, 74.0])

    def test_parent_vessel_scaled_residence_uses_the_first_video_minimum_as_reference(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        pre_label = "Pre-deployment"
        post_label = "Post-deployment"
        window.panels = [
            SimpleNamespace(label=pre_label, color=QColor("#38bdf8")),
            SimpleNamespace(label=post_label, color=QColor("#f97316")),
        ]
        self.addCleanup(lambda: setattr(window, "panels", []))
        roi = QRect(0, 0, 1, 1)
        window.results = {
            pre_label: build_analysis_result(
                pre_label, Path("pre.mov"), 2.0, roi, np.asarray([100.0, 50.0]), np.asarray([]), 0.2, False
            ),
            post_label: build_analysis_result(
                post_label, Path("post.mov"), 2.0, roi, np.asarray([100.0, 50.0]), np.asarray([]), 0.2, False
            ),
        }
        window.parent_vessel_roi_results = {
            pre_label: (np.asarray([0.0, 0.5]), np.asarray([100.0, 80.0])),
            post_label: (np.asarray([0.0, 0.5]), np.asarray([100.0, 40.0])),
        }

        window.refresh_parent_vessel_scaled_residence_plot()

        curves = window.parent_vessel_scaled_residence_plot.listDataItems()
        self.assertEqual(len(curves), 2)
        _pre_time, pre_signal = curves[0].getData()
        _post_time, post_signal = curves[1].getData()
        np.testing.assert_allclose(pre_signal, window.results[pre_label].normalized_signal)
        np.testing.assert_allclose(post_signal, window.results[post_label].normalized_signal * 0.5)
        self.assertTrue(window.analysis_tabs.isTabVisible(window.parent_vessel_scaled_residence_tab_index))

    def test_needle_mask_rejects_darker_compact_edge_artifact(self) -> None:
        camera_view_mask = np.zeros((100, 120), dtype=np.uint8)
        cv2.ellipse(camera_view_mask, (60, 50), (50, 42), 0, 0, 360, 1, thickness=-1)
        camera_view_mask = cv2.erode(camera_view_mask, np.ones((5, 5), dtype=np.uint8)) > 0
        frames: list[np.ndarray] = []
        for frame_index in range(8):
            frame = np.full(camera_view_mask.shape, 170, dtype=np.uint8)
            cv2.rectangle(frame, (92, 35), (96, 72), 16 + frame_index, thickness=-1)
            cv2.rectangle(frame, (116, 32), (119, 68), 1, thickness=-1)
            frames.append(frame)

        mask = segment_pre_injection_needle(frames, fps=4.0, camera_view_mask=camera_view_mask)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(mask[50, 94], 255)
        self.assertEqual(mask[50, 118], 0)
        self.assertFalse(np.any(mask[~camera_view_mask]))

    def test_needle_mask_uses_only_pre_injection_frames_and_tracks_full_video_brightness(self) -> None:
        frames: list[np.ndarray] = []
        for frame_index in range(12):
            frame = np.full((80, 100), 180, dtype=np.uint8)
            cv2.rectangle(frame, (12, 18), (15, 62), 18 + frame_index, thickness=-1)
            cv2.rectangle(frame, (88, 6), (92, 10), 2, thickness=-1)
            if frame_index >= 6:
                cv2.circle(frame, (70, 40), 15, 4, thickness=-1)
            frames.append(frame)

        mask = segment_pre_injection_needle(frames, fps=4.0)

        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(cv2.connectedComponents(mask)[0], 2)
        self.assertEqual(mask[40, 13], 255)
        self.assertEqual(mask[40, 70], 0)
        self.assertEqual(mask[8, 90], 0)
        np.testing.assert_allclose(masked_average_brightness(frames, mask), np.arange(18.0, 30.0))

    def test_needle_brightness_ignores_transient_mask_edge_pixels(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        cv2.rectangle(mask, (8, 14), (51, 25), 255, thickness=-1)
        frames = [np.full(mask.shape, 160, dtype=np.uint8) for _ in range(5)]
        for frame in frames:
            frame[mask > 0] = 24
        boundary = mask - cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=3)
        frames[2][boundary > 0] = 80

        raw_trace = masked_average_brightness(frames, mask)
        core_trace = needle_average_brightness(frames, mask)

        self.assertGreater(raw_trace[2], raw_trace[1] + 20.0)
        np.testing.assert_allclose(core_trace, np.full(5, 24.0))

    def test_needle_overlay_marks_only_brightness_analysis_core_in_red(self) -> None:
        mask = np.zeros((20, 30), dtype=np.uint8)
        cv2.rectangle(mask, (6, 5), (23, 14), 255, thickness=-1)
        frame = np.full((20, 30, 3), 100, dtype=np.uint8)

        overlay = main.overlay_needle_mask(frame, mask, opacity=1.0)

        np.testing.assert_array_equal(overlay[10, 15], (0, 0, 255))
        np.testing.assert_array_equal(overlay[5, 15], (100, 100, 100))
        np.testing.assert_array_equal(overlay[2, 2], (100, 100, 100))

    def test_temporal_derivative_uses_time_spacing(self) -> None:
        derivative = temporal_derivative(
            np.asarray([0.0, 0.5, 2.0]),
            np.asarray([0.0, 1.0, 4.0]),
        )

        np.testing.assert_allclose(derivative, np.asarray([2.0, 2.0, 2.0]))

    def test_baseline_to_apex_normalization_scales_each_result_independently(self) -> None:
        roi = QRect(0, 0, 1, 1)
        pre = build_analysis_result("Pre-deployment", Path("pre.mov"), 1.0, roi, np.asarray([100.0, 80.0]), np.asarray([]), 0.2, False)
        post = build_analysis_result("Post-deployment", Path("post.mov"), 1.0, roi, np.asarray([100.0, 20.0]), np.asarray([]), 0.2, False)
        results = normalize_analysis_results({pre.label: pre, post.label: post}, 0.2)

        np.testing.assert_allclose(results[pre.label].normalized_signal, [0.0, 1.0])
        np.testing.assert_allclose(results[post.label].normalized_signal, [0.0, 1.0])
        np.testing.assert_allclose(baseline_to_apex_normalized_signal(results[pre.label]), [0.0, 1.0])
        np.testing.assert_allclose(baseline_to_apex_normalized_signal(results[post.label]), [0.0, 1.0])

    def test_residence_time_measures_from_apex_to_clearance(self) -> None:
        result = build_analysis_result(
            "Pre-deployment",
            Path("pre.mov"),
            1.0,
            QRect(0, 0, 1, 1),
            np.asarray([100.0, 90.0, 60.0, 80.0, 95.0]),
            np.asarray([], dtype=float),
            0.2,
            False,
        )

        self.assertEqual(result.arrival_time, 1.0)
        self.assertEqual(result.peak_time, 2.0)
        self.assertEqual(result.clear_time, 4.0)
        self.assertEqual(result.residence_time, 2.0)

    def test_baseline_to_apex_plot_shows_the_clearance_threshold(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.active_mode = MODE_COMPARISON
        pre_label = "Pre-deployment"
        post_label = "Post-deployment"
        window.panels = [
            SimpleNamespace(label=pre_label, color=QColor("#38bdf8")),
            SimpleNamespace(label=post_label, color=QColor("#f97316")),
        ]
        self.addCleanup(lambda: setattr(window, "panels", []))
        roi = QRect(0, 0, 1, 1)
        results = {
            pre_label: build_analysis_result(pre_label, Path("pre.mov"), 1.0, roi, np.asarray([100.0, 80.0]), np.asarray([]), 0.35, False),
            post_label: build_analysis_result(post_label, Path("post.mov"), 1.0, roi, np.asarray([100.0, 20.0]), np.asarray([]), 0.35, False),
        }
        window.results = normalize_analysis_results(results, 0.35)
        window.threshold_spin.setValue(0.35)

        window.refresh_plots_and_metrics()

        normalized_thresholds = [
            item for item in window.normalized_plot.plotItem.items if isinstance(item, main.pg.InfiniteLine)
        ]
        apex_thresholds = [
            item for item in window.baseline_to_apex_plot.plotItem.items if isinstance(item, main.pg.InfiniteLine)
        ]
        self.assertEqual(normalized_thresholds, [])
        self.assertEqual([float(line.value()) for line in apex_thresholds], [0.35])

    def test_analysis_drawer_plots_normalized_signal_derivative(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        label = "Pre-deployment"
        window.panels = [SimpleNamespace(label=label, color=QColor("#38bdf8"))]
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.results = {
            label: build_analysis_result(
                label,
                Path("pre.mov"),
                2.0,
                QRect(0, 0, 1, 1),
                np.asarray([100.0, 75.0, 0.0]),
                np.asarray([], dtype=float),
                0.2,
                False,
            )
        }

        window.refresh_plots_and_metrics()

        self.assertEqual(window.analysis_tabs.tabText(2), "Derivative")
        curve = window.derivative_plot.listDataItems()[0]
        _, derivative = curve.getData()
        np.testing.assert_allclose(derivative, temporal_derivative(window.results[label].time, window.results[label].normalized_signal))

    def test_needle_stage_publishes_mask_and_full_duration_plot(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        frames: list[np.ndarray] = []
        for frame_index in range(10):
            frame = np.full((60, 80), 170, dtype=np.uint8)
            cv2.rectangle(frame, (8, 12), (11, 48), 20 + frame_index, thickness=-1)
            if frame_index >= 6:
                cv2.circle(frame, (58, 30), 12, 5, thickness=-1)
            frames.append(frame)
        publish_mask = Mock()
        panel = SimpleNamespace(
            label="Video",
            color=QColor("#38bdf8"),
            path=Path("video.avi"),
            info=SimpleNamespace(fps=4.0),
            source_gray_frames=frames,
            camera_view_mask=None,
            enhance_display=False,
            roi=Mock(return_value=None),
            roi_mask=Mock(return_value=None),
            encoded_analysis_frames=Mock(return_value=None),
            _sequence_key=Mock(return_value=()),
            _frame_sequence_key=Mock(return_value=()),
            set_needle_segmentation_mask=publish_mask,
        )
        window.panels = [panel]
        self.addCleanup(lambda: setattr(window, "panels", []))
        worker_threads: list[str] = []

        def record_worker(*args, **kwargs):  # noqa: ANN002, ANN003
            worker_threads.append(current_thread().name)
            return segment_pre_injection_needle(*args, **kwargs)

        with patch.object(window, "_update_stage_statuses"), patch.object(
            main,
            "segment_pre_injection_needle",
            side_effect=record_worker,
        ):
            self.assertTrue(window.run_needle_segmentation())
            self.assertFalse(window.enhancement_progress.isHidden())
            self.assertEqual(window._enhancement_progress_totals, [float(len(frames))])
            future = window._analysis_future
            self.assertIsNotNone(future)
            assert future is not None
            future.result(timeout=5.0)
            publish_mask.assert_not_called()
            self.assertEqual(window._enhancement_stage_messages, ["Segmenting needle"])
            window._poll_analysis()

        self.assertTrue(all(name.startswith("analysis-panel") for name in worker_threads))
        self.assertTrue(window.enhancement_progress.isHidden())
        self.assertEqual(window.analysis_tabs.tabText(window.analysis_tabs.count() - 1), "Needle brightness")
        mask = publish_mask.call_args.args[0]
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(mask[30, 9], 255)
        self.assertEqual(mask[30, 58], 0)
        time, brightness = window.needle_brightness_plot.listDataItems()[0].getData()
        np.testing.assert_allclose(time, np.arange(10, dtype=float) / 4.0)
        np.testing.assert_allclose(brightness, np.arange(20.0, 30.0))
        self.assertAlmostEqual(window.needle_brightness_baselines["Video"], 22.5)
        reference_lines = [
            item
            for item in window.needle_brightness_plot.plotItem.items
            if isinstance(item, main.pg.InfiniteLine)
        ]
        self.assertEqual(len(reference_lines), 1)
        y_range = window.needle_brightness_plot.viewRange()[1]
        self.assertGreaterEqual(y_range[1] - y_range[0], 10.0)

    def test_comparison_needle_brightness_plot_shows_one_baseline_per_video(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.active_mode = MODE_COMPARISON
        window.panels = [
            SimpleNamespace(label="Pre-deployment", color=QColor("#38bdf8")),
            SimpleNamespace(label="Post-deployment", color=QColor("#f97316")),
        ]
        self.addCleanup(lambda: setattr(window, "panels", []))
        time = np.asarray([0.0, 0.25, 0.5])
        window.needle_brightness_results = {
            "Pre-deployment": (time, np.asarray([20.0, 21.0, 26.0])),
            "Post-deployment": (time, np.asarray([40.0, 39.0, 32.0])),
        }
        window.needle_brightness_baselines = {
            "Pre-deployment": 20.5,
            "Post-deployment": 39.5,
        }

        window.refresh_needle_brightness_plot()

        reference_lines = [
            item
            for item in window.needle_brightness_plot.plotItem.items
            if isinstance(item, main.pg.InfiniteLine)
        ]
        self.assertEqual(len(reference_lines), 2)
        self.assertCountEqual([float(line.value()) for line in reference_lines], [20.5, 39.5])

    def test_comparison_drawer_plots_baseline_to_apex_curves(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.active_mode = MODE_COMPARISON
        window.panels = [
            SimpleNamespace(label="Pre-deployment", color=QColor("#38bdf8")),
            SimpleNamespace(label="Post-deployment", color=QColor("#f97316")),
        ]
        self.addCleanup(lambda: setattr(window, "panels", []))
        roi = QRect(0, 0, 1, 1)
        window.results = normalize_analysis_results(
            {
                "Pre-deployment": build_analysis_result("Pre-deployment", Path("pre.mov"), 1.0, roi, np.asarray([100.0, 80.0]), np.asarray([]), 0.2, False),
                "Post-deployment": build_analysis_result("Post-deployment", Path("post.mov"), 1.0, roi, np.asarray([100.0, 20.0]), np.asarray([]), 0.2, False),
            },
            0.2,
        )

        window.refresh_plots_and_metrics()

        self.assertTrue(window.analysis_tabs.isTabVisible(window.baseline_to_apex_tab_index))
        curves = [curve.getData()[1] for curve in window.baseline_to_apex_plot.listDataItems()]
        self.assertEqual(len(curves), 2)
        for curve in curves:
            np.testing.assert_allclose(curve, [0.0, 1.0])
        roi_baselines = [
            item
            for item in window.raw_plot.plotItem.items
            if isinstance(item, main.pg.InfiniteLine)
        ]
        self.assertEqual(len(roi_baselines), 2)
        self.assertCountEqual([float(line.value()) for line in roi_baselines], [100.0, 100.0])

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

    def test_manual_roi_click_uses_fixed_radius_at_selected_center(self) -> None:
        QApplication.instance() or QApplication([])
        display = main.VideoDisplay("Video", QColor("#ffffff"))
        self.addCleanup(display.close)
        display.set_comparison_enabled(False)
        display.resize(300, 300)
        display.set_frame(np.zeros((200, 200, 3), dtype=np.uint8))
        display.show()
        QApplication.processEvents()

        circles: list[tuple[int, int, int]] = []
        display.roiDrawn.connect(circles.append)
        center = display._display_rect.center()
        expected_center = display._display_to_frame_point(center)

        QTest.mouseClick(display, main.Qt.MouseButton.LeftButton, pos=center)

        self.assertEqual(circles, [(expected_center.x(), expected_center.y(), display.MANUAL_ROI_RADIUS)])
        mask = display.roi_mask()
        self.assertIsNotNone(mask)
        assert mask is not None
        self.assertEqual(mask.shape, (2 * display.MANUAL_ROI_RADIUS + 1,) * 2)
        self.assertTrue(mask[display.MANUAL_ROI_RADIUS, display.MANUAL_ROI_RADIUS])
        self.assertFalse(mask[0, 0])

    def test_automatic_roi_publication_does_not_emit_manual_change_signal(self) -> None:
        QApplication.instance() or QApplication([])
        display = main.VideoDisplay("Video", QColor("#ffffff"))
        self.addCleanup(display.close)
        changed = Mock()
        display.roiChanged.connect(changed)
        mask = np.ones((8, 10), dtype=bool)
        selection = main.ROISelection(QRect(4, 6, 10, 8), mask)
        panel = SimpleNamespace(display=display, stage_roi_selection=None)

        VideoPanel._activate_stage_roi_selection(panel, selection)

        changed.assert_not_called()
        self.assertEqual(display.roi(), selection.rect)
        np.testing.assert_array_equal(display.roi_mask(), mask)

    def test_manual_roi_change_preserves_needle_and_roi_independent_analysis(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        needle_mask = np.zeros((20, 20), dtype=np.uint8)
        needle_mask[2:18, 1:4] = 255
        panel = SimpleNamespace(
            label="Video",
            needle_segmentation_mask=needle_mask,
            roi=Mock(return_value=QRect(5, 5, 8, 8)),
            roi_mask=Mock(return_value=np.ones((8, 8), dtype=bool)),
        )
        window.panels = [panel]
        self.addCleanup(lambda: setattr(window, "panels", []))
        time = np.asarray([0.0, 0.1])
        window.results = {"Video": Mock()}
        window.frame_brightness_results = {
            "Video": (time, np.asarray([100.0, 101.0]), np.asarray([90.0, 91.0]))
        }
        window.needle_brightness_results = {"Video": (time, np.asarray([20.0, 21.0]))}
        window.needle_brightness_baselines = {"Video": 20.5}
        window.temporal_change_results = {
            "Video": (np.ones((20, 20), dtype=np.float32), np.ones((20, 20), dtype=np.float32))
        }

        with patch.object(window, "_update_stage_statuses"), patch.object(
            window,
            "_has_enabled_stage",
            return_value=False,
        ):
            window.on_roi_changed()

        self.assertEqual(window.results, {})
        self.assertIs(panel.needle_segmentation_mask, needle_mask)
        self.assertIn("Video", window.needle_brightness_results)
        self.assertEqual(window.needle_brightness_baselines, {"Video": 20.5})
        self.assertIn("Video", window.frame_brightness_results)
        self.assertIn("Video", window.temporal_change_results)

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

    def test_report_frame_uses_the_enhanced_sequence_from_roi_analysis(self) -> None:
        analysis_encoded = [cv2.imencode(".png", np.full((2, 2), value, dtype=np.uint8))[1] for value in (10, 20)]
        current_encoded = [cv2.imencode(".png", np.full((2, 2), value, dtype=np.uint8))[1] for value in (100, 200)]
        panel = SimpleNamespace(
            report_encoded_frames=analysis_encoded,
            enhanced_frames=current_encoded,
            current_frame=np.full((2, 2, 3), 250, dtype=np.uint8),
        )

        frame = main._report_frame(panel, 1)

        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(frame, np.full((2, 2, 3), 20, dtype=np.uint8))

    def test_comparison_pdf_report_writes_matched_heatmaps_before_residence_curve(self) -> None:
        QApplication.instance() or QApplication([])
        frame = np.full((40, 60), 120, dtype=np.uint8)
        encoded = cv2.imencode(".png", frame)[1]
        mask_image = np.zeros((12, 16), dtype=np.uint8)
        cv2.circle(mask_image, (8, 6), 5, 1, thickness=-1)
        mask = mask_image.astype(bool)
        heatmap = np.full((40, 60, 3), (10, 80, 180), dtype=np.uint8)
        panels = [
            SimpleNamespace(
                label="Pre-deployment",
                color=QColor("#38bdf8"),
                path=Path("pre.mp4"),
                enhanced_frames=[encoded],
                current_frame=None,
                roi=lambda: QRect(10, 8, 16, 12),
                roi_mask=lambda: mask,
                temporal_change_heatmap=heatmap,
            ),
            SimpleNamespace(
                label="Post-deployment",
                color=QColor("#f97316"),
                path=Path("post.mp4"),
                enhanced_frames=[encoded],
                current_frame=None,
                roi=lambda: QRect(10, 8, 16, 12),
                roi_mask=lambda: mask,
                temporal_change_heatmap=heatmap,
            ),
        ]
        results = {
            panel.label: SimpleNamespace(
                time=np.asarray([0.0, 0.5, 1.0]),
                normalized_signal=np.asarray([0.0, 1.0, 0.2]),
                fps=2.0,
            )
            for panel in panels
        }
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "comparison.pdf"
            with patch.object(main, "_draw_report_image", wraps=main._draw_report_image) as draw_image:
                self.assertTrue(main.render_comparison_report(report_path, panels, results, 0))
            self.assertTrue(report_path.read_bytes().startswith(b"%PDF"))
            self.assertGreater(report_path.stat().st_size, 1_000)
        self.assertEqual(draw_image.call_count, 4)
        frame_targets = [call.args[-1] for call in draw_image.call_args_list[:2]]
        heatmap_targets = [call.args[-1] for call in draw_image.call_args_list[2:]]
        self.assertEqual([target.size() for target in heatmap_targets], [target.size() for target in frame_targets])
        self.assertGreater(heatmap_targets[0].y(), frame_targets[0].y())

    def test_report_heatmap_uses_print_friendly_grayscale_without_changing_gui_heatmap(self) -> None:
        heatmap = np.asarray([[[20, 30, 240], [240, 220, 30]]], dtype=np.uint8)

        optimized = main._print_optimized_heatmap(heatmap)

        np.testing.assert_array_equal(heatmap, np.asarray([[[20, 30, 240], [240, 220, 30]]], dtype=np.uint8))
        np.testing.assert_array_equal(optimized[..., 0], optimized[..., 1])
        np.testing.assert_array_equal(optimized[..., 1], optimized[..., 2])
        self.assertNotEqual(int(optimized[0, 0, 0]), int(optimized[0, 1, 0]))

    def test_pdf_export_generates_missing_heatmaps_before_rendering(self) -> None:
        panels = [
            SimpleNamespace(temporal_change_heatmap=None),
            SimpleNamespace(temporal_change_heatmap=None),
        ]
        results = {
            "Pre-deployment": SimpleNamespace(
                normalized_signal=np.asarray([0.1, 0.4, 0.2]),
                time=np.asarray([0.0, 0.1, 0.2]),
                fps=10.0,
            ),
            "Post-deployment": SimpleNamespace(
                normalized_signal=np.asarray([0.1, 1.0, 0.2]),
                time=np.asarray([0.0, 0.1, 0.2]),
                fps=10.0,
            ),
        }
        status_bar = Mock()
        window = SimpleNamespace(
            active_mode=MODE_COMPARISON,
            panels=panels,
            results=results,
            current_frame_index=3,
            run_temporal_change_heatmap=Mock(return_value=True),
            statusBar=Mock(return_value=status_bar),
        )

        with (
            patch.object(main.QFileDialog, "getSaveFileName", return_value=("comparison.pdf", "PDF files (*.pdf)")),
            patch.object(main, "render_comparison_report", return_value=True) as render_report,
        ):
            ContrastWindow.export_pdf_report(window)

        window.run_temporal_change_heatmap.assert_called_once_with()
        render_report.assert_not_called()
        status_bar.showMessage.assert_called_once_with(
            "Building comparison heatmaps in the background. Export again when analysis completes."
        )

    def test_maximum_contrast_frame_uses_the_shared_signal_peak(self) -> None:
        results = {
            "Pre-deployment": SimpleNamespace(
                normalized_signal=np.asarray([0.1, 0.8, 0.2]),
                time=np.asarray([0.0, 0.1, 0.2]),
                fps=10.0,
            ),
            "Post-deployment": SimpleNamespace(
                normalized_signal=np.asarray([0.1, 1.0, 0.2]),
                time=np.asarray([0.0, 0.1, 0.2]),
                fps=10.0,
            ),
        }

        self.assertEqual(main.maximum_contrast_frame(results), 1)

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

    def test_camera_view_mask_excludes_border_from_brightness_and_temporal_analysis(self) -> None:
        camera_view_mask = np.zeros((20, 20), dtype=bool)
        camera_view_mask[3:17, 3:17] = True
        frames = [np.full((20, 20), 180, dtype=np.uint8) for _ in range(6)]
        frames[3][~camera_view_mask] = 0
        frames[4][~camera_view_mask] = 0
        frames[5][~camera_view_mask] = 0

        brightness = average_frame_brightness(frames, camera_view_mask)
        burden, peak = compute_temporal_change_summary(frames, fps=2.0, camera_view_mask=camera_view_mask)

        np.testing.assert_array_equal(brightness, np.full(6, 180.0))
        self.assertFalse(np.any(burden[~camera_view_mask]))
        self.assertFalse(np.any(peak[~camera_view_mask]))
        self.assertFalse(np.any(burden[camera_view_mask]))

    def test_camera_view_mask_excludes_border_from_roi_reference(self) -> None:
        gray = np.full((80, 80), 150, dtype=np.uint8)
        gray[:8] = 0
        camera_view_mask = np.ones(gray.shape, dtype=bool)
        camera_view_mask[:8] = False
        roi = QRect(25, 10, 20, 20)

        reference = main.reference_mean(gray, roi, camera_view_mask=camera_view_mask)

        self.assertEqual(reference, 150.0)

    def test_temporal_change_summary_emphasizes_persistent_dark_contrast(self) -> None:
        baseline = np.full((4, 4), 180, dtype=np.uint8)
        frames = [baseline.copy() for _ in range(3)]
        for frame_index in range(4):
            frame = baseline.copy()
            if frame_index == 0:
                frame[1, 1] = 130
            frame[1, 2] = 130
            frames.append(frame)

        burden, peak = compute_temporal_change_summary(frames, fps=1.0)

        self.assertGreater(burden[1, 2], burden[1, 1] * 3.5)
        self.assertAlmostEqual(float(peak[1, 1]), float(peak[1, 2]), delta=0.01)
        self.assertEqual(float(burden[0, 0]), 0.0)

    def test_temporal_change_summary_rejects_global_fluoroscopy_flicker(self) -> None:
        frames = [np.full((4, 4), value, dtype=np.uint8) for value in (180, 184, 177, 183, 178, 181)]

        burden, peak = compute_temporal_change_summary(frames, fps=2.0)

        np.testing.assert_array_equal(burden, np.zeros((4, 4), dtype=np.float32))
        np.testing.assert_array_equal(peak, np.zeros((4, 4), dtype=np.float32))

    def test_temporal_change_heatmap_is_black_for_static_pixels_and_brightest_for_long_residence(self) -> None:
        burden = np.asarray([[0.0, 1.0, 16.0]], dtype=np.float32)
        peak = np.asarray([[0.0, 50.0, 50.0]], dtype=np.float32)

        heatmap = main.render_temporal_change_heatmap(burden, peak, burden_peak=16.0, contrast_peak=50.0)

        np.testing.assert_array_equal(heatmap[0, 0], np.zeros(3, dtype=np.uint8))
        self.assertGreater(float(np.mean(heatmap[0, 2])), float(np.mean(heatmap[0, 1])) * 2.0)

    def test_temporal_change_heatmap_supports_distinct_colormaps(self) -> None:
        burden = np.asarray([[0.0, 4.0, 16.0]], dtype=np.float32)
        peak = np.asarray([[0.0, 50.0, 50.0]], dtype=np.float32)

        hot = main.render_temporal_change_heatmap(
            burden,
            peak,
            burden_peak=16.0,
            contrast_peak=50.0,
            colormap="hot",
        )
        viridis = main.render_temporal_change_heatmap(
            burden,
            peak,
            burden_peak=16.0,
            contrast_peak=50.0,
            colormap="viridis",
        )

        np.testing.assert_array_equal(hot[0, 0], np.zeros(3, dtype=np.uint8))
        np.testing.assert_array_equal(viridis[0, 0], np.zeros(3, dtype=np.uint8))
        self.assertFalse(np.array_equal(hot[0, 1:], viridis[0, 1:]))

    def test_temporal_change_heatmap_colormap_control_is_serialized(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        drawer = window._stage_drawers("temporal_change_heatmap")[0]
        combo = drawer.findChild(QComboBox, "heatmapColormap")

        self.assertIsNotNone(combo)
        assert combo is not None
        self.assertEqual([combo.itemData(index) for index in range(combo.count())], ["hot", "inferno", "turbo", "viridis", "cividis"])
        combo.setCurrentIndex(combo.findData("viridis"))

        self.assertEqual(window._drawer_control_values(drawer)["heatmapColormap"], "viridis")
        combo.setCurrentIndex(combo.findData("hot"))
        window._set_drawer_control_values(drawer, {"heatmapColormap": "viridis"})
        self.assertEqual(combo.currentData(), "viridis")

    def test_temporal_change_heatmap_colormap_change_only_rerenders_cached_results(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        drawer = window._stage_drawers("temporal_change_heatmap")[0]
        combo = drawer.findChild(QComboBox, "heatmapColormap")
        assert combo is not None
        window.temporal_change_results = {
            "video": (np.asarray([[1.0]], dtype=np.float32), np.asarray([[1.0]], dtype=np.float32))
        }

        with (
            patch.object(window, "refresh_temporal_change_views") as refresh,
            patch.object(window, "rebuild_enhancement_pipeline") as rebuild,
        ):
            combo.setCurrentIndex(combo.findData("inferno"))

        refresh.assert_called_once_with()
        rebuild.assert_not_called()

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
            self.assertEqual(call.kwargs, {"colormap": "hot"})

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

    def test_frame_brightness_analysis_combines_comparison_videos_on_one_plot(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.active_mode = MODE_COMPARISON
        window.panels = [
            SimpleNamespace(label="Pre-deployment", color=QColor("#38bdf8")),
            SimpleNamespace(label="Post-deployment", color=QColor("#f97316")),
        ]
        window.frame_brightness_results = {
            "Pre-deployment": (np.asarray([0.0, 1.0]), np.asarray([80.0, 81.0]), np.asarray([90.0, 91.0])),
            "Post-deployment": (np.asarray([0.0, 1.0]), np.asarray([82.0, 83.0]), np.asarray([92.0, 93.0])),
        }

        window.refresh_frame_brightness_plot()

        self.assertEqual(set(window.frame_brightness_plots), {"Comparison"})
        self.assertEqual(window.frame_brightness_layout.count(), 1)
        plot = window.frame_brightness_plots["Comparison"]
        self.assertEqual(len(plot.listDataItems()), 4)
        self.assertEqual(
            [item[1].text for item in plot.plotItem.legend.items],
            ["Pre-deployment original", "Pre-deployment enhanced", "Post-deployment original", "Post-deployment enhanced"],
        )

    def test_frame_brightness_analysis_without_enhancement_uses_a_single_curve(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        self.addCleanup(lambda: setattr(window, "panels", []))
        window.panels = [SimpleNamespace(label="Video", color=QColor("#38bdf8"))]
        brightness = np.asarray([80.0, 81.0])
        window.frame_brightness_results = {"Video": (np.asarray([0.0, 1.0]), brightness, brightness.copy())}

        window.refresh_frame_brightness_plot()

        plot = window.frame_brightness_plots["Video"]
        self.assertEqual(len(plot.listDataItems()), 1)
        self.assertEqual(plot.plotItem.legend.items[0][1].text, "Frame brightness")
        app.quit()

    def test_analysis_only_request_prepares_source_frames_for_brightness(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        panel = Mock()
        panel.estimate_prepare_work.return_value = 0.0
        panel.prepare_enhanced_frames.return_value = True
        window.panels = [panel]
        self.addCleanup(lambda: setattr(window, "panels", []))
        request = EnhancementRequest(
            generation=1,
            mode="ffdnet-native",
            model_label="FFDNet",
            stages=EnhancementStages(),
            parameters=EnhancementParameters(),
            noise_sigma=10,
            batch_size=1,
            precision="fp16",
            auto_crop=False,
            temporal_alignment=False,
            source_pipeline_current=True,
            prepare_source_frames=True,
            roi_parameters_by_panel=(EnhancementParameters(),),
        )

        self.assertTrue(window._run_enhancement_request(request, Event()))

        panel.prepare_enhanced_frames.assert_called_once()
        app.quit()

    def test_roi_needle_alignment_expands_whichever_video_has_the_narrower_range(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        window.active_mode = MODE_COMPARISON
        roi = main.ROISelection(QRect(5, 5, 8, 8), np.ones((8, 8), dtype=bool))
        needle_mask = np.zeros((20, 20), dtype=np.uint8)
        needle_mask[2:18, 1:4] = 255
        pre_frame = np.full((20, 20), 140, dtype=np.uint8)
        pre_frame[5:13, 5:13] = 100
        pre_frame[needle_mask > 0] = 50
        post_frame = np.full((20, 20), 180, dtype=np.uint8)
        post_frame[5:13, 5:13] = 170
        post_frame[needle_mask > 0] = 70
        panels = []
        for label, color, frame in (
            ("Pre-deployment", QColor("#38bdf8"), pre_frame),
            ("Post-deployment", QColor("#f97316"), post_frame),
        ):
            panel = Mock()
            panel.label = label
            panel.color = color
            panel.info = SimpleNamespace(fps=10.0)
            panel.camera_view_mask = None
            panel.source_gray_frames = [frame]
            panel.stage_frame_cache = {}
            panel.prepare_enhanced_frames.return_value = True
            panel._sequence_key.return_value = (("roi_extraction", ()),)
            panel._frame_sequence_key.return_value = ()
            panel._roi_selection_for_sequence.return_value = roi
            panels.append(panel)
        window.panels = panels
        self.addCleanup(lambda: setattr(window, "panels", []))
        stages = EnhancementStages(
            instances=(
                main.PipelineStage("roi_extraction", True),
                main.PipelineStage("roi_needle_level_alignment", True),
            )
        )
        request = EnhancementRequest(
            generation=1,
            mode="ffdnet-native",
            model_label="FFDNet",
            stages=stages,
            parameters=EnhancementParameters(),
            noise_sigma=10,
            batch_size=1,
            precision="fp16",
            auto_crop=False,
            temporal_alignment=False,
            source_pipeline_current=True,
            roi_parameters_by_panel=(EnhancementParameters(), EnhancementParameters()),
        )

        with patch.object(
            main,
            "measure_roi_needle_baselines",
            side_effect=((100.0, 50.0, needle_mask), (170.0, 70.0, needle_mask)),
        ):
            window._prepare_roi_needle_level_alignment(request, [None, None], "none", Event())

        self.assertAlmostEqual(panels[0].roi_needle_alignment_gain, 2.0)
        self.assertAlmostEqual(panels[0].roi_needle_alignment_offset, -30.0)
        self.assertEqual(panels[1].roi_needle_alignment_gain, 1.0)
        self.assertEqual(panels[1].roi_needle_alignment_offset, 0.0)
        pre_result = window.roi_needle_alignment_results["Pre-deployment"]
        self.assertAlmostEqual(pre_result.roi_baseline_before, 100.0)
        self.assertAlmostEqual(pre_result.needle_baseline_before, 50.0)
        self.assertAlmostEqual(pre_result.roi_baseline_after, 170.0)
        self.assertAlmostEqual(pre_result.needle_baseline_after, 70.0)
        window.refresh_roi_needle_alignment_plot()
        self.assertTrue(window.analysis_tabs.isTabVisible(window.roi_needle_alignment_tab_index))
        self.assertEqual(
            [item.name() for item in window.roi_needle_alignment_plot.listDataItems()],
            [
                "Pre-deployment ROI before",
                "Pre-deployment ROI after",
                "Post-deployment ROI before",
                "Post-deployment ROI after",
            ],
        )
        reference_lines = [
            item
            for item in window.roi_needle_alignment_plot.plotItem.items
            if isinstance(item, main.pg.InfiniteLine)
        ]
        self.assertEqual(len(reference_lines), 8)
        prefix_stages = panels[0].prepare_enhanced_frames.call_args.args[3]
        self.assertEqual(prefix_stages.enabled_stage_order, ("roi_extraction",))

        with patch.object(
            main,
            "measure_roi_needle_baselines",
            side_effect=((170.0, 70.0, needle_mask), (100.0, 50.0, needle_mask)),
        ):
            window._prepare_roi_needle_level_alignment(request, [None, None], "none", Event())

        self.assertEqual(panels[0].roi_needle_alignment_gain, 1.0)
        self.assertEqual(panels[0].roi_needle_alignment_offset, 0.0)
        self.assertAlmostEqual(panels[1].roi_needle_alignment_gain, 2.0)
        self.assertAlmostEqual(panels[1].roi_needle_alignment_offset, -30.0)
        app.quit()

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
        window.temporal_end_trim_spin.setValue(1.50)
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
        self.assertEqual(window.temporal_end_trim_spin.value(), 1.50)
        self.assertEqual(window.comparison_sync_offset_spin.value(), 0.12)

    def test_config_apply_starts_one_pipeline_after_all_controls_are_loaded(self) -> None:
        QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        config = {
            "version": 1,
            "videos": {"mode": "single", "paths": ["video.avi"]},
            "pipeline": [
                {"key": "auto_crop", "enabled": True, "controls": {"autoCropSizeOffset": -60}},
                {"key": "needle_segmentation", "enabled": True, "controls": {}},
            ],
            "view": {
                "show_source": False,
                "mask_overlay_enabled": True,
                "playback_speed": 125,
                "loop": False,
                "frame_index": 0,
                "roi_settings": [{"mode": "auto", "manual_circle": None}],
            },
            "analysis": {"clearance_threshold": 0.2},
        }

        with (
            patch.object(window, "_set_video_panels"),
            patch.object(window, "rebuild_enhancement_pipeline") as rebuild,
            patch.object(window, "clear_plots_and_metrics"),
        ):
            window._apply_config(config, [Path("video.avi")])

        rebuild.assert_called_once_with()
        self.assertTrue(window._has_enabled_stage("needle_segmentation"))
        self.assertTrue(window.overlay_mask_check.isChecked())

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

    def test_analysis_normalizes_multiple_videos_to_each_curve_apex(self) -> None:
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

        np.testing.assert_allclose(results[pre.label].normalized_signal, [0.0, 0.0, 1.0, 0.0])
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

    def test_softened_roi_remains_inside_camera_view(self) -> None:
        frames: list[np.ndarray] = []
        for index in range(8):
            frame = np.full((96, 96), 180, dtype=np.uint8)
            cv2.circle(frame, (18, 48), 14, 178 if index < 3 else 60, thickness=-1)
            frames.append(frame)
        camera_view_mask = np.ones((96, 96), dtype=bool)
        camera_view_mask[:, :10] = False

        roi = detect_aneurysm_roi(
            frames,
            fps=10.0,
            camera_view_mask=camera_view_mask,
            soften_mask=True,
            soften_radius_ratio=0.25,
            soften_threshold=0.10,
        )

        self.assertIsNotNone(roi)
        assert roi is not None
        full_mask = np.zeros(camera_view_mask.shape, dtype=bool)
        x, y, width, height = roi.rect.getRect()
        full_mask[y : y + height, x : x + width] = roi.mask
        self.assertFalse(np.any(full_mask[~camera_view_mask]))

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

    def test_circle_fit_refines_detected_roi_when_temporal_segmentation_has_no_region(self) -> None:
        detected_mask = np.zeros((50, 50), dtype=bool)
        cv2.circle(detected_mask, (22, 25), 12, 1, thickness=-1)
        cv2.rectangle(detected_mask, (31, 20), (44, 30), 1, thickness=-1)
        detected_roi = main.ROISelection(QRect(40, 30, 50, 50), detected_mask)
        frames = [np.full((100, 120), 180, dtype=np.uint8) for _ in range(3)]

        with (
            patch.object(main, "extract_aneurysm_roi", return_value=detected_roi),
            patch.object(main, "segment_temporal_change_map", return_value=np.zeros((100, 120), dtype=np.uint8)),
        ):
            roi, regions = main.extract_aneurysm_regions(
                frames,
                fps=10.0,
                parameters=EnhancementParameters(roi_circle_fit_enabled=True),
            )

        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertEqual(roi.rect, detected_roi.rect)
        np.testing.assert_array_equal(roi.mask, fit_circle_to_convex_hull(detected_mask))
        self.assertFalse(roi.mask[25, 44])
        self.assertFalse(np.any(regions))

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
            _analysis_future=None,
            panels=[panel],
            current_frame_index=0,
            _set_playback_limit=playback_limits.append,
        )

        ContrastWindow._poll_enhancement(window)

        self.assertFalse(pending_future.done())
        self.assertIs(panel.roi_region_masks[0], encoded_mask)
        self.assertEqual(seeks, [0])
        self.assertEqual(playback_limits, [0])

    def test_poll_restores_stream_targets_after_source_pipeline_clears_cache(self) -> None:
        encoded_frame = np.array([4, 5, 6], dtype=np.uint8)
        encoded_mask = np.array([1, 2, 3], dtype=np.uint8)
        panel = SimpleNamespace(
            enhanced_frames=[],
            roi_region_masks=[],
            enhance_display=True,
            seek=Mock(),
        )
        source_events: SimpleQueue[tuple[int, list[object], Event]] = SimpleQueue()
        source_applied = Event()
        source_events.put((7, [object()], source_applied))
        frame_events: SimpleQueue[tuple[int, int, int, np.ndarray]] = SimpleQueue()
        frame_events.put((7, 0, 0, encoded_frame))
        mask_events: SimpleQueue[tuple[int, int, int, np.ndarray]] = SimpleQueue()
        mask_events.put((7, 0, 0, encoded_mask))
        pending_future: Future[bool] = Future()

        def clear_panel_cache(_states):  # noqa: ANN001
            panel.enhanced_frames = None
            panel.roi_region_masks = None
            return True

        window = SimpleNamespace(
            _source_pipeline_events=source_events,
            _enhancement_frame_events=frame_events,
            _roi_region_mask_events=mask_events,
            _enhancement_generation=7,
            _enhancement_active_request=SimpleNamespace(
                generation=7,
                stages=SimpleNamespace(roi_extraction=True),
            ),
            _enhancement_future=pending_future,
            _analysis_future=None,
            _enhancement_progress_lock=main.Lock(),
            _enhancement_progress_values=[0.0],
            _enhancement_progress_totals=[1.0],
            _enhancement_stage_messages=["Encoding"],
            _enhancement_message="Preparing",
            enhancement_progress=SimpleNamespace(
                message_label=SimpleNamespace(setText=Mock()),
                set_progress=Mock(),
                set_panel_progress=Mock(),
            ),
            _apply_source_pipeline_states=clear_panel_cache,
            clear_plots_and_metrics=Mock(),
            panels=[panel],
            results={},
            current_frame_index=0,
            _set_playback_limit=Mock(),
        )

        ContrastWindow._poll_enhancement(window)

        self.assertTrue(source_applied.is_set())
        self.assertIs(panel.enhanced_frames[0], encoded_frame)
        self.assertIs(panel.roi_region_masks[0], encoded_mask)
        panel.seek.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()