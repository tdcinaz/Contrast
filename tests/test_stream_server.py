from __future__ import annotations

import http.client
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PySide6.QtCore import QRect

from main import (
    MODE_COMPARISON,
    MODE_SINGLE,
    CollapsibleDrawer,
    ContrastWindow,
    EnhancementParameters,
    EnhancementStages,
    PipelineStage,
    VideoDropPlaceholder,
    VideoPanel,
)
from contrast_pipeline import subtract_fluoroscopy_background
from stream_server import RawFrameRecorder, LiveStreamProcessor, StreamService, StreamSettings, create_http_server, load_stream_configuration


class StreamServerTests(unittest.TestCase):
    def test_mode_selection_panel_scales_when_graph_drawer_opens(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window.resize(1280, 760)
        window.show()
        QApplication.processEvents()

        self.assertIs(window.video_stack.currentWidget(), window.mode_selection_page)
        self.assertEqual(window.mode_selection_panel.size().width(), 720)
        self.assertEqual(window.mode_selection_panel.size().height(), 560)
        self.assertIn("background: #111827;", window.mode_selection_panel.styleSheet())

        window._set_graph_drawer_expanded(True)
        QApplication.processEvents()

        view = window.mode_selection_view
        self.assertLess(view.transform().m11(), 1.0)
        self.assertTrue(view.viewport().rect().contains(view.mapFromScene(view.sceneRect()).boundingRect()))

        window.close()
        app.quit()

    def test_playback_controls_move_up_after_selecting_comparison_mode(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window.show()
        QApplication.processEvents()

        window._show_video_placeholders(MODE_COMPARISON)
        QApplication.processEvents()
        baseline_playback_y = window.playback_row.y()
        baseline_stack_height = window.video_stack.height()
        self.assertTrue(window.video_placeholders)
        for placeholder in window.video_placeholders:
            assert placeholder is not None
            self.assertEqual(placeholder.height(), placeholder.heightForWidth(placeholder.width()))
            surface = placeholder.video_surface
            button = placeholder.hint_button
            self.assertLessEqual(abs(button.x() - (surface.width() - button.width()) // 2), 1)
            self.assertLessEqual(abs(button.y() - (surface.height() - button.height()) // 2), 1)

        splitter = window.main_splitter
        sizes = splitter.sizes()
        total_width = max(1, sum(sizes) or splitter.width())
        expanded_left = min(total_width - 1, sizes[0] + 120)
        splitter.setSizes([expanded_left, max(1, total_width - expanded_left)])
        QApplication.processEvents()

        self.assertIs(window.video_stack.currentWidget(), window.video_placeholder_row)
        self.assertLessEqual(window.video_stack.height(), baseline_stack_height)
        self.assertLessEqual(window.playback_row.y(), baseline_playback_y)
        self.assertEqual(
            window.playback_row.y(),
            window.video_stack.geometry().bottom() + window.video_stack.parentWidget().layout().spacing() + 1,
        )

        window.close()
        app.quit()

    def test_placeholder_selection_keeps_video_panel_inside_main_window(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window.show()

        with TemporaryDirectory() as directory:
            video_path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32), True)
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()

            window._show_video_placeholders(MODE_SINGLE)
            top_level_widgets_before = set(QApplication.topLevelWidgets())
            window._set_placeholder_video_path(0, video_path)
            QApplication.processEvents()

            self.assertEqual(len(window.panels), 1)
            self.assertIs(window.panels[0].parentWidget(), window.video_row)
            self.assertNotIn(window.panels[0], QApplication.topLevelWidgets())
            self.assertFalse(
                any(
                    widget.isVisible() and isinstance(widget, (VideoDropPlaceholder, VideoPanel))
                    for widget in set(QApplication.topLevelWidgets()) - top_level_widgets_before
                )
            )

        window.close()
        app.quit()

    def test_window_keeps_pipeline_drawer_expanded_by_default(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window.show()
        QApplication.processEvents()

        self.assertTrue(window.pipeline_drawer.toggle_button.isChecked())
        self.assertTrue(window.pipeline_drawer.content.isVisible())

        window.close()
        app.quit()

    def test_show_source_toggle_is_off_by_default(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window.show()
        QApplication.processEvents()

        self.assertFalse(window.compare_view_check.isChecked())

        window.close()
        app.quit()

    def test_live_mode_starts_network_stream_without_file_selection(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)

        with patch.object(window, "_start_desktop_stream_service", return_value=True) as start_stream, patch.object(
            window, "_show_video_placeholders"
        ) as show_placeholders:
            self.assertTrue(window._select_mode_and_videos("live"))

        start_stream.assert_called_once()
        show_placeholders.assert_not_called()
        self.assertEqual(window.active_mode, "live")
        self.assertIs(window.video_stack.currentWidget(), window.video_row)
        self.assertIsNotNone(window._network_stream_display)
        self.assertIn("Network live stream active", window.statusBar().currentMessage())
        app.quit()

    def test_live_mode_shows_recording_name_controls(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        service = MagicMock()
        service.configure_recording.return_value = Path("recordings/device_test_pre_0.avi")
        service.latest_frames.return_value = (0, None, None)
        window._stream_service = service
        window._stream_server = MagicMock()
        window.show()

        window._activate_network_stream_mode()
        window.live_device_name_edit.setText("C-arm 1")
        window.live_test_identifier_edit.setText("Case 7")
        window.live_phase_toggle.setChecked(True)

        self.assertFalse(window.play_button.isVisible())
        self.assertFalse(window.frame_label.isVisible())
        self.assertFalse(window.speed_control_label.isVisible())
        self.assertTrue(window.live_recording_controls.isVisible())
        self.assertTrue(window.live_export_toggle.isVisible())
        self.assertTrue(window.live_export_toggle.isChecked())
        self.assertTrue(window.compare_view_check.isVisible())
        self.assertEqual(service.configure_recording.call_args.args, ("C-arm 1", "Case 7", "post"))
        self.assertEqual(window.live_recording_name_label.text(), "device_test_pre_0.avi")

        window.close()
        app.quit()

    def test_window_close_finalizes_active_stream_recording(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        service = MagicMock()
        window._stream_service = service

        window.close()

        service.close.assert_called_once()
        app.quit()

    def test_network_live_mode_renders_matched_source_and_enhanced_frames(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=False,
        )
        service = StreamService(processor, max_frame_bytes=1024 * 1024)
        frame = np.full((48, 64, 3), 150, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        service.ingest(bytes(encoded))

        window = ContrastWindow()
        window._stream_server = MagicMock()
        window._stream_service = service
        self.addCleanup(window.close)
        window._select_mode_and_videos("live")
        window._poll_network_stream()

        display = window._network_stream_display
        self.assertIsNotNone(display)
        assert display is not None
        self.assertFalse(display._left_pixmap.isNull())
        self.assertFalse(display._right_pixmap.isNull())
        app.quit()

    def test_network_live_mode_updates_manual_roi_and_frame_brightness_analysis(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window._stream_server = MagicMock()
        self.addCleanup(window.close)
        window._select_mode_and_videos("live")
        display = window._network_stream_display
        assert display is not None
        display.set_roi(QRect(8, 8, 24, 24))
        for stage_key in ("roi_residence_analysis", "frame_brightness_analysis"):
            window._add_pipeline_stage(stage_key).enable_button.setChecked(True)

        source = np.full((48, 64, 3), 150, dtype=np.uint8)
        enhanced = np.full((48, 64), 120, dtype=np.uint8)
        window._record_live_measurements(source, enhanced)
        window._record_live_measurements(source, enhanced)

        self.assertIn("Live camera", window.frame_brightness_results)
        self.assertIn("Live camera", window.results)
        self.assertIn("Live camera", window.frame_brightness_plots)
        app.quit()

    def test_live_add_stage_menu_includes_analysis_stages(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        window._stream_server = MagicMock()
        self.addCleanup(window.close)
        window._select_mode_and_videos("live")
        window._show_add_stage_menu("live")

        action_labels = {action.text() for action in window._stage_menu.actions()}
        self.assertIn("ROI residence analysis", action_labels)
        self.assertIn("Frame brightness analysis", action_labels)
        app.quit()

    def test_network_pipeline_creates_denoiser_for_enabled_stage_instance(self) -> None:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        stage_instances = EnhancementStages(instances=(PipelineStage("denoise", True),))
        denoiser = MagicMock()
        denoiser.backend_id = "ffdnet-ngc-test"

        with patch("container_denoiser.ContainerDenoiser", return_value=denoiser) as container:
            self.assertIs(window._live_denoiser_for(stage_instances), denoiser)

        container.assert_called_once()
        app.quit()

    def test_pipeline_drawer_toggle_hides_content(self) -> None:
        from PySide6.QtWidgets import QApplication, QWidget

        app = QApplication.instance() or QApplication([])
        parent = QWidget()
        parent.show()
        drawer = CollapsibleDrawer("Pipeline")
        drawer.setParent(parent)
        content = QWidget()
        drawer.content_layout.addWidget(content)
        drawer.show()

        drawer.set_expanded(False)
        self.assertFalse(content.isVisible())

        drawer.set_expanded(True)
        self.assertTrue(content.isVisible())

        parent.close()
        app.quit()

    def test_pipeline_drawer_hides_title_when_collapsed(self) -> None:
        from PySide6.QtWidgets import QApplication, QWidget

        app = QApplication.instance() or QApplication([])
        parent = QWidget()
        parent.show()
        drawer = CollapsibleDrawer("Pipeline")
        drawer.setParent(parent)
        drawer.show()

        drawer.set_expanded(True)
        self.assertTrue(drawer.title_label.isVisible())

        drawer.set_expanded(False)
        self.assertFalse(drawer.title_label.isVisible())

        parent.close()
        app.quit()

    def setUp(self) -> None:
        image = np.full((48, 64, 3), 150, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        self.frame = bytes(encoded)

    def test_processor_waits_for_fixed_crop_samples(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=True,
        )

        self.assertIsNone(processor.process_jpeg(self.frame))
        self.assertIsNone(processor.process_jpeg(self.frame))
        enhanced = processor.process_jpeg(self.frame)

        self.assertIsNotNone(enhanced)
        self.assertTrue(processor.crop_ready)

    def test_processor_reapplies_live_crop_size_offset(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=True,
        )
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        cv2.circle(frame, (200, 120), 110, (160, 160, 160), thickness=-1)

        for _ in range(2):
            self.assertIsNone(processor.process_frame(frame))
        base_output = processor.process_frame(frame)
        self.assertIsNotNone(base_output)
        processor.configure(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            auto_crop_enabled=True,
            denoiser=None,
            auto_crop_size_offset=32,
        )
        adjusted_output = processor.process_frame(frame)

        assert base_output is not None
        assert adjusted_output is not None
        base_frame = cv2.imdecode(np.frombuffer(base_output, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        adjusted_frame = cv2.imdecode(np.frombuffer(adjusted_output, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(base_frame.shape, (160, 160))
        self.assertEqual(adjusted_frame.shape, (192, 192))

    def test_live_dsa_waits_one_second_before_acquiring_mask(self) -> None:
        stages = EnhancementStages(instances=(PipelineStage("background_subtraction", True),))
        processor = LiveStreamProcessor(
            stages,
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=100,
            auto_crop_enabled=False,
            dsa_mask_delay_frames=3,
        )
        processor.begin_recording()
        baseline = np.full((48, 64, 3), 150, dtype=np.uint8)
        contrast = np.full((48, 64, 3), 100, dtype=np.uint8)

        for _ in range(2):
            output = processor.process_frame(baseline)
            decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.assertLess(abs(float(decoded.mean()) - 150), 3)

        output = processor.process_frame(baseline)
        decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertGreater(float(decoded.mean()), 250)

        output = processor.process_frame(contrast)
        decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        expected = subtract_fluoroscopy_background(
            np.full((48, 64), 100, dtype=np.uint8),
            np.full((48, 64), 150, dtype=np.uint8),
            0,
        )
        self.assertLess(abs(float(decoded.mean()) - float(expected.mean())), 3)

    def test_live_dsa_resets_for_each_detected_recording(self) -> None:
        stages = EnhancementStages(instances=(PipelineStage("background_subtraction", True),))
        processor = LiveStreamProcessor(
            stages,
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=100,
            auto_crop_enabled=False,
            dsa_mask_delay_frames=1,
        )
        with TemporaryDirectory() as directory:
            service = StreamService(
                processor,
                max_frame_bytes=1024 * 1024,
                recorder=RawFrameRecorder(directory, fps=15.0),
            )

            def ingest_level(level: int) -> np.ndarray:
                frame = np.full((48, 64, 3), level, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
                _frame_id, _source, enhanced = service.latest_frames()
                assert enhanced is not None
                return cv2.imdecode(np.frombuffer(enhanced, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

            ingest_level(40)
            first_mask = ingest_level(150)
            self.assertGreater(float(first_mask.mean()), 250)
            ingest_level(150)
            ingest_level(150)
            ingest_level(150)

            ingest_level(70)
            second_mask = ingest_level(180)
            self.assertGreater(float(second_mask.mean()), 250)
            contrast = ingest_level(120)
            self.assertLess(float(contrast.mean()), 200)

    def test_raw_recorder_waits_for_three_identical_frames_before_cutting(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=False,
        )
        first = np.full((48, 64, 3), 40, dtype=np.uint8)
        second = np.full((48, 64, 3), 120, dtype=np.uint8)
        third = np.full((48, 64, 3), 200, dtype=np.uint8)
        fourth = np.full((48, 64, 3), 230, dtype=np.uint8)
        with TemporaryDirectory() as directory:
            recorder = RawFrameRecorder(directory, fps=15.0)
            service = StreamService(processor, max_frame_bytes=1024 * 1024, recorder=recorder)
            for frame in (first, second, second, third, fourth, fourth, fourth):
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
            service.close()

            self.assertEqual(len(recorder.completed_paths), 1)
            counts: list[int] = []
            values: list[list[float]] = []
            for path in recorder.completed_paths:
                capture = cv2.VideoCapture(str(path))
                frames: list[np.ndarray] = []
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                capture.release()
                counts.append(len(frames))
                values.append([float(frame.mean()) for frame in frames])

            self.assertEqual(counts, [6])
            self.assertLess(abs(values[0][0] - 40), 5)
            self.assertLess(abs(values[0][1] - 120), 5)
            self.assertLess(abs(values[0][2] - 120), 5)
            self.assertLess(abs(values[0][3] - 200), 5)
            self.assertLess(abs(values[0][4] - 230), 5)
            self.assertLess(abs(values[0][5] - 230), 5)

    def test_raw_recorder_ignores_paused_frames_at_startup(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=False,
        )
        paused = np.full((48, 64, 3), 40, dtype=np.uint8)
        first_live = np.full((48, 64, 3), 120, dtype=np.uint8)
        with TemporaryDirectory() as directory:
            recorder = RawFrameRecorder(directory, fps=15.0)
            service = StreamService(processor, max_frame_bytes=1024 * 1024, recorder=recorder)
            for frame in (paused, paused, first_live, first_live):
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
            service.close()

            self.assertEqual(recorder.completed_paths, [])

    def test_raw_recorder_uses_configured_name_and_clip_number(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=False,
        )
        frames = [
            np.full((48, 64, 3), value, dtype=np.uint8)
            for value in (20, 40, 40, 40, 60, 80, 80, 80, 100, 120, 120, 120)
        ]
        with TemporaryDirectory() as directory:
            recorder = RawFrameRecorder(directory, fps=15.0)
            service = StreamService(processor, max_frame_bytes=1024 * 1024, recorder=recorder)
            service.configure_recording("C-arm 1", "Case 7", "pre")
            for frame in frames[:8]:
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
            service.configure_recording("C-arm 1", "Case 7", "post")
            for frame in frames[8:]:
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
            service.close()

            self.assertEqual(
                [path.name for path in recorder.completed_paths],
                ["C-arm_1_Case_7_pre_0.avi", "C-arm_1_Case_7_pre_1.avi", "C-arm_1_Case_7_post_0.avi"],
            )

    def test_recording_can_be_disabled_and_reenabled(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=False,
        )
        frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in (20, 40, 40, 60, 80, 80)]
        with TemporaryDirectory() as directory:
            recorder = RawFrameRecorder(directory, fps=15.0)
            service = StreamService(processor, max_frame_bytes=1024 * 1024, recorder=recorder)
            service.set_recording_enabled(False)
            for frame in frames[:3]:
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
            service.set_recording_enabled(True)
            for frame in frames[3:]:
                ok, encoded = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                service.ingest(bytes(encoded))
            service.close()

            self.assertEqual(len(recorder.completed_paths), 1)

    def test_http_ingest_and_mjpeg_egress(self) -> None:
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            noise_sigma=10,
            crop_sample_frames=3,
            jpeg_quality=92,
            auto_crop_enabled=False,
        )
        service = StreamService(processor, max_frame_bytes=1024 * 1024)
        server = create_http_server(StreamSettings(host="127.0.0.1", port=0), service)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        ingest = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        ingest.request("POST", "/ingest", body=self.frame, headers={"Content-Type": "image/jpeg"})
        response = ingest.getresponse()
        self.assertEqual(response.status, 202)
        self.assertTrue(json.loads(response.read())["egress_ready"])
        ingest.close()

        egress = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        egress.request("GET", "/egress.mjpg")
        response = egress.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn("multipart/x-mixed-replace", response.getheader("Content-Type"))
        self.assertIn(b"Content-Type: image/jpeg", response.read(256))
        egress.close()

    def test_rejects_temporal_stages_in_headless_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "stream.json"
            config_path.write_text(json.dumps({
                "stream": {},
                "pipeline": [{"key": "temporal_filter", "enabled": True, "controls": {}}],
            }))
            with self.assertRaisesRegex(ValueError, "temporal_filter"):
                load_stream_configuration(str(config_path))

    def test_accepts_dsa_mask_subtraction_in_headless_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "stream.json"
            config_path.write_text(json.dumps({
                "stream": {},
                "pipeline": [{"key": "background_subtraction", "enabled": True, "controls": {}}],
            }))

            _settings, stages, _parameters, _strength, _auto_crop, _denoiser = load_stream_configuration(
                str(config_path)
            )

            self.assertIn("background_subtraction", stages.enabled_stage_order)

    def test_accepts_non_local_means_in_headless_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "stream.json"
            config_path.write_text(json.dumps({
                "stream": {},
                "pipeline": [{
                    "key": "denoise",
                    "enabled": True,
                    "controls": {
                        "denoiseMode": "non-local-means",
                        "denoiseStrength": 5,
                    },
                }],
            }))

            _settings, stages, _parameters, strength, _auto_crop, denoiser = load_stream_configuration(
                str(config_path)
            )

            self.assertIn("denoise", stages.enabled_stage_order)
            self.assertEqual(strength, 5)
            self.assertEqual(denoiser.mode, "non-local-means")

    def test_accepts_ngc_tensor_nlm_in_headless_configuration(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "stream.json"
            config_path.write_text(json.dumps({
                "stream": {},
                "pipeline": [{
                    "key": "denoise",
                    "enabled": True,
                    "controls": {
                        "denoiseMode": "tensor-nlm-ngc",
                        "denoiseStrength": 5,
                        "denoiseBatchSize": 4,
                        "denoisePrecision": "fp16",
                    },
                }],
            }))

            _settings, stages, _parameters, strength, _auto_crop, denoiser = load_stream_configuration(
                str(config_path)
            )

            self.assertIn("denoise", stages.enabled_stage_order)
            self.assertEqual(strength, 5)
            self.assertEqual(denoiser.mode, "tensor-nlm-ngc")
            self.assertEqual(denoiser.batch_size, 4)
            self.assertEqual(denoiser.precision, "fp16")


if __name__ == "__main__":
    unittest.main()