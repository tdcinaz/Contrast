from __future__ import annotations

import http.client
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

import cv2
import numpy as np

from main import MODE_COMPARISON, CollapsibleDrawer, ContrastWindow, EnhancementParameters, EnhancementStages
from stream_server import LiveStreamProcessor, StreamService, StreamSettings, create_http_server, load_stream_configuration


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


if __name__ == "__main__":
    unittest.main()