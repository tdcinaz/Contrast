from __future__ import annotations

import http.client
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

import cv2
import numpy as np

from main import EnhancementParameters, EnhancementStages
from stream_server import LiveStreamProcessor, StreamService, StreamSettings, create_http_server, load_stream_configuration


class StreamServerTests(unittest.TestCase):
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