from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from threading import Condition, Lock
from typing import Any

import cv2
import numpy as np

from contrast_pipeline import (
    BUILTIN_STAGES,
    EnhancementParameters,
    EnhancementStages,
    FrameContext,
    FramePipelineExecutor,
    PipelineStage,
    subtract_fluoroscopy_background,
)
from main import (
    FrameDenoiser,
    _adjust_auto_crop_square,
    _detect_aligned_field_crop,
    _detect_pillarbox_crop,
    crop_frame,
)


JPEG_CONTENT_TYPES = {"image/jpeg", "image/jpg"}
LOGGER = logging.getLogger("contrast.stream")


@dataclass(frozen=True, slots=True)
class StreamSettings:
    host: str = "0.0.0.0"
    port: int = 8080
    crop_sample_frames: int = 24
    jpeg_quality: int = 92
    max_frame_bytes: int = 16 * 1024 * 1024
    recording_directory: str = "recordings"
    recording_fps: float = 15.0


@dataclass(frozen=True, slots=True)
class DenoiserSettings:
    mode: str = "ffdnet-ngc"
    precision: str = "fp16"
    batch_size: int = 1


def _control_value(controls: dict[str, object], key: str, default: Any) -> Any:
    value = controls.get(key, default)
    return value if isinstance(value, type(default)) else default


def _parameters(controls: dict[str, object]) -> EnhancementParameters:
    gain_min = float(_control_value(controls, "gainMinimum", 0.70))
    gain_max = float(_control_value(controls, "gainMaximum", 1.45))
    return EnhancementParameters(
        gain_use_auto_target=bool(_control_value(controls, "gainUseAutoTarget", True)),
        gain_target_median=int(_control_value(controls, "gainTargetMedian", 128)),
        gain_min=min(gain_min, gain_max),
        gain_max=max(gain_min, gain_max),
        scanline_bias_clip=float(_control_value(controls, "scanlineBiasClip", 6.0)),
        scanline_sigma_y=float(_control_value(controls, "scanlineSigmaY", 2.0)),
        clahe_clip_limit=float(_control_value(controls, "claheClipLimit", 1.0)),
        clahe_tile_size=int(_control_value(controls, "claheTileSize", 6)),
        adjustments_brightness_offset=int(_control_value(controls, "adjustmentsBrightness", 0)),
        adjustments_contrast_gain=float(_control_value(controls, "adjustmentsContrast", 1.0)),
        adjustments_sharpen_amount=float(_control_value(controls, "adjustmentsSharpen", 0.0)),
        adjustments_gamma=float(_control_value(controls, "adjustmentsGamma", 1.0)),
        smoothing_sigma_x=float(_control_value(controls, "smoothingSigma", 0.55)),
    )


def load_stream_configuration(
    config_path: str,
) -> tuple[StreamSettings, EnhancementStages, EnhancementParameters, int, bool, DenoiserSettings]:
    with open(config_path) as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be an object.")
    stream = config.get("stream", {})
    pipeline = config.get("pipeline", [])
    if not isinstance(stream, dict) or not isinstance(pipeline, list):
        raise ValueError("Headless configuration requires stream and pipeline sections.")
    settings = StreamSettings(
        host=str(stream.get("host", "0.0.0.0")),
        port=int(stream.get("port", 8080)),
        crop_sample_frames=max(3, int(stream.get("crop_sample_frames", 24))),
        jpeg_quality=max(1, min(100, int(stream.get("jpeg_quality", 92)))),
        max_frame_bytes=max(1024, int(stream.get("max_frame_bytes", 16 * 1024 * 1024))),
        recording_directory=str(stream.get("recording_directory", "recordings")),
        recording_fps=max(1.0, float(stream.get("recording_fps", 15.0))),
    )
    instances: list[PipelineStage] = []
    parameters = EnhancementParameters()
    noise_sigma = int(stream.get("noise_sigma", 10))
    auto_crop_enabled = False
    denoiser_settings = DenoiserSettings()
    for item in pipeline:
        if not isinstance(item, dict):
            raise ValueError("Each pipeline stage must be an object.")
        key = item.get("key")
        enabled = item.get("enabled")
        controls = item.get("controls", {})
        if not isinstance(key, str) or not isinstance(enabled, bool) or not isinstance(controls, dict):
            raise ValueError("Each pipeline stage requires key, enabled, and controls values.")
        definition = BUILTIN_STAGES.require(key)
        if key == "auto_crop":
            auto_crop_enabled = enabled
            continue
        stage_parameters = _parameters(controls)
        if enabled and not definition.supports_live(stage_parameters):
            raise ValueError(f"{key} is not supported for headless live streams.")
        if key == "denoise":
            noise_sigma = int(_control_value(controls, "denoiseStrength", noise_sigma))
            mode = str(_control_value(controls, "denoiseMode", "ffdnet-ngc"))
            if mode not in {"ffdnet-ngc", "ffdnet-native", "tensor-nlm-ngc", "non-local-means"}:
                raise ValueError(f"Unsupported headless denoiser mode: {mode}")
            denoiser_settings = DenoiserSettings(
                mode=mode,
                precision=str(_control_value(controls, "denoisePrecision", "fp16")),
                batch_size=max(1, int(_control_value(controls, "denoiseBatchSize", 1))),
            )
        instances.append(PipelineStage(key, enabled, stage_parameters, noise_sigma if key == "denoise" else None))
        parameters = stage_parameters
    stages = EnhancementStages(instances=tuple(instances))
    LOGGER.info(
        "Loaded stream configuration %s: host=%s port=%s auto_crop=%s enabled_stages=%s denoiser=%s",
        config_path,
        settings.host,
        settings.port,
        auto_crop_enabled,
        [stage.key for stage in stages.instances if stage.enabled],
        denoiser_settings.mode,
    )
    return settings, stages, parameters, noise_sigma, auto_crop_enabled, denoiser_settings


class LiveStreamProcessor:
    """Processes independent frames and retains one crop rectangle for a stream."""

    def __init__(
        self,
        stages: EnhancementStages,
        parameters: EnhancementParameters,
        noise_sigma: int,
        crop_sample_frames: int,
        jpeg_quality: int,
        auto_crop_enabled: bool,
        denoiser: FrameDenoiser | None = None,
        auto_crop_size_offset: int = 0,
        dsa_mask_delay_frames: int = 15,
        dsa_darkening_threshold: int = 0,
    ) -> None:
        self.stages = stages
        self.parameters = parameters
        self.noise_sigma = noise_sigma
        self.crop_sample_frames = crop_sample_frames
        self.jpeg_quality = jpeg_quality
        self.auto_crop_enabled = auto_crop_enabled
        self.denoiser = denoiser
        self.auto_crop_size_offset = auto_crop_size_offset
        self.dsa_mask_delay_frames = max(1, dsa_mask_delay_frames)
        self.dsa_darkening_threshold = max(0, dsa_darkening_threshold)
        self.pipeline = FramePipelineExecutor()
        self._crop_samples: list[np.ndarray] = []
        self._auto_crop_rect = None
        self._crop_rect = None
        self._shape: tuple[int, int] | None = None
        self._dsa_mask: np.ndarray | None = None
        self._dsa_recording_frame_count: int | None = None
        self._lock = Lock()
        LOGGER.info(
            "Initialized live stream processor: auto_crop=%s crop_samples=%s jpeg_quality=%s denoiser=%s",
            auto_crop_enabled,
            crop_sample_frames,
            jpeg_quality,
            getattr(denoiser, "backend_id", "none"),
        )

    @property
    def crop_ready(self) -> bool:
        return self._crop_rect is not None

    def configure(
        self,
        stages: EnhancementStages,
        parameters: EnhancementParameters,
        noise_sigma: int,
        auto_crop_enabled: bool,
        denoiser: FrameDenoiser | None,
        auto_crop_size_offset: int,
        dsa_darkening_threshold: int = 0,
    ) -> None:
        """Apply updated desktop controls before processing the next frame."""
        with self._lock:
            reset_crop = self.auto_crop_enabled != auto_crop_enabled
            self.stages = stages
            self.parameters = parameters
            self.noise_sigma = noise_sigma
            self.auto_crop_enabled = auto_crop_enabled
            self.denoiser = denoiser
            self.auto_crop_size_offset = auto_crop_size_offset
            self.dsa_darkening_threshold = max(0, dsa_darkening_threshold)
            if reset_crop:
                self._crop_samples.clear()
                self._auto_crop_rect = None
                self._crop_rect = None
            elif self._auto_crop_rect is not None and self._shape is not None:
                width, height = self._shape
                self._crop_rect = _adjust_auto_crop_square(
                    self._auto_crop_rect,
                    width,
                    height,
                    self.auto_crop_size_offset,
                )
        LOGGER.info(
            "Updated live stream pipeline: auto_crop=%s enabled_stages=%s denoiser=%s",
            auto_crop_enabled,
            [stage.key for stage in stages.instances if stage.enabled],
            getattr(denoiser, "backend_id", "none"),
        )

    def begin_recording(self) -> None:
        """Begin a new DSA acquisition window for a newly detected recording."""
        with self._lock:
            self._dsa_mask = None
            self._dsa_recording_frame_count = 0

    def reset_dsa_recording(self) -> None:
        with self._lock:
            self._dsa_mask = None
            self._dsa_recording_frame_count = None

    def process_jpeg(self, encoded: bytes) -> bytes | None:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Ingest body is not a valid JPEG frame.")
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray) -> bytes | None:
        with self._lock:
            height, width = frame.shape[:2]
            if self._shape != (width, height):
                LOGGER.info("Input dimensions changed from %s to %sx%s; resetting crop state", self._shape, width, height)
                self._shape = (width, height)
                self._crop_samples.clear()
                self._auto_crop_rect = None
                self._crop_rect = None
                self._dsa_mask = None
            if self._crop_rect is None and self.auto_crop_enabled:
                self._crop_samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if len(self._crop_samples) < self.crop_sample_frames:
                    LOGGER.debug("Collecting crop sample %s/%s", len(self._crop_samples), self.crop_sample_frames)
                    return None
                self._auto_crop_rect = _detect_aligned_field_crop(self._crop_samples) or _detect_pillarbox_crop(
                    self._crop_samples, width, height
                )
                self._crop_rect = _adjust_auto_crop_square(
                    self._auto_crop_rect,
                    width,
                    height,
                    self.auto_crop_size_offset,
                )
                self._crop_samples.clear()
                LOGGER.info("Auto-crop selected rectangle %s", self._crop_rect)
            elif self._crop_rect is None:
                from PySide6.QtCore import QRect

                self._crop_rect = QRect(0, 0, width, height)
                LOGGER.info("Using full-frame rectangle %s", self._crop_rect)
            enhanced = cv2.cvtColor(crop_frame(frame, self._crop_rect), cv2.COLOR_BGR2GRAY)
            dsa_enabled = "background_subtraction" in self.stages.enabled_stage_order
            if dsa_enabled:
                enhanced = self._apply_live_dsa(enhanced)
            pipeline_stages = self.stages
            if dsa_enabled:
                pipeline_stages = EnhancementStages(
                    instances=tuple(stage for stage in self.stages.instances if stage.key != "background_subtraction")
                )
            enhanced = self.pipeline.process(
                enhanced,
                pipeline_stages,
                self.parameters,
                FrameContext(
                    target_median=128.0,
                    noise_sigma=self.noise_sigma,
                    denoise_batch=self.denoiser.denoise_batch if self.denoiser is not None else None,
                ),
            )
            ok, output = cv2.imencode(".jpg", np.clip(enhanced, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError("Could not encode enhanced frame.")
            LOGGER.debug("Enhanced frame %sx%s into %s JPEG bytes", width, height, len(output))
            return bytes(output)

    def _apply_live_dsa(self, frame: np.ndarray) -> np.ndarray:
        if self._dsa_recording_frame_count is None:
            return frame
        self._dsa_recording_frame_count += 1
        if self._dsa_mask is None and self._dsa_recording_frame_count >= self.dsa_mask_delay_frames:
            self._dsa_mask = frame.copy()
            LOGGER.info("Acquired live DSA mask after %s recording frames", self._dsa_recording_frame_count)
        if self._dsa_mask is None:
            return frame
        return subtract_fluoroscopy_background(frame, self._dsa_mask, self.dsa_darkening_threshold)


class RawFrameRecorder:
    """Writes each contiguous sequence of distinct raw source frames to its own video."""

    IDENTICAL_FRAME_CUTOFF = 3

    def __init__(self, directory: str | Path, fps: float) -> None:
        self.directory = Path(directory)
        self.fps = fps
        self._last_frame: np.ndarray | None = None
        self._identical_frame_count = 0
        self._candidate_frame: np.ndarray | None = None
        self._initial_frame_pending = True
        self._writer: cv2.VideoWriter | None = None
        self._recording_path: Path | None = None
        self.completed_paths: list[Path] = []
        self._device_name = "device"
        self._test_identifier = "test"
        self._phase = "pre"

    def configure_naming(self, device_name: str, test_identifier: str, phase: str) -> Path:
        if phase not in {"pre", "post"}:
            raise ValueError("Recording phase must be 'pre' or 'post'.")
        self._device_name = self._filename_component(device_name, "device")
        self._test_identifier = self._filename_component(test_identifier, "test")
        self._phase = phase
        return self._next_recording_path()

    def record(self, frame: np.ndarray) -> bool:
        if self._last_frame is None:
            self._last_frame = frame.copy()
            return False

        if self._writer is not None:
            if np.array_equal(frame, self._last_frame):
                self._identical_frame_count += 1
                if self._identical_frame_count >= self.IDENTICAL_FRAME_CUTOFF:
                    self.close_recording()
                    return False
                self._writer.write(frame)
                return False
            if frame.shape[:2] != self._last_frame.shape[:2]:
                self.close_recording()
                self._candidate_frame = frame.copy()
                self._last_frame = self._candidate_frame
                return False
            self._writer.write(frame)
            self._last_frame = frame.copy()
            self._identical_frame_count = 1
            return False

        if self._initial_frame_pending:
            self._initial_frame_pending = False
            if np.array_equal(frame, self._last_frame):
                return False
            if frame.shape[:2] == self._last_frame.shape[:2]:
                self._open_recording(self._last_frame)
                assert self._writer is not None
                self._writer.write(self._last_frame)
                self._writer.write(frame)
                self._last_frame = frame.copy()
                return True
            self._candidate_frame = frame.copy()
            self._last_frame = self._candidate_frame
            return False

        if self._candidate_frame is None:
            if np.array_equal(frame, self._last_frame):
                return False
            self._candidate_frame = frame.copy()
            self._last_frame = self._candidate_frame
            return False

        if np.array_equal(frame, self._candidate_frame):
            self._candidate_frame = None
            return False
        if frame.shape[:2] != self._candidate_frame.shape[:2]:
            self._candidate_frame = frame.copy()
            self._last_frame = self._candidate_frame
            return False
        self._open_recording(self._candidate_frame)
        assert self._writer is not None
        self._writer.write(self._candidate_frame)
        self._writer.write(frame)
        self._candidate_frame = None
        self._last_frame = frame.copy()
        return True

    def close_recording(self) -> None:
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        assert self._recording_path is not None
        self.completed_paths.append(self._recording_path)
        LOGGER.info("Saved raw fluoroscopy recording %s", self._recording_path)
        self._recording_path = None

    def reset(self) -> None:
        self.close_recording()
        self._last_frame = None
        self._identical_frame_count = 0
        self._candidate_frame = None
        self._initial_frame_pending = True

    def _open_recording(self, frame: np.ndarray) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._recording_path = self._next_recording_path()
        self._identical_frame_count = 1
        height, width = frame.shape[:2]
        self._writer = cv2.VideoWriter(
            str(self._recording_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps,
            (width, height),
            True,
        )
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise RuntimeError(f"Could not open recording file: {self._recording_path}")
        LOGGER.info("Started raw fluoroscopy recording %s", self._recording_path)

    def _next_recording_path(self) -> Path:
        stem = f"{self._device_name}_{self._test_identifier}_{self._phase}"
        clip_number = 0
        while (path := self.directory / f"{stem}_{clip_number}.avi").exists():
            clip_number += 1
        return path

    @staticmethod
    def _filename_component(value: str, fallback: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9-]+", "_", value.strip()).strip("_")
        return sanitized or fallback


class StreamService:
    def __init__(
        self,
        processor: LiveStreamProcessor,
        max_frame_bytes: int,
        recorder: RawFrameRecorder | None = None,
    ) -> None:
        self.processor = processor
        self.max_frame_bytes = max_frame_bytes
        self.recorder = recorder
        self.recording_enabled = True
        self._condition = Condition()
        self._ingest_lock = Lock()
        self._latest_source: bytes | None = None
        self._latest_frame: bytes | None = None
        self._frame_id = 0
        self._ingested = 0

    def ingest(self, encoded: bytes) -> tuple[int, bool]:
        if not encoded or len(encoded) > self.max_frame_bytes:
            raise ValueError(f"JPEG frame must be between 1 and {self.max_frame_bytes} bytes.")
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Ingest body is not a valid JPEG frame.")
        with self._ingest_lock:
            if self.recorder is not None and self.recording_enabled:
                if self.recorder.record(frame):
                    self.processor.begin_recording()
            enhanced = self.processor.process_frame(frame)
            with self._condition:
                self._ingested += 1
                if enhanced is not None:
                    self._latest_source = encoded
                    self._latest_frame = enhanced
                    self._frame_id += 1
                    self._condition.notify_all()
                LOGGER.debug("Ingested frame=%s egress_ready=%s", self._ingested, enhanced is not None)
                return self._ingested, enhanced is not None

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close_recording()

    def configure_recording(self, device_name: str, test_identifier: str, phase: str) -> Path | None:
        if self.recorder is None:
            return None
        with self._ingest_lock:
            return self.recorder.configure_naming(device_name, test_identifier, phase)

    def set_recording_enabled(self, enabled: bool) -> None:
        with self._ingest_lock:
            self.recording_enabled = enabled
            if not enabled and self.recorder is not None:
                self.recorder.reset()
                self.processor.reset_dsa_recording()

    def next_frame(self, last_frame_id: int, timeout: float = 15.0) -> tuple[int, bytes | None]:
        with self._condition:
            if self._frame_id <= last_frame_id:
                self._condition.wait(timeout)
            return self._frame_id, self._latest_frame

    def health(self) -> dict[str, object]:
        with self._condition:
            return {
                "ingested_frames": self._ingested,
                "egress_frames": self._frame_id,
                "crop_ready": self.processor.crop_ready,
                "saved_recordings": len(self.recorder.completed_paths) if self.recorder is not None else 0,
            }

    def latest_frames(self) -> tuple[int, bytes | None, bytes | None]:
        """Return the most recent matched source and enhanced JPEG frames."""
        with self._condition:
            return self._frame_id, self._latest_source, self._latest_frame


def create_http_server(settings: StreamSettings, service: StreamService) -> ThreadingHTTPServer:
    class StreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                LOGGER.debug("Health check from %s", self.client_address[0])
                self._json(HTTPStatus.OK, service.health())
                return
            if self.path != "/egress.mjpg":
                LOGGER.warning("Rejected GET %s from %s", self.path, self.client_address[0])
                self._json(HTTPStatus.NOT_FOUND, {"error": "Use /health or /egress.mjpg."})
                return
            boundary = b"frame"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            frame_id = 0
            LOGGER.info("MJPEG client connected from %s", self.client_address[0])
            try:
                while True:
                    frame_id, frame = service.next_frame(frame_id)
                    if frame is None:
                        continue
                    self.wfile.write(b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                LOGGER.info("MJPEG client disconnected from %s", self.client_address[0])
                return

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ingest":
                LOGGER.warning("Rejected POST %s from %s", self.path, self.client_address[0])
                self._json(HTTPStatus.NOT_FOUND, {"error": "Use POST /ingest with image/jpeg."})
                return
            content_type = self.headers.get_content_type().lower()
            length = self.headers.get("Content-Length")
            if content_type not in JPEG_CONTENT_TYPES or length is None:
                LOGGER.warning("Rejected ingest from %s with content type %s", self.client_address[0], content_type)
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "POST one image/jpeg frame with Content-Length."})
                return
            try:
                size = int(length)
                if size < 1 or size > service.max_frame_bytes:
                    raise ValueError
                frame_id, ready = service.ingest(self.rfile.read(size))
            except ValueError as exc:
                LOGGER.warning("Rejected invalid ingest from %s: %s", self.client_address[0], exc)
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JPEG frame body."})
                return
            self._json(HTTPStatus.ACCEPTED, {"ingested_frame": frame_id, "egress_ready": ready})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((settings.host, settings.port), StreamHandler)
    LOGGER.info("Created HTTP stream server at %s:%s", *server.server_address[:2])
    return server
