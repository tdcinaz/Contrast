from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Lock
from typing import Any

import cv2
import numpy as np

from main import (
    EnhancementParameters,
    EnhancementStages,
    FrameDenoiser,
    PipelineStage,
    _detect_aligned_field_crop,
    _detect_pillarbox_crop,
    correct_scanlines,
    crop_frame,
    enhance_local_contrast,
    apply_image_adjustments,
    segment_dark_contrast,
    smooth_final_frame,
    spatial_bilateral_filter,
    stabilize_frame_gain,
)


LIVE_UNSUPPORTED_STAGES = frozenset(
    {
        "temporal_alignment",
        "brightness_stabilization",
        "roi_extraction",
        "temporal_filter",
        "roi_residence_analysis",
    }
)
JPEG_CONTENT_TYPES = {"image/jpeg", "image/jpg"}


@dataclass(frozen=True, slots=True)
class StreamSettings:
    host: str = "0.0.0.0"
    port: int = 8080
    crop_sample_frames: int = 24
    jpeg_quality: int = 92
    max_frame_bytes: int = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DenoiserSettings:
    mode: str = "classical"
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
        bilateral_diameter=int(_control_value(controls, "bilateralDiameter", 7)),
        bilateral_sigma_color=float(_control_value(controls, "bilateralSigmaColor", 18.0)),
        bilateral_sigma_space=float(_control_value(controls, "bilateralSigmaSpace", 4.0)),
        clahe_clip_limit=float(_control_value(controls, "claheClipLimit", 1.0)),
        clahe_tile_size=int(_control_value(controls, "claheTileSize", 6)),
        adjustments_brightness_offset=int(_control_value(controls, "adjustmentsBrightness", 0)),
        adjustments_contrast_gain=float(_control_value(controls, "adjustmentsContrast", 1.0)),
        adjustments_sharpen_amount=float(_control_value(controls, "adjustmentsSharpen", 0.0)),
        adjustments_gamma=float(_control_value(controls, "adjustmentsGamma", 1.0)),
        smoothing_sigma_x=float(_control_value(controls, "smoothingSigma", 0.55)),
        segmentation_mode=str(_control_value(controls, "segmentationMode", "dark_contrast")),
        segmentation_block_size=int(_control_value(controls, "segmentationBlockSize", 51)),
        segmentation_sensitivity=float(_control_value(controls, "segmentationSensitivity", 7.0)),
        segmentation_level_tolerance=int(_control_value(controls, "segmentationTolerance", 12)),
        segmentation_min_area=int(_control_value(controls, "segmentationMinimumArea", 80)),
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
        if enabled and key in LIVE_UNSUPPORTED_STAGES:
            raise ValueError(f"{key} is not supported for headless live streams.")
        if key == "auto_crop":
            auto_crop_enabled = enabled
            continue
        stage_parameters = _parameters(controls)
        if key == "denoise":
            noise_sigma = int(_control_value(controls, "denoiseStrength", noise_sigma))
            denoiser_settings = DenoiserSettings(
                mode=str(_control_value(controls, "denoiseMode", "classical")),
                precision=str(_control_value(controls, "denoisePrecision", "fp16")),
                batch_size=max(1, int(_control_value(controls, "denoiseBatchSize", 1))),
            )
        instances.append(PipelineStage(key, enabled, stage_parameters, noise_sigma if key == "denoise" else None))
        parameters = stage_parameters
    return settings, EnhancementStages(instances=tuple(instances)), parameters, noise_sigma, auto_crop_enabled, denoiser_settings


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
    ) -> None:
        self.stages = stages
        self.parameters = parameters
        self.noise_sigma = noise_sigma
        self.crop_sample_frames = crop_sample_frames
        self.jpeg_quality = jpeg_quality
        self.auto_crop_enabled = auto_crop_enabled
        self.denoiser = denoiser
        self._crop_samples: list[np.ndarray] = []
        self._crop_rect = None
        self._shape: tuple[int, int] | None = None
        self._lock = Lock()

    @property
    def crop_ready(self) -> bool:
        return self._crop_rect is not None

    def process_jpeg(self, encoded: bytes) -> bytes | None:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Ingest body is not a valid JPEG frame.")
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray) -> bytes | None:
        with self._lock:
            height, width = frame.shape[:2]
            if self._shape != (width, height):
                self._shape = (width, height)
                self._crop_samples.clear()
                self._crop_rect = None
            if self._crop_rect is None and self.auto_crop_enabled:
                self._crop_samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if len(self._crop_samples) < self.crop_sample_frames:
                    return None
                self._crop_rect = _detect_aligned_field_crop(self._crop_samples) or _detect_pillarbox_crop(
                    self._crop_samples, width, height
                )
                self._crop_samples.clear()
            elif self._crop_rect is None:
                from PySide6.QtCore import QRect

                self._crop_rect = QRect(0, 0, width, height)
            enhanced = cv2.cvtColor(crop_frame(frame, self._crop_rect), cv2.COLOR_BGR2GRAY)
            for stage in self.stages.enabled_stage_instances(self.parameters):
                stage_parameters = stage.parameters or self.parameters
                if stage.key == "denoise":
                    if self.denoiser is not None:
                        enhanced = self.denoiser.denoise_batch(
                            [np.clip(enhanced, 0, 255).astype(np.uint8)],
                            stage.noise_sigma or self.noise_sigma,
                        )[0]
                    else:
                        enhanced = spatial_bilateral_filter(
                            np.clip(enhanced, 0, 255).astype(np.uint8),
                            stage_parameters.bilateral_diameter,
                            stage_parameters.bilateral_sigma_color,
                            stage_parameters.bilateral_sigma_space,
                        )
                elif stage.key == "gain_stabilization":
                    target = 128.0 if stage_parameters.gain_use_auto_target else float(stage_parameters.gain_target_median)
                    enhanced = stabilize_frame_gain(enhanced, target, stage_parameters.gain_min, stage_parameters.gain_max)
                elif stage.key == "scanline_correction":
                    enhanced = correct_scanlines(enhanced, stage_parameters.scanline_bias_clip, stage_parameters.scanline_sigma_y)
                elif stage.key == "local_contrast":
                    enhanced = enhance_local_contrast(np.clip(enhanced, 0, 255).astype(np.uint8), stage_parameters.clahe_clip_limit, stage_parameters.clahe_tile_size)
                elif stage.key == "image_adjustments":
                    enhanced = apply_image_adjustments(np.clip(enhanced, 0, 255).astype(np.uint8), stage_parameters.adjustments_brightness_offset, stage_parameters.adjustments_contrast_gain, stage_parameters.adjustments_sharpen_amount, stage_parameters.adjustments_gamma)
                elif stage.key == "final_smoothing":
                    enhanced = smooth_final_frame(np.clip(enhanced, 0, 255).astype(np.uint8), stage_parameters.smoothing_sigma_x)
                elif stage.key == "segmentation" and stage_parameters.segmentation_mode == "dark_contrast":
                    segment_dark_contrast(np.clip(enhanced, 0, 255).astype(np.uint8), stage_parameters.segmentation_block_size, stage_parameters.segmentation_sensitivity, stage_parameters.segmentation_level_tolerance, stage_parameters.segmentation_min_area)
            ok, output = cv2.imencode(".jpg", np.clip(enhanced, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError("Could not encode enhanced frame.")
            return bytes(output)


class StreamService:
    def __init__(self, processor: LiveStreamProcessor, max_frame_bytes: int) -> None:
        self.processor = processor
        self.max_frame_bytes = max_frame_bytes
        self._condition = Condition()
        self._latest_frame: bytes | None = None
        self._frame_id = 0
        self._ingested = 0

    def ingest(self, encoded: bytes) -> tuple[int, bool]:
        if not encoded or len(encoded) > self.max_frame_bytes:
            raise ValueError(f"JPEG frame must be between 1 and {self.max_frame_bytes} bytes.")
        enhanced = self.processor.process_jpeg(encoded)
        with self._condition:
            self._ingested += 1
            if enhanced is not None:
                self._latest_frame = enhanced
                self._frame_id += 1
                self._condition.notify_all()
            return self._ingested, enhanced is not None

    def next_frame(self, last_frame_id: int, timeout: float = 15.0) -> tuple[int, bytes | None]:
        with self._condition:
            if self._frame_id <= last_frame_id:
                self._condition.wait(timeout)
            return self._frame_id, self._latest_frame

    def health(self) -> dict[str, object]:
        with self._condition:
            return {"ingested_frames": self._ingested, "egress_frames": self._frame_id, "crop_ready": self.processor.crop_ready}


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
                self._json(HTTPStatus.OK, service.health())
                return
            if self.path != "/egress.mjpg":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Use /health or /egress.mjpg."})
                return
            boundary = b"frame"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            frame_id = 0
            try:
                while True:
                    frame_id, frame = service.next_frame(frame_id)
                    if frame is None:
                        continue
                    self.wfile.write(b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ingest":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Use POST /ingest with image/jpeg."})
                return
            content_type = self.headers.get_content_type().lower()
            length = self.headers.get("Content-Length")
            if content_type not in JPEG_CONTENT_TYPES or length is None:
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "POST one image/jpeg frame with Content-Length."})
                return
            try:
                size = int(length)
                if size < 1 or size > service.max_frame_bytes:
                    raise ValueError
                frame_id, ready = service.ingest(self.rfile.read(size))
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JPEG frame body."})
                return
            self._json(HTTPStatus.ACCEPTED, {"ingested_frame": frame_id, "egress_ready": ready})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return ThreadingHTTPServer((settings.host, settings.port), StreamHandler)
