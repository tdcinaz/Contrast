from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from .models import EnhancementParameters

BatchDenoiser = Callable[[list[np.ndarray], float], list[np.ndarray]]
StageToken = tuple[str, tuple[object, ...]]
FrameProcessor = Callable[[np.ndarray, EnhancementParameters, "FrameContext"], np.ndarray]
TokenBuilder = Callable[[EnhancementParameters, str, int], tuple[object, ...]]
LivePredicate = Callable[[EnhancementParameters], bool]


class ExecutionShape(StrEnum):
    FRAME = "frame"
    TEMPORAL = "temporal"
    BATCH = "batch"
    SEQUENCE = "sequence"
    OBSERVER = "observer"
    SOURCE = "source"
    ANALYSIS = "analysis"


@dataclass(frozen=True, slots=True)
class FrameContext:
    target_median: float = 128.0
    noise_sigma: int = 10
    denoise_batch: BatchDenoiser | None = None


@dataclass(frozen=True, slots=True)
class StageDefinition:
    key: str
    display_name: str
    execution_shape: ExecutionShape
    default_seconds_per_frame: float
    conservative_seconds_per_frame: float
    processor: FrameProcessor | None = None
    token_builder: TokenBuilder = lambda _parameters, _backend_id, _noise_sigma: ()
    live_supported: bool | LivePredicate = True
    modifies_frame_data: bool = True

    def cache_token(
        self,
        parameters: EnhancementParameters,
        backend_id: str = "none",
        noise_sigma: int = 10,
    ) -> StageToken:
        return self.key, self.token_builder(parameters, backend_id, noise_sigma)

    def supports_live(self, parameters: EnhancementParameters) -> bool:
        if isinstance(self.live_supported, bool):
            return self.live_supported
        return self.live_supported(parameters)

    def process_frame(
        self,
        frame: np.ndarray,
        parameters: EnhancementParameters,
        context: FrameContext = FrameContext(),
    ) -> np.ndarray:
        if self.processor is None:
            raise TypeError(f"{self.key} is a {self.execution_shape} stage and cannot process one frame independently")
        return self.processor(frame, parameters, context)


class StageRegistry:
    def __init__(self, definitions: Iterable[StageDefinition] = ()) -> None:
        self._definitions: dict[str, StageDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: StageDefinition) -> None:
        if definition.key in self._definitions:
            raise ValueError(f"Pipeline stage is already registered: {definition.key}")
        self._definitions[definition.key] = definition

    def require(self, key: str) -> StageDefinition:
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise ValueError(f"Unknown pipeline stage: {key}") from exc

    def definitions(self, execution_shape: ExecutionShape | None = None) -> tuple[StageDefinition, ...]:
        values = tuple(self._definitions.values())
        if execution_shape is None:
            return values
        return tuple(definition for definition in values if definition.execution_shape == execution_shape)


def _gain(frame: np.ndarray, parameters: EnhancementParameters, context: FrameContext) -> np.ndarray:
    current_median = max(1.0, float(np.median(frame)))
    target_median = context.target_median if parameters.gain_use_auto_target else float(parameters.gain_target_median)
    gain = float(np.clip(target_median / current_median, parameters.gain_min, parameters.gain_max))
    return np.clip(frame.astype(np.float32) * gain, 0, 255)


def _scanline(frame: np.ndarray, parameters: EnhancementParameters, _context: FrameContext) -> np.ndarray:
    corrected = frame.astype(np.float32)
    vertical_smooth = cv2.GaussianBlur(corrected, (1, 9), sigmaX=0, sigmaY=parameters.scanline_sigma_y)
    row_bias = np.median(corrected - vertical_smooth, axis=1)
    row_bias -= np.median(row_bias)
    corrected -= np.clip(row_bias, -parameters.scanline_bias_clip, parameters.scanline_bias_clip)[:, np.newaxis]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _denoise(frame: np.ndarray, _parameters: EnhancementParameters, context: FrameContext) -> np.ndarray:
    if context.denoise_batch is None:
        raise ValueError("Spatial denoising requires an FFDNet backend.")
    source = np.clip(frame, 0, 255).astype(np.uint8)
    result = context.denoise_batch([source], context.noise_sigma)
    if len(result) != 1:
        raise RuntimeError("Denoiser returned an unexpected number of frames.")
    return result[0]


def _local_contrast(frame: np.ndarray, parameters: EnhancementParameters, _context: FrameContext) -> np.ndarray:
    source = np.clip(frame, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=parameters.clahe_clip_limit,
        tileGridSize=(parameters.clahe_tile_size, parameters.clahe_tile_size),
    )
    return clahe.apply(source)


def _image_adjustments(frame: np.ndarray, parameters: EnhancementParameters, _context: FrameContext) -> np.ndarray:
    adjusted = frame.astype(np.float32) * float(parameters.adjustments_contrast_gain)
    adjusted += float(parameters.adjustments_brightness_offset)
    gamma = max(0.1, float(parameters.adjustments_gamma))
    if abs(gamma - 1.0) > 1e-4:
        adjusted = np.power(np.clip(adjusted, 0, 255) / 255.0, 1.0 / gamma).astype(np.float32) * 255.0
    amount = max(0.0, float(parameters.adjustments_sharpen_amount))
    if amount > 1e-4:
        blurred = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=1.0, sigmaY=1.0)
        adjusted = cv2.addWeighted(adjusted, 1.0 + amount, blurred, -amount, 0.0)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _smooth(frame: np.ndarray, parameters: EnhancementParameters, _context: FrameContext) -> np.ndarray:
    source = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(source, (0, 0), sigmaX=parameters.smoothing_sigma_x)


def subtract_fluoroscopy_background(
    frame: np.ndarray,
    background_reference: np.ndarray,
    darkening_threshold: int,
) -> np.ndarray:
    darkening = background_reference.astype(np.int16) - frame.astype(np.int16) - int(darkening_threshold)
    return np.clip(darkening, 0, 255).astype(np.uint8)


def _gain_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    target = "auto" if parameters.gain_use_auto_target else int(parameters.gain_target_median)
    return target, round(float(parameters.gain_min), 4), round(float(parameters.gain_max), 4)


def _roi_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return (
        str(parameters.roi_mode),
        parameters.roi_manual_rect,
        bool(parameters.roi_softening_enabled),
        round(float(parameters.roi_softening_radius_ratio), 4),
        round(float(parameters.roi_softening_threshold), 4),
        bool(parameters.roi_convex_hull_enabled),
        bool(parameters.roi_circle_fit_enabled),
    )


def _scanline_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return round(float(parameters.scanline_bias_clip), 4), round(float(parameters.scanline_sigma_y), 4)


def _denoise_token(_parameters: EnhancementParameters, backend_id: str, noise_sigma: int) -> tuple[object, ...]:
    return backend_id, int(noise_sigma)


def _temporal_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return (round(float(parameters.temporal_motion_sigma), 4),)


def _mottle_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return round(float(parameters.mottle_similarity_sigma), 4), int(parameters.mottle_window_radius)


def _contrast_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return round(float(parameters.clahe_clip_limit), 4), int(parameters.clahe_tile_size)


def _adjustments_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return (
        int(parameters.adjustments_brightness_offset),
        round(float(parameters.adjustments_contrast_gain), 4),
        round(float(parameters.adjustments_sharpen_amount), 4),
        round(float(parameters.adjustments_gamma), 4),
    )


def _smoothing_token(parameters: EnhancementParameters, _backend_id: str, _noise_sigma: int) -> tuple[object, ...]:
    return (round(float(parameters.smoothing_sigma_x), 4),)


BUILTIN_STAGES = StageRegistry(
    (
        StageDefinition("auto_crop", "Auto-crop fluoroscope field", ExecutionShape.SOURCE, 0.0, 0.0, live_supported=True),
        StageDefinition("temporal_alignment", "Temporal alignment (trim onset)", ExecutionShape.SOURCE, 0.0, 0.0, live_supported=False),
        StageDefinition("contrast_gain_alignment", "Align fluoroscope contrast gain", ExecutionShape.SOURCE, 0.0002, 0.0004, live_supported=False),
        StageDefinition("background_subtraction", "Manual background subtraction", ExecutionShape.SOURCE, 0.0002, 0.0004, live_supported=False),
        StageDefinition("brightness_stabilization", "Gain / brightness stabilization", ExecutionShape.SEQUENCE, 0.0020, 0.0030, live_supported=False),
        StageDefinition(
            "roi_extraction",
            "Aneurysm ROI extraction",
            ExecutionShape.SEQUENCE,
            0.0018,
            0.0028,
            modifies_frame_data=False,
            token_builder=_roi_token,
            live_supported=False,
        ),
        StageDefinition("gain_stabilization", "Median gain normalization", ExecutionShape.FRAME, 0.0012, 0.0018, processor=_gain, token_builder=_gain_token),
        StageDefinition("scanline_correction", "Scanline correction", ExecutionShape.FRAME, 0.0025, 0.0035, processor=_scanline, token_builder=_scanline_token),
        StageDefinition("denoise", "Spatial denoising", ExecutionShape.BATCH, 0.0065, 0.0090, processor=_denoise, token_builder=_denoise_token),
        StageDefinition("temporal_filter", "Motion-aware temporal filtering", ExecutionShape.TEMPORAL, 0.0022, 0.0030, token_builder=_temporal_token, live_supported=False),
        StageDefinition("quantum_mottle_filter", "Quantum mottle reduction", ExecutionShape.TEMPORAL, 0.0110, 0.0150, token_builder=_mottle_token, live_supported=False),
        StageDefinition("local_contrast", "Local contrast (CLAHE)", ExecutionShape.FRAME, 0.0028, 0.0038, processor=_local_contrast, token_builder=_contrast_token),
        StageDefinition("image_adjustments", "Image adjustments", ExecutionShape.FRAME, 0.0016, 0.0026, processor=_image_adjustments, token_builder=_adjustments_token),
        StageDefinition("final_smoothing", "Final Gaussian smoothing", ExecutionShape.FRAME, 0.0010, 0.0015, processor=_smooth, token_builder=_smoothing_token),
        StageDefinition(
            "roi_residence_analysis",
            "ROI residence analysis",
            ExecutionShape.ANALYSIS,
            0.0,
            0.0,
            modifies_frame_data=False,
        ),
        StageDefinition(
            "frame_brightness_analysis",
            "Frame brightness analysis",
            ExecutionShape.ANALYSIS,
            0.0,
            0.0,
            modifies_frame_data=False,
        ),
        StageDefinition(
            "temporal_change_heatmap",
            "Temporal pixel-change heatmap",
            ExecutionShape.ANALYSIS,
            0.0,
            0.0,
            modifies_frame_data=False,
            live_supported=False,
        ),
    )
)
