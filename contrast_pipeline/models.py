from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnhancementParameters:
    gain_use_auto_target: bool = True
    gain_target_median: int = 128
    gain_min: float = 0.70
    gain_max: float = 1.45
    scanline_bias_clip: float = 6.0
    scanline_sigma_y: float = 2.0
    temporal_motion_sigma: float = 12.0
    mottle_similarity_sigma: float = 12.0
    mottle_window_radius: int = 2
    clahe_clip_limit: float = 1.0
    clahe_tile_size: int = 6
    adjustments_brightness_offset: int = 0
    adjustments_contrast_gain: float = 1.0
    adjustments_sharpen_amount: float = 0.0
    adjustments_gamma: float = 1.0
    smoothing_sigma_x: float = 0.55
    roi_mode: str = "auto"
    roi_manual_circle: tuple[int, int, int] | None = None
    roi_softening_enabled: bool = False
    roi_softening_radius_ratio: float = 0.12
    roi_softening_threshold: float = 0.10
    roi_convex_hull_enabled: bool = True
    roi_circle_fit_enabled: bool = False


@dataclass(frozen=True, slots=True)
class PipelineStage:
    key: str
    enabled: bool
    parameters: EnhancementParameters | None = None
    noise_sigma: int | None = None


@dataclass(frozen=True, slots=True)
class EnhancementStages:
    gain_stabilization: bool = False
    brightness_stabilization: bool = False
    roi_extraction: bool = False
    scanline_correction: bool = False
    denoise: bool = False
    temporal_filter: bool = False
    quantum_mottle_filter: bool = False
    local_contrast: bool = False
    image_adjustments: bool = False
    final_smoothing: bool = False
    stage_order: tuple[str, ...] = (
        "brightness_stabilization",
        "roi_extraction",
        "gain_stabilization",
        "scanline_correction",
        "denoise",
        "temporal_filter",
        "quantum_mottle_filter",
        "local_contrast",
        "image_adjustments",
        "final_smoothing",
    )
    instances: tuple[PipelineStage, ...] = ()

    @property
    def any_enabled(self) -> bool:
        if self.instances:
            return any(stage.enabled for stage in self.instances)
        return any(bool(getattr(self, stage)) for stage in self.stage_order)

    @property
    def enabled_stage_order(self) -> tuple[str, ...]:
        if self.instances:
            return tuple(stage.key for stage in self.instances if stage.enabled)
        return tuple(stage for stage in self.stage_order if bool(getattr(self, stage)))

    def enabled_stage_instances(
        self,
        default_parameters: EnhancementParameters,
    ) -> tuple[PipelineStage, ...]:
        if self.instances:
            return tuple(stage for stage in self.instances if stage.enabled)
        return tuple(
            PipelineStage(key=stage_key, enabled=True, parameters=default_parameters)
            for stage_key in self.enabled_stage_order
        )


@dataclass(frozen=True, slots=True)
class EnhancementRequest:
    generation: int
    mode: str
    model_label: str
    stages: EnhancementStages
    parameters: EnhancementParameters
    noise_sigma: int
    batch_size: int
    precision: str
    auto_crop: bool
    temporal_alignment: bool
    source_pipeline_current: bool
    auto_crop_size_offset: int = 0
    temporal_trim_offset_seconds: float = 0.0
    temporal_end_trim_seconds: float = 0.0
    comparison_sync_offset_seconds: float = 0.0
    background_subtraction_settings: tuple[tuple[bool, int], ...] = ()
    roi_parameters_by_panel: tuple[EnhancementParameters, ...] = ()
    contrast_gain_alignment: bool = False
    histogram_matching: bool = False
    metal_needle_removal: bool = False
