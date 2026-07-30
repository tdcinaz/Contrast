from __future__ import annotations

import numpy as np

from .models import EnhancementParameters, EnhancementStages
from .stages import BUILTIN_STAGES, ExecutionShape, FrameContext, StageRegistry


class FramePipelineExecutor:
    """Executes one frame through stages that do not require a video sequence."""

    def __init__(self, registry: StageRegistry = BUILTIN_STAGES) -> None:
        self.registry = registry

    def process(
        self,
        frame: np.ndarray,
        stages: EnhancementStages,
        default_parameters: EnhancementParameters = EnhancementParameters(),
        context: FrameContext = FrameContext(),
    ) -> np.ndarray:
        output = frame
        for stage in stages.enabled_stage_instances(default_parameters):
            parameters = stage.parameters or default_parameters
            definition = self.registry.require(stage.key)
            if not definition.supports_live(parameters):
                raise ValueError(f"{stage.key} is not supported for single-frame pipelines.")
            if definition.processor is None:
                if definition.execution_shape == ExecutionShape.OBSERVER:
                    continue
                raise ValueError(f"{stage.key} does not provide a single-frame processor.")
            stage_context = FrameContext(
                target_median=context.target_median,
                noise_sigma=stage.noise_sigma if stage.noise_sigma is not None else context.noise_sigma,
                denoise_batch=context.denoise_batch,
            )
            output = definition.process_frame(output, parameters, stage_context)
        return output
