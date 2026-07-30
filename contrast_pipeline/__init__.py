"""Frontend-neutral pipeline contracts and execution support."""

from .executor import FramePipelineExecutor
from .models import EnhancementParameters, EnhancementRequest, EnhancementStages, PipelineStage
from .stages import BUILTIN_STAGES, ExecutionShape, FrameContext, StageDefinition, StageRegistry

__all__ = [
    "BUILTIN_STAGES",
    "FramePipelineExecutor",
    "EnhancementParameters",
    "EnhancementRequest",
    "EnhancementStages",
    "ExecutionShape",
    "FrameContext",
    "PipelineStage",
    "StageDefinition",
    "StageRegistry",
]
