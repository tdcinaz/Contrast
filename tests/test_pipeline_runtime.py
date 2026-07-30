from __future__ import annotations

import unittest

import numpy as np

from contrast_pipeline import (
    BUILTIN_STAGES,
    EnhancementParameters,
    EnhancementStages,
    ExecutionShape,
    FrameContext,
    FramePipelineExecutor,
    PipelineStage,
    StageDefinition,
    StageRegistry,
)


class PipelineRuntimeTests(unittest.TestCase):
    def test_registry_rejects_duplicate_and_unknown_stages(self) -> None:
        definition = StageDefinition("test", "Test", ExecutionShape.FRAME, 0.001, 0.002)
        registry = StageRegistry((definition,))

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(definition)
        with self.assertRaisesRegex(ValueError, "Unknown pipeline stage"):
            registry.require("missing")

    def test_builtin_cache_tokens_include_stage_parameters(self) -> None:
        first = EnhancementParameters(clahe_clip_limit=1.0, clahe_tile_size=6)
        second = EnhancementParameters(clahe_clip_limit=2.5, clahe_tile_size=8)
        definition = BUILTIN_STAGES.require("local_contrast")

        self.assertNotEqual(definition.cache_token(first), definition.cache_token(second))
        self.assertEqual(definition.display_name, "Local contrast (CLAHE)")

    def test_roi_extraction_does_not_modify_frame_data(self) -> None:
        self.assertFalse(BUILTIN_STAGES.require("roi_extraction").modifies_frame_data)
        self.assertTrue(BUILTIN_STAGES.require("local_contrast").modifies_frame_data)

    def test_frame_executor_preserves_stage_order_and_instance_settings(self) -> None:
        calls: list[int] = []

        def denoise_batch(frames: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]:
            calls.append(int(noise_sigma))
            return [np.clip(frame + 5, 0, 255).astype(np.uint8) for frame in frames]

        parameters = EnhancementParameters(gain_use_auto_target=False, gain_target_median=120)
        stages = EnhancementStages(
            instances=(
                PipelineStage("gain_stabilization", True, parameters),
                PipelineStage("denoise", True, parameters, noise_sigma=0),
            )
        )

        result = FramePipelineExecutor().process(
            np.full((8, 8), 100, dtype=np.uint8),
            stages,
            parameters,
            FrameContext(noise_sigma=10, denoise_batch=denoise_batch),
        )

        np.testing.assert_allclose(result, 125, atol=1)
        self.assertEqual(calls, [0])

    def test_frame_executor_rejects_sequence_stage(self) -> None:
        stages = EnhancementStages(
            instances=(PipelineStage("temporal_filter", True, EnhancementParameters()),)
        )

        with self.assertRaisesRegex(ValueError, "not supported"):
            FramePipelineExecutor().process(np.zeros((8, 8), dtype=np.uint8), stages)


if __name__ == "__main__":
    unittest.main()
