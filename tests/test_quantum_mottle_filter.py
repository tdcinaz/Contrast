from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from contrast_pipeline import BUILTIN_STAGES, EnhancementParameters, EnhancementStages, ExecutionShape, PipelineStage
from main import VideoPanel, reduce_quantum_mottle


class QuantumMottleFilterTests(unittest.TestCase):
    def test_five_frame_filter_reduces_uncorrelated_grain(self) -> None:
        generator = np.random.default_rng(42)
        frames = [generator.poisson(100.0, size=(128, 128)).astype(np.uint8) for _ in range(5)]

        filtered = reduce_quantum_mottle(frames, similarity_sigma=12.0)

        self.assertLess(float(np.std(filtered)), float(np.std(frames[2])) * 0.70)

    def test_filter_rejects_neighbors_when_contrast_changes(self) -> None:
        frames = [np.full((64, 64), 150, dtype=np.uint8) for _ in range(5)]
        frames[2][20:44, 20:44] = 50

        filtered = reduce_quantum_mottle(frames, similarity_sigma=12.0)

        self.assertLess(float(np.mean(filtered[20:44, 20:44])), 51.0)
        self.assertEqual(int(np.median(filtered[:16, :16])), 150)

    def test_stage_is_registered_as_temporal_with_parameterized_token(self) -> None:
        definition = BUILTIN_STAGES.require("quantum_mottle_filter")
        first = EnhancementParameters(mottle_similarity_sigma=8.0, mottle_window_radius=2)
        second = EnhancementParameters(mottle_similarity_sigma=12.0, mottle_window_radius=3)

        self.assertEqual(definition.execution_shape, ExecutionShape.TEMPORAL)
        self.assertNotEqual(definition.cache_token(first), definition.cache_token(second))

    def test_video_scheduler_replicates_boundaries_for_centered_window(self) -> None:
        panel = VideoPanel.__new__(VideoPanel)
        panel.path = "synthetic.avi"
        panel.info = SimpleNamespace(fps=15.0)
        panel.trim_frame_count = 5
        panel.target_median = 128.0
        panel.source_gray_frames = [np.full((8, 8), level, dtype=np.uint8) for level in (10, 20, 30, 40, 50)]
        panel.stage_frame_cache = {}
        panel.encoded_frame_cache = {}
        panel.segmentation_mask_cache = {}
        panel.roi_selection_cache = {}
        panel.temporal_change_map_cache = {}
        panel.active_sequence_key = None
        panel.inactive_sequence_key = None
        panel.stage_duration_per_frame = {}
        panel.enhanced_frames = None
        panel.segmentation_masks = None
        panel.stage_roi_selection = None
        panel.display = SimpleNamespace(set_roi=lambda *_args: None)
        parameters = EnhancementParameters(mottle_similarity_sigma=0.1, mottle_window_radius=2)
        stages = EnhancementStages(
            instances=(PipelineStage("quantum_mottle_filter", True, parameters),)
        )

        with patch("main.reduce_quantum_mottle", wraps=reduce_quantum_mottle) as filter_call:
            self.assertTrue(panel.prepare_enhanced_frames(stages=stages, parameters=parameters))

        windows = [
            [int(np.median(frame)) for frame in call.args[0]]
            for call in filter_call.call_args_list
        ]
        windows.sort(key=lambda window: window[len(window) // 2])
        self.assertEqual(
            windows,
            [
                [10, 10, 10, 20, 30],
                [10, 10, 20, 30, 40],
                [10, 20, 30, 40, 50],
                [20, 30, 40, 50, 50],
                [30, 40, 50, 50, 50],
            ],
        )


if __name__ == "__main__":
    unittest.main()