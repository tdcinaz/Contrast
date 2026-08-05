from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QSpinBox

from contrast_pipeline import EnhancementStages, PipelineStage
from main import ContrastWindow
from opencv_denoiser import NonLocalMeansDenoiser


class NonLocalMeansTests(unittest.TestCase):
    def test_zero_strength_preserves_input(self) -> None:
        frame = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        denoiser = NonLocalMeansDenoiser()

        result = denoiser.denoise_batch([frame], 0)

        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], frame)
        self.assertIsNot(result[0], frame)

    def test_non_local_means_reduces_grain_without_shifting_mean(self) -> None:
        generator = np.random.default_rng(42)
        frame = np.clip(120 + generator.normal(0, 6, (96, 96)), 0, 255).astype(np.uint8)
        denoiser = NonLocalMeansDenoiser()

        result = denoiser.denoise_batch([frame], 5)[0]

        self.assertEqual(result.dtype, np.uint8)
        self.assertLess(float(np.std(result)), float(np.std(frame)) * 0.45)
        self.assertLess(abs(float(np.mean(result)) - float(np.mean(frame))), 0.25)

    def test_spatial_denoising_drawer_selects_non_local_means_backend(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        drawer = window._add_pipeline_stage("denoise")
        mode = drawer.findChild(QComboBox, "denoiseMode")
        strength_label = drawer.findChild(QLabel, "denoiseStrengthLabel")
        strength = drawer.findChild(QSpinBox, "denoiseStrength")
        batch_size = drawer.findChild(QSpinBox, "denoiseBatchSize")
        precision = drawer.findChild(QComboBox, "denoisePrecision")
        assert mode is not None
        assert strength_label is not None
        assert strength is not None
        assert batch_size is not None
        assert precision is not None

        mode.blockSignals(True)
        mode.setCurrentIndex(mode.findData("non-local-means"))
        mode.blockSignals(False)
        drawer.enable_button.setChecked(True)
        window._sync_active_denoise_controls()

        self.assertEqual(mode.currentText(), "Non-local means (CPU)")
        self.assertEqual(strength_label.text(), "NLM filter strength")
        self.assertTrue(strength.isEnabled())
        self.assertFalse(batch_size.isEnabled())
        self.assertFalse(precision.isEnabled())
        self.assertEqual(window._drawer_control_values(drawer)["denoiseMode"], "non-local-means")

        stages = EnhancementStages(instances=(PipelineStage("denoise", True),))
        denoiser = window._live_denoiser_for(stages)
        self.assertIsInstance(denoiser, NonLocalMeansDenoiser)
        self.assertEqual(window._current_backend_id(EnhancementStages(denoise=True)), denoiser.backend_id)
        app.quit()

    def test_spatial_denoising_drawer_selects_ngc_tensor_backend(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = ContrastWindow()
        self.addCleanup(window.close)
        drawer = window._add_pipeline_stage("denoise")
        mode = drawer.findChild(QComboBox, "denoiseMode")
        strength_label = drawer.findChild(QLabel, "denoiseStrengthLabel")
        batch_size = drawer.findChild(QSpinBox, "denoiseBatchSize")
        precision = drawer.findChild(QComboBox, "denoisePrecision")
        assert mode is not None
        assert strength_label is not None
        assert batch_size is not None
        assert precision is not None
        mode.blockSignals(True)
        mode.setCurrentIndex(mode.findData("tensor-nlm-ngc"))
        mode.blockSignals(False)
        drawer.enable_button.setChecked(True)
        window._sync_active_denoise_controls()

        self.assertEqual(mode.currentText(), "NGC Tensor NLM (GPU)")
        self.assertEqual(strength_label.text(), "NLM filter strength")
        self.assertTrue(batch_size.isEnabled())
        self.assertTrue(precision.isEnabled())
        denoiser = MagicMock()
        denoiser.backend_id = "tensor-nlm-ngc-test"
        with patch("container_denoiser.ContainerDenoiser", return_value=denoiser) as container:
            stages = EnhancementStages(instances=(PipelineStage("denoise", True),))
            self.assertIs(window._live_denoiser_for(stages), denoiser)

        container.assert_called_once_with("tensor-nlm", None, batch_size.value(), precision.currentData())
        app.quit()


if __name__ == "__main__":
    unittest.main()