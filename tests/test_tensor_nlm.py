from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import deep_denoiser
from container_denoiser import ContainerDenoiser


class TensorNLMTests(unittest.TestCase):
    def test_tensor_kernel_preserves_constant_and_reduces_grain(self) -> None:
        torch = deep_denoiser.torch
        constant = torch.full((1, 1, 24, 24), 120.0)
        constant_result = deep_denoiser.tensor_non_local_means(constant, 5.0, 3, 5, 4)
        generator = torch.Generator().manual_seed(42)
        noisy = (120.0 + torch.randn((1, 1, 32, 32), generator=generator) * 6.0).clamp(0, 255)

        filtered = deep_denoiser.tensor_non_local_means(noisy, 8.0, 3, 5, 4)

        self.assertTrue(torch.equal(constant_result, constant))
        self.assertLess(float(filtered.std()), float(noisy.std()) * 0.40)
        self.assertLess(abs(float(filtered.mean() - noisy.mean())), 0.10)

    def test_zero_strength_returns_an_independent_tensor(self) -> None:
        torch = deep_denoiser.torch
        source = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)

        result = deep_denoiser.tensor_non_local_means(source, 0.0, 3, 5, 4)

        self.assertTrue(torch.equal(result, source))
        self.assertNotEqual(result.data_ptr(), source.data_ptr())

    def test_tensor_nlm_container_launch_does_not_require_weights(self) -> None:
        process = MagicMock()
        process.poll.return_value = 0
        with (
            patch("container_denoiser.shutil.which", return_value="/usr/bin/docker"),
            patch("container_denoiser.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("container_denoiser.subprocess.Popen", return_value=process) as popen,
            patch.object(ContainerDenoiser, "_read_response", return_value={"status": "ready", "device": "GB10"}),
        ):
            denoiser = ContainerDenoiser("tensor-nlm", None, batch_size=4, precision="fp16")

        command = popen.call_args.args[0]
        self.assertIn("tensor-nlm", command)
        self.assertNotIn("--weights", command)
        self.assertEqual(denoiser.backend_id, "tensor-nlm-ngc-26.06-fp16-batch4")
        denoiser.close()

    def test_ffdnet_container_still_requires_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            ContainerDenoiser("ffdnet", None, batch_size=1)


if __name__ == "__main__":
    unittest.main()
