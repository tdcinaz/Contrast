from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np


def _preload_arm_nvidia_libraries() -> None:
    site_packages = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    flags = os.RTLD_GLOBAL | os.RTLD_LAZY
    libraries = [
        site_packages / "nvpl/lib/libnvpl_blas_core.so.0",
        site_packages / "nvpl/lib/libnvpl_blas_lp64_gomp.so.0",
        site_packages / "nvpl/lib/libnvpl_lapack_core.so.0",
        site_packages / "nvpl/lib/libnvpl_lapack_lp64_gomp.so.0",
        site_packages / "nvidia/cu13/lib/libcudss.so.0",
    ]
    for library in libraries:
        if library.exists():
            ctypes.CDLL(str(library), mode=flags)


_preload_arm_nvidia_libraries()

import torch
from torch import nn
from torch.nn import functional as F


FFDNET_WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth"
FFDNET_WEIGHTS_SHA256 = "3c592bc022b4ec609e5e3b03776267c6297eba2714fd3c5f5dae62c12f7ac9c3"
DNCNN_WEIGHTS = {
    15: (
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_15.pth",
        "d1f48a581f42bd932de630a13e0b776ace33f6a24efa5572c112028f632a963f",
    ),
    25: (
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_25.pth",
        "0451a70de9b672ae037270498fbb1c17a1c1c4403785df586ff65df5b858e5b0",
    ),
    50: (
        "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_50.pth",
        "83c11202a88e7b238d08107060c909cb3e692f8177d3470d26d3f1a7b3475a83",
    ),
}


class FFDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(5, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(13):
            layers.extend((nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(64, 4, 3, padding=1))
        self.model = nn.Sequential(*layers)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, image: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_bottom = height % 2
        pad_right = width % 2
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")
        image = F.pixel_unshuffle(image, 2)
        noise_map = sigma.expand(-1, -1, image.shape[-2], image.shape[-1])
        output = self.model(torch.cat((image, noise_map), dim=1))
        return self.pixel_shuffle(output)[..., :height, :width]


class DnCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(15):
            layers.extend((nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(64, 1, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image - self.model(image)


def download_weights(destination: Path, url: str, expected_sha256: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(f"Model checkpoint checksum mismatch: {digest}")
        temporary.replace(destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def ensure_weights(destination: Path, url: str, expected_sha256: str) -> None:
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256:
        return
    download_weights(destination, url, expected_sha256)


def _validate_precision(precision: str) -> None:
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported inference precision: {precision}")


class FFDNetDenoiser:
    def __init__(self, weights_path: Path, precision: str = "fp16") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("FFDNet requires a CUDA-capable NVIDIA GPU, but CUDA is not available.")
        _validate_precision(precision)
        ensure_weights(weights_path, FFDNET_WEIGHTS_URL, FFDNET_WEIGHTS_SHA256)

        self.backend_id = f"ffdnet-native-{precision}"
        self.precision = precision
        self.device = torch.device("cuda", 0)
        self.model = FFDNet()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device, memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)

    def denoise(self, image: np.ndarray, noise_sigma: float) -> np.ndarray:
        return self.denoise_batch([image], noise_sigma)[0]

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]:
        if not images:
            return []
        tensor = torch.from_numpy(np.stack(images, axis=0)).to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        tensor = tensor.unsqueeze(1).div_(255.0).contiguous(memory_format=torch.channels_last)
        sigma = torch.full(
            (len(images), 1, 1, 1),
            noise_sigma / 255.0,
            device=self.device,
        )

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.precision == "fp16",
        ):
            output = self.model(tensor, sigma)
            output = (output.squeeze(1).clamp(0, 1) * 255).to(torch.uint8)
        return list(output.cpu().numpy())


class DnCNNDenoiser:
    def __init__(self, weights_path: Path, noise_level: int, precision: str = "fp16") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("DnCNN requires a CUDA-capable NVIDIA GPU, but CUDA is not available.")
        if noise_level not in DNCNN_WEIGHTS:
            raise ValueError(f"Unsupported DnCNN noise level: {noise_level}")
        _validate_precision(precision)
        weights_url, weights_sha256 = DNCNN_WEIGHTS[noise_level]
        ensure_weights(weights_path, weights_url, weights_sha256)

        self.backend_id = f"dncnn-{noise_level}-native-{precision}"
        self.precision = precision
        self.noise_level = noise_level
        self.device = torch.device("cuda", 0)
        self.model = DnCNN()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device, memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)

    def denoise(self, image: np.ndarray, noise_sigma: float = 0.0) -> np.ndarray:
        return self.denoise_batch([image], noise_sigma)[0]

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float = 0.0) -> list[np.ndarray]:
        if not images:
            return []
        tensor = torch.from_numpy(np.stack(images, axis=0)).to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        tensor = tensor.unsqueeze(1).div_(255.0).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.precision == "fp16",
        ):
            output = self.model(tensor)
            output = (output.squeeze(1).clamp(0, 1) * 255).to(torch.uint8)
        return list(output.cpu().numpy())