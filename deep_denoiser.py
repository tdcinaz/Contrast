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


def download_ffdnet_weights(destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(FFDNET_WEIGHTS_URL, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != FFDNET_WEIGHTS_SHA256:
            raise RuntimeError(f"FFDNet checkpoint checksum mismatch: {digest}")
        temporary.replace(destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


class FFDNetDenoiser:
    def __init__(self, weights_path: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("FFDNet requires a CUDA-capable NVIDIA GPU, but CUDA is not available.")
        if not weights_path.exists():
            download_ffdnet_weights(weights_path)
        elif hashlib.sha256(weights_path.read_bytes()).hexdigest() != FFDNET_WEIGHTS_SHA256:
            raise RuntimeError(f"Invalid FFDNet checkpoint: {weights_path}")

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

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = self.model(tensor, sigma)
            output = (output.squeeze(1).clamp(0, 1) * 255).to(torch.uint8)
        return list(output.cpu().numpy())