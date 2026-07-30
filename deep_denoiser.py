from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import sys
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("contrast.denoiser.native")


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


def download_weights(destination: Path, url: str, expected_sha256: str) -> str:
    LOGGER.info("Downloading FFDNet weights to %s", destination)
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
        LOGGER.debug("Validated existing FFDNet weights at %s", destination)
        return
    download_weights(destination, url, expected_sha256)


def _validate_precision(precision: str) -> None:
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported inference precision: {precision}")


@dataclass(slots=True)
class _CudaGraphState:
    graph: torch.cuda.CUDAGraph
    image: torch.Tensor
    sigma: torch.Tensor | None
    output: torch.Tensor


class _CudaGraphRunner:
    def __init__(self, model: nn.Module, precision: str, uses_sigma: bool) -> None:
        self.model = model
        self.precision = precision
        self.uses_sigma = uses_sigma
        self.enabled = os.environ.get("CONTRAST_CUDA_GRAPHS", "1") != "0"
        self._graphs: OrderedDict[tuple[int, ...], _CudaGraphState] = OrderedDict()
        self._graph_limit = 8

    def _forward(self, image: torch.Tensor, sigma: torch.Tensor | None) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.precision == "fp16",
        ):
            if self.uses_sigma:
                assert sigma is not None
                return self.model(image, sigma)
            return self.model(image)

    def _capture(self, image: torch.Tensor, sigma: torch.Tensor | None) -> _CudaGraphState:
        static_image = torch.empty_like(image)
        static_image.copy_(image)
        static_sigma = torch.empty_like(sigma) if sigma is not None else None
        if static_sigma is not None and sigma is not None:
            static_sigma.copy_(sigma)

        current_stream = torch.cuda.current_stream()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                self._forward(static_image, static_sigma)
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._forward(static_image, static_sigma)
        return _CudaGraphState(graph, static_image, static_sigma, output)

    def run(self, image: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        key = tuple(image.shape)
        if not self.enabled:
            return self._forward(image, sigma)
        state = self._graphs.get(key)
        if state is None:
            LOGGER.debug("Capturing CUDA graph for input shape %s", key)
            state = self._capture(image, sigma)
            if len(self._graphs) >= self._graph_limit:
                self._graphs.popitem(last=False)
            self._graphs[key] = state
        else:
            self._graphs.move_to_end(key)
        state.image.copy_(image)
        if state.sigma is not None and sigma is not None:
            state.sigma.copy_(sigma)
        state.graph.replay()
        return state.output


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
        self._runner = _CudaGraphRunner(self.model, precision, uses_sigma=True)
        LOGGER.info("Initialized native FFDNet denoiser on %s with precision=%s", self.device_name, precision)

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)

    def denoise(self, image: np.ndarray, noise_sigma: float) -> np.ndarray:
        return self.denoise_batch([image], noise_sigma)[0]

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]:
        if not images:
            return []
        return self.denoise_array(np.stack(images, axis=0), noise_sigma)

    def denoise_array(self, images: np.ndarray, noise_sigma: float) -> list[np.ndarray]:
        if images.ndim != 3 or images.dtype != np.uint8:
            raise ValueError("FFDNet requires a uint8 grayscale frame batch.")
        LOGGER.debug("Denoising native batch count=%s shape=%s sigma=%s", len(images), images.shape[1:], noise_sigma)
        tensor = torch.from_numpy(images).to(
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

        output = self._runner.run(tensor, sigma)
        output = (output.squeeze(1).clamp(0, 1) * 255).to(torch.uint8)
        return list(output.cpu().numpy())