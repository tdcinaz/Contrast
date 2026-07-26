from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


NGC_IMAGE = "nvcr.io/nvidia/pytorch:26.06-py3"
FFDNET_WEIGHTS_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray.pth"
FFDNET_WEIGHTS_SHA256 = "3c592bc022b4ec609e5e3b03776267c6297eba2714fd3c5f5dae62c12f7ac9c3"


def ensure_ffdnet_weights(destination: Path) -> None:
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == FFDNET_WEIGHTS_SHA256:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(FFDNET_WEIGHTS_URL, temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != FFDNET_WEIGHTS_SHA256:
            raise RuntimeError(f"FFDNet checkpoint checksum mismatch: {digest}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class ContainerFFDNetDenoiser:
    backend_id = "ffdnet-ngc-26.06"

    def __init__(self, weights_path: Path, image: str = NGC_IMAGE) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is required for NGC enhancement but was not found.")
        weights_path = weights_path.resolve()
        ensure_ffdnet_weights(weights_path)

        image_check = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if image_check.returncode:
            raise RuntimeError(f"NGC image is not available locally: {image}")

        project_root = Path(__file__).resolve().parent
        self._stderr = tempfile.TemporaryFile(mode="w+t")
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--gpus",
            "all",
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "-v",
            f"{project_root}:/workspace/contrast:ro",
            "-w",
            "/workspace/contrast",
            "--entrypoint",
            "python",
            image,
            "-u",
            "ngc_denoiser_worker.py",
            "--weights",
            f"/workspace/contrast/{weights_path.relative_to(project_root)}",
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._input: np.memmap | None = None
        self._output: np.memmap | None = None
        self._shape: tuple[int, int] | None = None

        try:
            ready = self._read_response(timeout=90.0)
        except Exception:
            self.close()
            raise
        if ready.get("status") != "ready":
            self.close()
            raise RuntimeError(f"NGC worker failed to start: {ready}")
        self._device_name = str(ready["device"])

    @property
    def device_name(self) -> str:
        return f"{self._device_name} via NGC 26.06"

    def _error_output(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().strip()

    def _read_response(self, timeout: float = 30.0) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("NGC worker output is unavailable.")
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            if not selector.select(timeout):
                raise RuntimeError(f"NGC worker timed out. {self._error_output()}")
            line = self._process.stdout.readline()
        finally:
            selector.close()
        if not line:
            raise RuntimeError(f"NGC worker exited unexpectedly. {self._error_output()}")
        response = json.loads(line)
        if response.get("status") == "error":
            raise RuntimeError(str(response.get("message", "NGC worker error")))
        return response

    def _request(self, request: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"NGC worker is not running. {self._error_output()}")
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        return self._read_response(timeout)

    def _ensure_buffers(self, shape: tuple[int, int]) -> None:
        if self._shape == shape:
            return
        self._release_buffers()

        shared_directory = Path("/dev/shm")
        if not os.access(shared_directory, os.W_OK):
            shared_directory = Path(tempfile.gettempdir())
        byte_count = 4 * shape[0] * shape[1]
        input_fd, input_name = tempfile.mkstemp(prefix="contrast-ngc-in-", dir=shared_directory)
        output_fd, output_name = tempfile.mkstemp(prefix="contrast-ngc-out-", dir=shared_directory)
        os.ftruncate(input_fd, byte_count)
        os.ftruncate(output_fd, byte_count)
        os.close(input_fd)
        os.close(output_fd)
        self._input_path = Path(input_name)
        self._output_path = Path(output_name)
        self._input = np.memmap(self._input_path, mode="r+", dtype=np.uint8, shape=(4, *shape))
        self._output = np.memmap(self._output_path, mode="r+", dtype=np.uint8, shape=(4, *shape))
        self._shape = shape
        self._request(
            {
                "command": "setup",
                "input": str(self._input_path),
                "output": str(self._output_path),
                "shape": list(shape),
            }
        )

    def denoise(self, image: np.ndarray, noise_sigma: float) -> np.ndarray:
        return self.denoise_batch([image], noise_sigma)[0]

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]:
        if not images:
            return []
        if len(images) > 4:
            raise ValueError("NGC FFDNet accepts at most four frames per batch.")
        shape = images[0].shape
        if any(image.shape != shape or image.dtype != np.uint8 for image in images):
            raise ValueError("NGC FFDNet requires equally sized uint8 grayscale frames.")
        self._ensure_buffers(shape)
        assert self._input is not None and self._output is not None
        self._input[: len(images)] = images
        self._request(
            {"command": "denoise", "count": len(images), "sigma": noise_sigma},
            timeout=60.0,
        )
        return [self._output[index].copy() for index in range(len(images))]

    def _release_buffers(self) -> None:
        self._input = None
        self._output = None
        for path in (self._input_path, self._output_path):
            if path is not None:
                path.unlink(missing_ok=True)
        self._input_path = None
        self._output_path = None
        self._shape = None

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                self._request({"command": "close"}, timeout=10.0)
            except (BrokenPipeError, RuntimeError):
                process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._release_buffers()
        stderr = getattr(self, "_stderr", None)
        if stderr is not None:
            stderr.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass