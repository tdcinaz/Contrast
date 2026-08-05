from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


DEFAULT_NLM_BACKEND_ID = "non-local-means-opencv-t7-s21"


class NonLocalMeansDenoiser:
    def __init__(
        self,
        template_window_size: int = 7,
        search_window_size: int = 21,
        max_workers: int = 4,
    ) -> None:
        if template_window_size < 1 or template_window_size % 2 == 0:
            raise ValueError("NLM template window size must be a positive odd number.")
        if search_window_size < template_window_size or search_window_size % 2 == 0:
            raise ValueError("NLM search window size must be odd and at least the template window size.")
        if max_workers < 1:
            raise ValueError("NLM worker count must be positive.")
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size
        self.backend_id = f"non-local-means-opencv-t{template_window_size}-s{search_window_size}"
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nlm-frame")

    @property
    def device_name(self) -> str:
        return "CPU (OpenCV non-local means)"

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]:
        if any(image.ndim != 2 or image.dtype != np.uint8 for image in images):
            raise ValueError("Non-local means requires uint8 grayscale frames.")
        strength = max(0.0, float(noise_sigma))
        if strength == 0.0:
            return [image.copy() for image in images]
        futures = [
            self._executor.submit(
                cv2.fastNlMeansDenoising,
                image,
                None,
                strength,
                self.template_window_size,
                self.search_window_size,
            )
            for image in images
        ]
        return [future.result() for future in futures]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)