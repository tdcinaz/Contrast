from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.misc import is_dicom
from pydicom.pixels import iter_pixels


@dataclass(frozen=True, slots=True)
class DicomVideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    window_center: float
    window_width: float


def _first_float(value: object, default: float) -> float:
    if isinstance(value, Sequence) and not isinstance(value, str):
        value = value[0] if value else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_dicom_video(path: Path) -> bool:
    if not path.is_file() or not is_dicom(str(path)):
        return False
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["Rows", "Columns", "NumberOfFrames", "PhotometricInterpretation"],
        )
    except (OSError, ValueError):
        return False
    return all(hasattr(dataset, name) for name in ("Rows", "Columns", "PhotometricInterpretation"))


def probe_dicom_video(path: Path) -> DicomVideoInfo:
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    frame_count = int(getattr(dataset, "NumberOfFrames", 1))
    width = int(dataset.Columns)
    height = int(dataset.Rows)
    cine_rate = _first_float(getattr(dataset, "CineRate", None), 0.0)
    frame_time = _first_float(getattr(dataset, "FrameTime", None), 0.0)
    fps = cine_rate if cine_rate > 0.0 else (1000.0 / frame_time if frame_time > 0.0 else 30.0)

    bits_stored = int(getattr(dataset, "BitsStored", 16))
    signed = int(getattr(dataset, "PixelRepresentation", 0)) == 1
    default_min = float(-(1 << (bits_stored - 1)) if signed else 0)
    default_max = float((1 << (bits_stored - (1 if signed else 0))) - 1)
    default_width = default_max - default_min + 1.0
    default_center = (default_min + default_max + 1.0) / 2.0
    window_center = _first_float(getattr(dataset, "WindowCenter", None), default_center)
    window_width = max(1.0, _first_float(getattr(dataset, "WindowWidth", None), default_width))
    return DicomVideoInfo(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        window_center=window_center,
        window_width=window_width,
    )


class DicomVideoCapture:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.info = probe_dicom_video(path)
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        if str(getattr(dataset, "PhotometricInterpretation", "")) not in {"MONOCHROME1", "MONOCHROME2"}:
            raise ValueError(f"Unsupported DICOM photometric interpretation: {dataset.PhotometricInterpretation}")
        self._photometric_interpretation = str(dataset.PhotometricInterpretation)
        self._rescale_slope = _first_float(getattr(dataset, "RescaleSlope", None), 1.0)
        self._rescale_intercept = _first_float(getattr(dataset, "RescaleIntercept", None), 0.0)
        self._position = 0
        self._frames: Iterator[np.ndarray] | None = None
        self._opened = True

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FPS: self.info.fps,
            cv2.CAP_PROP_FRAME_COUNT: self.info.frame_count,
            cv2.CAP_PROP_FRAME_WIDTH: self.info.width,
            cv2.CAP_PROP_FRAME_HEIGHT: self.info.height,
            cv2.CAP_PROP_POS_FRAMES: self._position,
        }
        return float(values.get(property_id, 0.0))

    def set(self, property_id: int, value: float) -> bool:
        if property_id != cv2.CAP_PROP_POS_FRAMES or not self._opened:
            return False
        self._position = max(0, min(int(value), self.info.frame_count))
        self._frames = None
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened or self._position >= self.info.frame_count:
            return False, None
        if self._frames is None:
            self._frames = iter_pixels(
                self.path,
                indices=range(self._position, self.info.frame_count),
                raw=True,
            )
        try:
            pixels = next(self._frames)
        except StopIteration:
            return False, None
        self._position += 1
        gray = self._apply_window(pixels)
        return True, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def release(self) -> None:
        self._frames = None
        self._opened = False

    def _apply_window(self, pixels: np.ndarray) -> np.ndarray:
        values = pixels.astype(np.float32) * self._rescale_slope + self._rescale_intercept
        center = self.info.window_center
        width = self.info.window_width
        if width <= 1.0:
            windowed = np.where(values > center - 0.5, 255.0, 0.0)
        else:
            windowed = ((values - (center - 0.5)) / (width - 1.0) + 0.5) * 255.0
        gray = np.clip(windowed, 0.0, 255.0).astype(np.uint8)
        if self._photometric_interpretation == "MONOCHROME1":
            gray = 255 - gray
        return gray


def open_video_capture(path: Path) -> cv2.VideoCapture | DicomVideoCapture:
    if is_dicom_video(path):
        return DicomVideoCapture(path)
    return cv2.VideoCapture(str(path))