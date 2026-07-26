from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QTabWidget,
    QToolButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS = {
    "Pre-deployment": ROOT / "PPI150_PreDeployment_Contrast.mov",
    "Post-deployment": ROOT / "PPI150_PostDeployment_Contrast.mov",
}


class FrameDenoiser(Protocol):
    backend_id: str

    @property
    def device_name(self) -> str: ...

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]: ...


@dataclass(slots=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass(frozen=True, slots=True)
class EnhancementStages:
    gain_stabilization: bool = False
    scanline_correction: bool = False
    denoise: bool = False
    temporal_filter: bool = False
    local_contrast: bool = False
    final_smoothing: bool = False
    stage_order: tuple[str, ...] = (
        "gain_stabilization",
        "scanline_correction",
        "denoise",
        "temporal_filter",
        "local_contrast",
        "final_smoothing",
    )

    @property
    def any_enabled(self) -> bool:
        return any(
            (
                self.gain_stabilization,
                self.scanline_correction,
                self.denoise,
                self.temporal_filter,
                self.local_contrast,
                self.final_smoothing,
            )
        )

    @property
    def enabled_stage_order(self) -> tuple[str, ...]:
        return tuple(stage for stage in self.stage_order if bool(getattr(self, stage, False)))


@dataclass(frozen=True, slots=True)
class EnhancementParameters:
    gain_use_auto_target: bool = True
    gain_target_median: int = 128
    gain_min: float = 0.70
    gain_max: float = 1.45
    scanline_bias_clip: float = 6.0
    scanline_sigma_y: float = 2.0
    bilateral_diameter: int = 7
    bilateral_sigma_color: float = 18.0
    bilateral_sigma_space: float = 4.0
    temporal_motion_sigma: float = 12.0
    clahe_clip_limit: float = 1.0
    clahe_tile_size: int = 6
    smoothing_sigma_x: float = 0.55


@dataclass(slots=True)
class AnalysisResult:
    label: str
    path: Path
    fps: float
    roi: QRect
    time: np.ndarray
    mean_intensity: np.ndarray
    reference_intensity: np.ndarray
    contrast_signal: np.ndarray
    normalized_signal: np.ndarray
    gain_corrected: bool
    threshold_fraction: float
    threshold_value: float
    arrival_time: float | None
    peak_time: float | None
    clear_time: float | None
    residence_time: float | None
    peak_signal: float
    auc: float


def format_seconds(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.2f} s"


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)


def baseline_sample_count(fps: float, sample_count: int) -> int:
    return max(1, min(round(fps * 2), sample_count // 5 or 1))


def smooth_temporal_signal(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 3:
        return values.astype(float, copy=True)

    median_window = min(len(values) if len(values) % 2 else len(values) - 1, max(3, round(fps * 0.1) | 1))
    radius = median_window // 2
    padded = np.pad(values.astype(float), radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, median_window)
    despiked = np.median(windows, axis=1)

    gaussian_window = min(len(values) if len(values) % 2 else len(values) - 1, max(3, round(fps * 0.4) | 1))
    sigma = max(0.8, fps * 0.05)
    return cv2.GaussianBlur(
        despiked.reshape(-1, 1),
        (1, gaussian_window),
        sigmaX=0,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    ).ravel()


def detect_vertical_bar_crop(path: Path, info: VideoInfo) -> QRect:
    # Estimate content bounds from sampled columns to remove pillarbox bars.
    full_frame = QRect(0, 0, info.width, info.height)
    if info.width <= 0 or info.height <= 0:
        return full_frame

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return full_frame

    try:
        sample_count = min(24, max(6, info.frame_count if info.frame_count > 0 else 6))
        frame_indexes = np.unique(np.linspace(0, max(0, info.frame_count - 1), num=sample_count, dtype=int))
        column_profiles: list[np.ndarray] = []
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            column_profiles.append(np.percentile(gray, 90, axis=0).astype(np.float32))

        if len(column_profiles) < 3:
            return full_frame

        profile = np.median(np.stack(column_profiles, axis=0), axis=0)
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
        smooth = np.convolve(profile, kernel / np.sum(kernel), mode="same")

        center_start = info.width // 4
        center_end = (info.width * 3) // 4
        center_slice = smooth[center_start:center_end]
        center_level = float(np.median(center_slice)) if center_slice.size else float(np.median(smooth))

        edge_width = max(1, info.width // 12)
        edge_left = float(np.median(smooth[:edge_width]))
        edge_right = float(np.median(smooth[-edge_width:]))
        edge_level = min(edge_left, edge_right)

        threshold = max(3.0, edge_level + (center_level - edge_level) * 0.35)
        content = smooth > threshold
        if not np.any(content):
            return full_frame

        left = int(np.argmax(content))
        right = int(info.width - 1 - np.argmax(content[::-1]))
        left = max(0, left - 2)
        right = min(info.width - 1, right + 2)
        cropped_width = right - left + 1
        left_margin = left
        right_margin = info.width - right - 1

        # Only crop when bars are clearly present on both sides and crop is reasonable.
        if cropped_width < int(info.width * 0.45):
            return full_frame
        if left_margin < 2 or right_margin < 2:
            return full_frame

        return QRect(left, 0, cropped_width, info.height)
    finally:
        capture.release()


def crop_frame(frame: np.ndarray, crop_rect: QRect) -> np.ndarray:
    x = max(0, crop_rect.x())
    y = max(0, crop_rect.y())
    width = max(1, crop_rect.width())
    height = max(1, crop_rect.height())
    max_height, max_width = frame.shape[:2]
    x2 = min(max_width, x + width)
    y2 = min(max_height, y + height)
    if x >= x2 or y >= y2:
        return frame
    return frame[y:y2, x:x2]


def estimate_video_median(path: Path, crop_rect: QRect, frame_count: int) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        medians: list[float] = []
        sample_indexes = np.unique(np.linspace(0, max(0, frame_count - 1), num=min(24, max(1, frame_count)), dtype=int))
        for frame_index in sample_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY)
                medians.append(float(np.median(gray)))
        return float(np.median(medians)) if medians else 128.0
    finally:
        capture.release()


def stabilize_frame_gain(gray: np.ndarray, target_median: float, min_gain: float, max_gain: float) -> np.ndarray:
    current_median = max(1.0, float(np.median(gray)))
    gain = float(np.clip(target_median / current_median, min_gain, max_gain))
    return np.clip(gray.astype(np.float32) * gain, 0, 255)


def correct_scanlines(gray: np.ndarray, bias_clip: float, sigma_y: float) -> np.ndarray:
    corrected = gray.astype(np.float32)
    vertical_smooth = cv2.GaussianBlur(corrected, (1, 9), sigmaX=0, sigmaY=sigma_y)
    row_bias = np.median(corrected - vertical_smooth, axis=1)
    row_bias -= np.median(row_bias)
    corrected -= np.clip(row_bias, -bias_clip, bias_clip)[:, np.newaxis]
    return np.clip(corrected, 0, 255).astype(np.uint8)


def spatial_bilateral_filter(gray: np.ndarray, diameter: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(
        gray,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def motion_aware_temporal_filter(
    previous: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    motion_sigma: float,
) -> np.ndarray:
    current_float = current.astype(np.float32)
    previous_float = previous.astype(np.float32)
    following_float = following.astype(np.float32)
    previous_weight = np.exp(-np.abs(previous_float - current_float) / motion_sigma)
    following_weight = np.exp(-np.abs(following_float - current_float) / motion_sigma)
    neighbor_sum = previous_float * previous_weight + following_float * following_weight
    neighbor_weight = previous_weight + following_weight
    temporal = (current_float + neighbor_sum) / (1.0 + neighbor_weight)
    return np.clip(temporal, 0, 255).astype(np.uint8)


def enhance_local_contrast(gray: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def smooth_final_frame(gray: np.ndarray, sigma_x: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_x)


def reference_mean(gray: np.ndarray, roi: QRect) -> float:
    frame_height, frame_width = gray.shape
    pad_x = max(30, roi.width() // 2)
    pad_y = max(30, roi.height() // 2)
    left = max(0, roi.left() - pad_x)
    right = min(frame_width, roi.right() + pad_x + 1)
    top = max(0, roi.top() - pad_y)
    bottom = min(frame_height, roi.bottom() + pad_y + 1)

    reference = gray[top:bottom, left:right].copy()
    roi_left = max(0, roi.left() - left)
    roi_right = min(reference.shape[1], roi.right() - left + 1)
    roi_top = max(0, roi.top() - top)
    roi_bottom = min(reference.shape[0], roi.bottom() - top + 1)
    mask = np.ones(reference.shape, dtype=bool)
    mask[roi_top:roi_bottom, roi_left:roi_right] = False
    pixels = reference[mask]

    if pixels.size < 200:
        mask = np.ones(gray.shape, dtype=bool)
        mask[roi.y() : roi.y() + roi.height(), roi.x() : roi.x() + roi.width()] = False
        pixels = gray[mask]
    return float(np.median(pixels)) if pixels.size else float(np.median(gray))


class VideoDisplay(QLabel):
    roiChanged = Signal(QRect)

    def __init__(self, title: str, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.roi_color = color
        self.frame_size = (0, 0)
        self._left_pixmap = QPixmap()
        self._right_pixmap = QPixmap()
        self._comparison_enabled = True
        self._roi: QRect | None = None
        self._drag_origin: QPoint | None = None
        self._display_rect = QRect()
        self._right_display_rect = QRect()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(620, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0e1116; border: 1px solid #253044; border-radius: 8px;")

    def _to_pixmap(self, frame: np.ndarray) -> QPixmap:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def set_frames(self, original_frame: np.ndarray, enhanced_frame: np.ndarray | None = None) -> None:
        self._left_pixmap = self._to_pixmap(original_frame)
        self.frame_size = (original_frame.shape[1], original_frame.shape[0])
        if enhanced_frame is None:
            self._right_pixmap = self._to_pixmap(original_frame)
        else:
            self._right_pixmap = self._to_pixmap(enhanced_frame)
        self.update()

    def set_frame(self, frame: np.ndarray) -> None:
        # Backward-compatible single-frame entry point.
        self.set_frames(frame, frame)

    def set_comparison_enabled(self, enabled: bool) -> None:
        self._comparison_enabled = enabled
        self.update()

    def roi(self) -> QRect | None:
        return QRect(self._roi) if self._roi and self._roi.isValid() else None

    def clear_roi(self) -> None:
        self._roi = None
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._left_pixmap.isNull():
            if self._comparison_enabled:
                gap = 10
                slot_width = max(1, (self.width() - gap) // 2)
                left_slot = QRect(0, 0, slot_width, self.height())
                right_slot = QRect(slot_width + gap, 0, self.width() - (slot_width + gap), self.height())

                left_scaled = self._left_pixmap.scaled(
                    left_slot.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                right_scaled = self._right_pixmap.scaled(
                    right_slot.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

                left_x = left_slot.left() + (left_slot.width() - left_scaled.width()) // 2
                left_y = left_slot.top() + (left_slot.height() - left_scaled.height()) // 2
                right_x = right_slot.left() + (right_slot.width() - right_scaled.width()) // 2
                right_y = right_slot.top() + (right_slot.height() - right_scaled.height()) // 2

                self._display_rect = QRect(left_x, left_y, left_scaled.width(), left_scaled.height())
                self._right_display_rect = QRect(right_x, right_y, right_scaled.width(), right_scaled.height())
                painter.drawPixmap(self._display_rect, left_scaled)
                painter.drawPixmap(self._right_display_rect, right_scaled)

                painter.setPen(QColor("#64748b"))
                painter.drawText(left_slot.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "Original")
                painter.drawText(right_slot.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "Enhanced")
            else:
                full_slot = QRect(0, 0, self.width(), self.height())
                scaled = self._right_pixmap.scaled(
                    full_slot.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                draw_x = full_slot.left() + (full_slot.width() - scaled.width()) // 2
                draw_y = full_slot.top() + (full_slot.height() - scaled.height()) // 2
                self._display_rect = QRect(draw_x, draw_y, scaled.width(), scaled.height())
                self._right_display_rect = QRect()
                painter.drawPixmap(self._display_rect, scaled)
                painter.setPen(QColor("#64748b"))
                painter.drawText(full_slot.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "Video")
        else:
            self._display_rect = QRect()
            self._right_display_rect = QRect()
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame loaded")

        if self._roi and self._roi.isValid() and self.frame_size != (0, 0):
            display_roi = self._frame_to_display_rect(self._roi)
            pen = QPen(self.roi_color, 2)
            painter.setPen(pen)
            painter.setBrush(QColor(self.roi_color.red(), self.roi_color.green(), self.roi_color.blue(), 35))
            painter.drawRect(display_roi)

            painter.setPen(QColor("#f8fafc"))
            painter.drawText(display_roi.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop, "ROI")

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton or self.frame_size == (0, 0):
            return
        point = event.position().toPoint()
        if not self._display_rect.contains(point):
            return
        self._drag_origin = point
        frame_point = self._display_to_frame_point(point)
        self._roi = QRect(frame_point, frame_point)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._drag_origin is None or self.frame_size == (0, 0):
            return
        point = event.position().toPoint()
        point.setX(max(self._display_rect.left(), min(point.x(), self._display_rect.right())))
        point.setY(max(self._display_rect.top(), min(point.y(), self._display_rect.bottom())))
        start = self._display_to_frame_point(self._drag_origin)
        end = self._display_to_frame_point(point)
        self._roi = QRect(start, end).normalized()
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.button() != Qt.MouseButton.LeftButton or self._drag_origin is None:
            return
        self._drag_origin = None
        if self._roi and self._roi.width() >= 4 and self._roi.height() >= 4:
            self.roiChanged.emit(QRect(self._roi))
        else:
            self._roi = None
        self.update()

    def _display_to_frame_point(self, point: QPoint) -> QPoint:
        width, height = self.frame_size
        x_fraction = (point.x() - self._display_rect.left()) / max(1, self._display_rect.width())
        y_fraction = (point.y() - self._display_rect.top()) / max(1, self._display_rect.height())
        x = round(max(0.0, min(1.0, x_fraction)) * (width - 1))
        y = round(max(0.0, min(1.0, y_fraction)) * (height - 1))
        return QPoint(x, y)

    def _frame_to_display_rect(self, rect: QRect) -> QRect:
        width, height = self.frame_size
        x_scale = self._display_rect.width() / max(1, width)
        y_scale = self._display_rect.height() / max(1, height)
        return QRect(
            round(self._display_rect.left() + rect.left() * x_scale),
            round(self._display_rect.top() + rect.top() * y_scale),
            round(rect.width() * x_scale),
            round(rect.height() * y_scale),
        )


class VideoPanel(QFrame):
    roiChanged = Signal()

    def __init__(self, label: str, color: QColor, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self.color = color
        self.path = path
        self.info = probe_video(path)
        self.crop_rect = detect_vertical_bar_crop(path, self.info)
        self.capture = cv2.VideoCapture(str(path))
        self.current_frame: np.ndarray | None = None
        self.current_frame_index = -1
        self.enhance_display = False
        self.comparison_display = True
        self.target_median = estimate_video_median(path, self.crop_rect, self.info.frame_count)
        self.enhanced_frames: list[np.ndarray] | None = None
        self.enhancement_signature: tuple[str, int, EnhancementStages, EnhancementParameters] | None = None

        self.display = VideoDisplay(label, color)
        self.display.set_comparison_enabled(self.comparison_display)
        self.display.roiChanged.connect(self.roiChanged.emit)

        self.title_label = QLabel(label)
        self.title_label.setObjectName("panelTitle")
        self.path_label = QLabel(path.name)
        self.path_label.setObjectName("subtleLabel")
        self.meta_label = QLabel(self._metadata_text())
        self.meta_label.setObjectName("subtleLabel")

        header = QHBoxLayout()
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.meta_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self.path_label)
        layout.addWidget(self.display, 1)

        self.setObjectName("videoPanel")
        self.seek(0)

    def prepare_enhanced_frames(
        self,
        denoiser: FrameDenoiser | None = None,
        noise_sigma: int = 10,
        batch_size: int = 4,
        stages: EnhancementStages = EnhancementStages(),
        parameters: EnhancementParameters = EnhancementParameters(),
        progress_callback: Callable[[int, int], bool] | None = None,
    ) -> bool:
        backend_id = denoiser.backend_id if denoiser is not None else "classical"
        signature = (backend_id, noise_sigma if stages.denoise else 0, stages, parameters)
        if (
            self.enhanced_frames is not None
            and len(self.enhanced_frames) == self.info.frame_count
            and self.enhancement_signature == signature
        ):
            return True

        capture = cv2.VideoCapture(str(self.path))
        try:
            stage_frames: list[np.ndarray] = []
            for frame_index in range(self.info.frame_count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not precompute enhancement for video: {self.path}")
                processed = cv2.cvtColor(crop_frame(frame, self.crop_rect), cv2.COLOR_BGR2GRAY)
                stage_frames.append(processed)

            for stage_key in stages.enabled_stage_order:
                if stage_key == "gain_stabilization":
                    target_median = self.target_median if parameters.gain_use_auto_target else float(parameters.gain_target_median)
                    stage_frames = [
                        np.clip(stabilize_frame_gain(frame, target_median, parameters.gain_min, parameters.gain_max), 0, 255)
                        for frame in stage_frames
                    ]
                    continue

                if stage_key == "scanline_correction":
                    stage_frames = [
                        correct_scanlines(frame, parameters.scanline_bias_clip, parameters.scanline_sigma_y)
                        for frame in stage_frames
                    ]
                    continue

                if stage_key == "denoise":
                    denoise_input = [np.clip(frame, 0, 255).astype(np.uint8) for frame in stage_frames]
                    if denoiser is None:
                        stage_frames = [
                            spatial_bilateral_filter(
                                frame,
                                parameters.bilateral_diameter,
                                parameters.bilateral_sigma_color,
                                parameters.bilateral_sigma_space,
                            )
                            for frame in denoise_input
                        ]
                        continue

                    denoised_frames: list[np.ndarray] = []
                    for batch_start in range(0, len(denoise_input), batch_size):
                        batch = denoise_input[batch_start : batch_start + batch_size]
                        denoised_frames.extend(denoiser.denoise_batch(batch, noise_sigma))
                    stage_frames = denoised_frames
                    continue

                if stage_key == "temporal_filter":
                    temporal_frames: list[np.ndarray] = []
                    frame_count = len(stage_frames)
                    for index, current in enumerate(stage_frames):
                        previous = stage_frames[index - 1] if index > 0 else current
                        following = stage_frames[index + 1] if index + 1 < frame_count else current
                        temporal_frames.append(
                            motion_aware_temporal_filter(previous, current, following, parameters.temporal_motion_sigma)
                        )
                    stage_frames = temporal_frames
                    continue

                if stage_key == "local_contrast":
                    stage_frames = [
                        enhance_local_contrast(
                            np.clip(frame, 0, 255).astype(np.uint8),
                            parameters.clahe_clip_limit,
                            parameters.clahe_tile_size,
                        )
                        for frame in stage_frames
                    ]
                    continue

                if stage_key == "final_smoothing":
                    stage_frames = [
                        smooth_final_frame(np.clip(frame, 0, 255).astype(np.uint8), parameters.smoothing_sigma_x)
                        for frame in stage_frames
                    ]

            enhanced_frames: list[np.ndarray] = []
            for index, enhanced in enumerate(stage_frames, start=1):
                if enhanced.dtype != np.uint8:
                    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
                encoded_ok, encoded = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not encoded_ok:
                    raise RuntimeError(f"Could not cache enhanced video frame: {self.path}")
                enhanced_frames.append(encoded)
                if progress_callback is not None and not progress_callback(index, self.info.frame_count):
                    return False

            self.enhanced_frames = enhanced_frames
            self.enhancement_signature = signature
            return True
        finally:
            capture.release()

    def _metadata_text(self) -> str:
        crop_width = self.crop_rect.width()
        crop_height = self.crop_rect.height()
        if crop_width != self.info.width or crop_height != self.info.height:
            return f"{crop_width}x{crop_height} (auto-cropped) | {self.info.fps:.1f} fps | {self.info.duration:.1f} s"
        return f"{self.info.width}x{self.info.height} | {self.info.fps:.1f} fps | {self.info.duration:.1f} s"

    def _display_frame(self, frame: np.ndarray, apply_enhancement: bool | None = None) -> None:
        self.current_frame = frame
        if apply_enhancement is None:
            apply_enhancement = self.enhance_display
        can_enhance = self.enhanced_frames is not None and 0 <= self.current_frame_index < len(self.enhanced_frames)
        enhanced_frame = frame
        if apply_enhancement and can_enhance:
            enhanced = cv2.imdecode(self.enhanced_frames[self.current_frame_index], cv2.IMREAD_GRAYSCALE)
            if enhanced is not None:
                enhanced_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        self.display.set_frames(frame, enhanced_frame)

    def read_next(self, playback: bool = False) -> bool:
        if self.current_frame_index >= self.info.frame_count - 1:
            return False
        ok, frame = self.capture.read()
        if ok:
            self.current_frame_index += 1
            self._display_frame(crop_frame(frame, self.crop_rect), apply_enhancement=self.enhance_display)
        return ok

    def seek(self, frame_index: int) -> bool:
        frame_index = max(0, min(frame_index, self.info.frame_count - 1))
        if frame_index == self.current_frame_index and self.current_frame is not None:
            self._display_frame(self.current_frame)
            return True
        if frame_index == self.current_frame_index + 1:
            return self.read_next()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if ok:
            self.current_frame_index = frame_index
            self._display_frame(crop_frame(frame, self.crop_rect))
        return ok

    def roi(self) -> QRect | None:
        return self.display.roi()

    def set_video(self, path: Path) -> None:
        self.capture.release()
        self.path = path
        self.info = probe_video(path)
        self.crop_rect = detect_vertical_bar_crop(path, self.info)
        self.capture = cv2.VideoCapture(str(path))
        self.current_frame_index = -1
        self.target_median = estimate_video_median(path, self.crop_rect, self.info.frame_count)
        self.enhanced_frames = None
        self.enhancement_signature = None
        self.path_label.setText(path.name)
        self.meta_label.setText(self._metadata_text())
        self.display.clear_roi()
        self.display.set_comparison_enabled(self.comparison_display)
        self.seek(0)

    def set_enhancement(self, enabled: bool, frame_index: int) -> None:
        self.enhance_display = enabled
        self.seek(frame_index)

    def set_comparison(self, enabled: bool, frame_index: int) -> None:
        self.comparison_display = enabled
        self.display.set_comparison_enabled(enabled)
        self.seek(frame_index)

    def clear_enhancement_cache(self) -> None:
        self.enhanced_frames = None
        self.enhancement_signature = None

    def close(self) -> None:
        self.capture.release()
        super().close()


class MetricCard(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = QLabel(title)
        self.title.setObjectName("metricTitle")
        self.value = QLabel("--")
        self.value.setObjectName("metricValue")
        self.detail = QLabel("")
        self.detail.setObjectName("metricDetail")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)
        self.setObjectName("metricCard")

    def set_metric(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class LoadingOverlay(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("loadingOverlay")
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.panel = QFrame(self)
        self.panel.setObjectName("loadingOverlayPanel")

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(18, 16, 18, 16)
        panel_layout.setSpacing(10)

        self.message_label = QLabel("Preparing enhanced video...")
        self.message_label.setObjectName("loadingOverlayLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        panel_layout.addWidget(self.message_label)
        panel_layout.addWidget(self.progress_bar)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self.panel, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        self.hide()

    def begin(self, message: str, maximum: int) -> None:
        self.message_label.setText(message)
        self.progress_bar.setRange(0, max(1, maximum))
        self.progress_bar.setValue(0)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        QApplication.processEvents()

    def set_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)
        QApplication.processEvents()

    def finish(self) -> None:
        self.hide()


class StageDrawer(QFrame):
    enabledChanged = Signal(int)
    moveRequested = Signal(int)

    def __init__(self, stage_key: str, title: str, stage_index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stageDrawer")
        self.stage_key = stage_key
        self.stage_title = title

        self.enable_check = QCheckBox()
        self.expand_button = QToolButton()
        self.expand_button.setText("Options")
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.expand_button.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_button.setCheckable(True)
        self.expand_button.setChecked(False)
        self.set_stage_index(stage_index)

        self.move_up_button = QToolButton()
        self.move_up_button.setArrowType(Qt.ArrowType.UpArrow)
        self.move_up_button.setToolTip("Move stage up")
        self.move_down_button = QToolButton()
        self.move_down_button.setArrowType(Qt.ArrowType.DownArrow)
        self.move_down_button.setToolTip("Move stage down")

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.addWidget(self.enable_check)
        header.addStretch()
        header.addWidget(self.move_up_button)
        header.addWidget(self.move_down_button)
        header.addWidget(self.expand_button)

        self.content = QWidget()
        self.content.setVisible(False)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 4, 8, 8)
        self.content_layout.setSpacing(6)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addWidget(self.content)

        self.enable_check.stateChanged.connect(self.enabledChanged.emit)
        self.expand_button.toggled.connect(self._set_expanded)
        self.move_up_button.clicked.connect(lambda: self.moveRequested.emit(-1))
        self.move_down_button.clicked.connect(lambda: self.moveRequested.emit(1))

    def _set_expanded(self, expanded: bool) -> None:
        self.expand_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)

    def set_stage_index(self, stage_index: int) -> None:
        self.enable_check.setText(f"{stage_index}. {self.stage_title}")

    def set_move_enabled(self, can_move_up: bool, can_move_down: bool) -> None:
        self.move_up_button.setEnabled(can_move_up)
        self.move_down_button.setEnabled(can_move_down)


class ContrastWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        missing = [str(path) for path in DEFAULT_VIDEOS.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing default video files: " + ", ".join(missing))

        self.setWindowTitle("Contrast Residence Analyzer")
        self.resize(1500, 940)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.is_playing = False
        self.current_frame_index = 0
        self.results: dict[str, AnalysisResult] = {}
        self.deep_denoisers: dict[str, FrameDenoiser] = {}

        self.pre_panel = VideoPanel("Pre-deployment", QColor("#38bdf8"), DEFAULT_VIDEOS["Pre-deployment"])
        self.post_panel = VideoPanel("Post-deployment", QColor("#f97316"), DEFAULT_VIDEOS["Post-deployment"])
        self.panels = [self.pre_panel, self.post_panel]
        for panel in self.panels:
            panel.roiChanged.connect(self.on_roi_changed)

        self.max_frame = min(panel.info.frame_count for panel in self.panels) - 1
        self.fps = min(panel.info.fps for panel in self.panels)
        self.playback_speed = 1.0
        self.play_interval_ms = self._play_interval_ms()

        self._build_actions()
        self._build_ui()
        self._apply_style()
        self.on_enhancement_settings_changed()
        self.set_display_enhancement(False)
        self.update_time_label()
        self.statusBar().showMessage("Draw one ROI on each video, then run analysis.")

    def _build_actions(self) -> None:
        self.open_pre_action = QAction("Open pre-deployment video...", self)
        self.open_pre_action.triggered.connect(lambda: self.open_video(self.pre_panel))
        self.open_post_action = QAction("Open post-deployment video...", self)
        self.open_post_action.triggered.connect(lambda: self.open_video(self.post_panel))
        self.export_action = QAction("Export analysis CSV...", self)
        self.export_action.triggered.connect(self.export_csv)
        self.export_action.setEnabled(False)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_pre_action)
        file_menu.addAction(self.open_post_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)

    def _build_ui(self) -> None:
        self.play_button = QPushButton("Play")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)

        self.step_back_button = QPushButton("-1 frame")
        self.step_back_button.clicked.connect(lambda: self.set_frame_index(self.current_frame_index - 1))
        self.step_forward_button = QPushButton("+1 frame")
        self.step_forward_button.clicked.connect(lambda: self.set_frame_index(self.current_frame_index + 1))

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, self.max_frame)
        self.frame_slider.valueChanged.connect(self.set_frame_index)
        self.frame_slider.setMinimumWidth(360)

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, self.max_frame)
        self.frame_spin.valueChanged.connect(self.set_frame_index)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(25, 400)
        self.speed_slider.setSingleStep(25)
        self.speed_slider.setPageStep(50)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.setValue(100)
        self.speed_slider.setFixedWidth(140)
        self.speed_slider.valueChanged.connect(self.set_playback_speed)

        self.speed_label = QLabel()
        self.speed_label.setObjectName("timeLabel")
        self.update_speed_label()

        self.time_label = QLabel()
        self.time_label.setObjectName("timeLabel")

        self.compare_view_check = QCheckBox("Side-by-side compare")
        self.compare_view_check.setChecked(True)
        self.compare_view_check.toggled.connect(self.on_compare_view_toggled)

        video_row = QWidget()
        video_layout = QHBoxLayout(video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(14)
        video_layout.addWidget(self.pre_panel)
        video_layout.addWidget(self.post_panel)

        playback_row = QWidget()
        playback_layout = QHBoxLayout(playback_row)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        playback_layout.setSpacing(8)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.step_back_button)
        playback_layout.addWidget(self.step_forward_button)
        playback_layout.addWidget(self.frame_slider, 1)
        playback_layout.addWidget(QLabel("Frame"))
        playback_layout.addWidget(self.frame_spin)
        playback_layout.addWidget(QLabel("Speed"))
        playback_layout.addWidget(self.speed_slider)
        playback_layout.addWidget(self.speed_label)
        playback_layout.addWidget(self.time_label)
        playback_layout.addWidget(self.compare_view_check)

        controls_panel = self._build_controls_panel()
        plot_panel = self._build_plot_panel()

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)
        right_layout.addWidget(video_row, 3)
        right_layout.addWidget(playback_row)
        right_layout.addWidget(plot_panel, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls_panel)
        splitter.addWidget(right_column)
        splitter.setSizes([420, 1080])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.loading_overlay = LoadingOverlay(central)
        self.loading_overlay.setGeometry(central.rect())
        self.setStatusBar(QStatusBar())

    def _build_controls_panel(self) -> QWidget:
        controls = QTabWidget()
        controls.setMaximumWidth(410)
        controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        enhancement_tab = QWidget()
        enhancement_layout = QVBoxLayout(enhancement_tab)
        enhancement_layout.setContentsMargins(12, 12, 12, 12)
        enhancement_layout.setSpacing(10)

        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(12, 12, 12, 12)
        analysis_layout.setSpacing(10)

        controls.addTab(enhancement_tab, "Enhancement")
        controls.addTab(analysis_tab, "ROI analysis")

        hint = QLabel("Draw a box over the aneurysm sac in each video. The signal is computed as baseline brightness minus ROI brightness, so darker contrast produces a positive curve.")
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        analysis_layout.addWidget(hint)

        pipeline_label = QLabel("Enhancement pipeline (applied in this order, top to bottom)")
        pipeline_label.setObjectName("pipelineLabel")
        enhancement_layout.addWidget(pipeline_label)
        self.enhancement_layout = enhancement_layout
        self.gain_stage_drawer = StageDrawer("gain_stabilization", "Gain stabilization", 1)
        self.scanline_stage_drawer = StageDrawer("scanline_correction", "Scanline correction", 2)
        self.denoise_stage_drawer = StageDrawer("denoise", "Spatial denoising", 3)
        self.temporal_stage_drawer = StageDrawer("temporal_filter", "Motion-aware temporal filtering", 4)
        self.contrast_stage_drawer = StageDrawer("local_contrast", "Local contrast (CLAHE)", 5)
        self.smoothing_stage_drawer = StageDrawer("final_smoothing", "Final Gaussian smoothing", 6)
        self.gain_stage_check = self.gain_stage_drawer.enable_check
        self.scanline_stage_check = self.scanline_stage_drawer.enable_check
        self.denoise_stage_check = self.denoise_stage_drawer.enable_check
        self.temporal_stage_check = self.temporal_stage_drawer.enable_check
        self.contrast_stage_check = self.contrast_stage_drawer.enable_check
        self.smoothing_stage_check = self.smoothing_stage_drawer.enable_check
        self.pipeline_stage_checks = [
            self.gain_stage_check,
            self.scanline_stage_check,
            self.denoise_stage_check,
            self.temporal_stage_check,
            self.contrast_stage_check,
            self.smoothing_stage_check,
        ]
        self.pipeline_stage_drawers = [
            self.gain_stage_drawer,
            self.scanline_stage_drawer,
            self.denoise_stage_drawer,
            self.temporal_stage_drawer,
            self.contrast_stage_drawer,
            self.smoothing_stage_drawer,
        ]
        for check in self.pipeline_stage_checks:
            check.setChecked(False)
            check.stateChanged.connect(self.on_pipeline_stages_changed)
        self._build_stage_drawer_controls()
        for drawer in self.pipeline_stage_drawers:
            drawer.moveRequested.connect(lambda direction, current=drawer: self._move_pipeline_stage(current, direction))
        for drawer in self.pipeline_stage_drawers:
            enhancement_layout.addWidget(drawer)

        self.reset_pipeline_button = QPushButton("Show original videos")
        self.reset_pipeline_button.clicked.connect(self.reset_enhancement_pipeline)
        enhancement_layout.addWidget(self.reset_pipeline_button)
        self._refresh_pipeline_stage_ui()
        enhancement_layout.addStretch()
        self.enhancement_mode_combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)
        self.denoise_strength_spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.inference_batch_spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.inference_precision_combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)

        self.gain_correct_check = QCheckBox("Correct gain drift in analysis")
        self.gain_correct_check.setChecked(True)
        self.gain_correct_check.stateChanged.connect(lambda: self.on_analysis_filter_changed())
        analysis_layout.addWidget(self.gain_correct_check)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Clearance threshold"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.20)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self.refresh_analysis_from_existing)
        threshold_row.addWidget(self.threshold_spin)
        analysis_layout.addLayout(threshold_row)

        self.analyze_button = QPushButton("Analyze ROIs")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(self.analyze_button)

        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        analysis_layout.addWidget(self.export_button)
        analysis_layout.addStretch()

        self.pre_card = MetricCard("Pre residence")
        self.post_card = MetricCard("Post residence")
        self.delta_card = MetricCard("Difference")
        analysis_layout.addWidget(self.pre_card)
        analysis_layout.addWidget(self.post_card)
        analysis_layout.addWidget(self.delta_card)

        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("controlsScroll")
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setMaximumWidth(430)
        controls_scroll.setMinimumWidth(390)
        return controls_scroll

    def _build_plot_panel(self) -> QWidget:
        plot_group = QFrame()
        plot_group.setObjectName("plotPanel")
        plot_layout = QGridLayout(plot_group)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)

        self.normalized_plot = pg.PlotWidget(title="Normalized Contrast Residence")
        self.raw_plot = pg.PlotWidget(title="Denoised ROI Brightness")
        for plot in [self.normalized_plot, self.raw_plot]:
            plot.setBackground("#111827")
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis("bottom").setPen("#8aa0b8")
            plot.getAxis("left").setPen("#8aa0b8")
            plot.getAxis("bottom").setTextPen("#cbd5e1")
            plot.getAxis("left").setTextPen("#cbd5e1")
            plot.addLegend(offset=(12, 12))
        self.normalized_plot.setLabel("bottom", "Time", units="s")
        self.normalized_plot.setLabel("left", "Normalized signal")
        self.raw_plot.setLabel("bottom", "Time", units="s")
        self.raw_plot.setLabel("left", "Mean pixel value")

        plot_layout.addWidget(self.normalized_plot, 0, 0)
        plot_layout.addWidget(self.raw_plot, 1, 0)
        return plot_group

    def _build_stage_drawer_controls(self) -> None:
        enhancement_mode_row = QHBoxLayout()
        enhancement_mode_row.addWidget(QLabel("Enhancement model"))
        self.enhancement_mode_combo = QComboBox()
        self.enhancement_mode_combo.addItem("NGC FFDNet (Docker)", "ffdnet-ngc")
        self.enhancement_mode_combo.addItem("NGC DnCNN 15 (mild)", "dncnn-15-ngc")
        self.enhancement_mode_combo.addItem("NGC DnCNN 25 (balanced)", "dncnn-25-ngc")
        self.enhancement_mode_combo.addItem("NGC DnCNN 50 (strong)", "dncnn-50-ngc")
        self.enhancement_mode_combo.addItem("Native FFDNet (GPU)", "ffdnet-native")
        self.enhancement_mode_combo.addItem("Classical", "classical")
        enhancement_mode_row.addWidget(self.enhancement_mode_combo, 1)
        self.denoise_stage_drawer.content_layout.addLayout(enhancement_mode_row)

        denoise_strength_row = QHBoxLayout()
        self.denoise_strength_label = QLabel("FFDNet noise sigma")
        denoise_strength_row.addWidget(self.denoise_strength_label)
        self.denoise_strength_spin = QSpinBox()
        self.denoise_strength_spin.setRange(0, 50)
        self.denoise_strength_spin.setValue(10)
        self.denoise_strength_spin.setSuffix(" / 255")
        self.denoise_strength_spin.setToolTip("FFDNet's assumed input noise standard deviation")
        denoise_strength_row.addWidget(self.denoise_strength_spin)
        self.denoise_stage_drawer.content_layout.addLayout(denoise_strength_row)

        inference_row = QHBoxLayout()
        inference_row.addWidget(QLabel("Batch frames"))
        self.inference_batch_spin = QSpinBox()
        self.inference_batch_spin.setRange(1, 16)
        self.inference_batch_spin.setValue(4)
        self.inference_batch_spin.setToolTip("More frames improve GPU throughput but use more GPU and shared memory")
        inference_row.addWidget(self.inference_batch_spin)
        inference_row.addWidget(QLabel("Precision"))
        self.inference_precision_combo = QComboBox()
        self.inference_precision_combo.addItem("FP16", "fp16")
        self.inference_precision_combo.addItem("FP32", "fp32")
        self.inference_precision_combo.setToolTip("FP16 is faster; FP32 is useful for numerical comparisons")
        inference_row.addWidget(self.inference_precision_combo)
        self.denoise_stage_drawer.content_layout.addLayout(inference_row)

        gain_auto_row = QHBoxLayout()
        self.gain_auto_target_check = QCheckBox("Use per-video auto target median")
        self.gain_auto_target_check.setChecked(True)
        gain_auto_row.addWidget(self.gain_auto_target_check)
        gain_auto_row.addStretch()
        self.gain_stage_drawer.content_layout.addLayout(gain_auto_row)

        gain_target_row = QHBoxLayout()
        gain_target_row.addWidget(QLabel("Manual target median"))
        self.gain_target_spin = QSpinBox()
        self.gain_target_spin.setRange(1, 255)
        self.gain_target_spin.setValue(128)
        self.gain_target_spin.setEnabled(False)
        gain_target_row.addWidget(self.gain_target_spin)
        self.gain_stage_drawer.content_layout.addLayout(gain_target_row)

        gain_bounds_row = QHBoxLayout()
        gain_bounds_row.addWidget(QLabel("Gain clamp"))
        self.gain_min_spin = QDoubleSpinBox()
        self.gain_min_spin.setRange(0.10, 2.00)
        self.gain_min_spin.setSingleStep(0.05)
        self.gain_min_spin.setDecimals(2)
        self.gain_min_spin.setValue(0.70)
        self.gain_min_spin.setPrefix("min ")
        self.gain_max_spin = QDoubleSpinBox()
        self.gain_max_spin.setRange(0.10, 2.00)
        self.gain_max_spin.setSingleStep(0.05)
        self.gain_max_spin.setDecimals(2)
        self.gain_max_spin.setValue(1.45)
        self.gain_max_spin.setPrefix("max ")
        gain_bounds_row.addWidget(self.gain_min_spin)
        gain_bounds_row.addWidget(self.gain_max_spin)
        self.gain_stage_drawer.content_layout.addLayout(gain_bounds_row)

        scanline_clip_row = QHBoxLayout()
        scanline_clip_row.addWidget(QLabel("Row bias clip"))
        self.scanline_clip_spin = QDoubleSpinBox()
        self.scanline_clip_spin.setRange(0.5, 20.0)
        self.scanline_clip_spin.setSingleStep(0.5)
        self.scanline_clip_spin.setDecimals(1)
        self.scanline_clip_spin.setValue(6.0)
        scanline_clip_row.addWidget(self.scanline_clip_spin)
        self.scanline_stage_drawer.content_layout.addLayout(scanline_clip_row)

        scanline_sigma_row = QHBoxLayout()
        scanline_sigma_row.addWidget(QLabel("Vertical blur sigma"))
        self.scanline_sigma_spin = QDoubleSpinBox()
        self.scanline_sigma_spin.setRange(0.2, 8.0)
        self.scanline_sigma_spin.setSingleStep(0.1)
        self.scanline_sigma_spin.setDecimals(1)
        self.scanline_sigma_spin.setValue(2.0)
        scanline_sigma_row.addWidget(self.scanline_sigma_spin)
        self.scanline_stage_drawer.content_layout.addLayout(scanline_sigma_row)

        denoise_d_row = QHBoxLayout()
        denoise_d_row.addWidget(QLabel("Classical bilateral diameter"))
        self.bilateral_diameter_spin = QSpinBox()
        self.bilateral_diameter_spin.setRange(1, 31)
        self.bilateral_diameter_spin.setSingleStep(2)
        self.bilateral_diameter_spin.setValue(7)
        denoise_d_row.addWidget(self.bilateral_diameter_spin)
        self.denoise_stage_drawer.content_layout.addLayout(denoise_d_row)

        denoise_sigma_row = QHBoxLayout()
        denoise_sigma_row.addWidget(QLabel("Classical sigma color"))
        self.bilateral_sigma_color_spin = QDoubleSpinBox()
        self.bilateral_sigma_color_spin.setRange(1.0, 120.0)
        self.bilateral_sigma_color_spin.setSingleStep(1.0)
        self.bilateral_sigma_color_spin.setDecimals(1)
        self.bilateral_sigma_color_spin.setValue(18.0)
        denoise_sigma_row.addWidget(self.bilateral_sigma_color_spin)
        denoise_sigma_row.addWidget(QLabel("sigma space"))
        self.bilateral_sigma_space_spin = QDoubleSpinBox()
        self.bilateral_sigma_space_spin.setRange(1.0, 80.0)
        self.bilateral_sigma_space_spin.setSingleStep(1.0)
        self.bilateral_sigma_space_spin.setDecimals(1)
        self.bilateral_sigma_space_spin.setValue(4.0)
        denoise_sigma_row.addWidget(self.bilateral_sigma_space_spin)
        self.denoise_stage_drawer.content_layout.addLayout(denoise_sigma_row)

        temporal_row = QHBoxLayout()
        temporal_row.addWidget(QLabel("Motion sensitivity sigma"))
        self.temporal_sigma_spin = QDoubleSpinBox()
        self.temporal_sigma_spin.setRange(1.0, 60.0)
        self.temporal_sigma_spin.setSingleStep(0.5)
        self.temporal_sigma_spin.setDecimals(1)
        self.temporal_sigma_spin.setValue(12.0)
        temporal_row.addWidget(self.temporal_sigma_spin)
        self.temporal_stage_drawer.content_layout.addLayout(temporal_row)

        clahe_clip_row = QHBoxLayout()
        clahe_clip_row.addWidget(QLabel("CLAHE clip limit"))
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.1, 10.0)
        self.clahe_clip_spin.setSingleStep(0.1)
        self.clahe_clip_spin.setDecimals(1)
        self.clahe_clip_spin.setValue(1.0)
        clahe_clip_row.addWidget(self.clahe_clip_spin)
        self.contrast_stage_drawer.content_layout.addLayout(clahe_clip_row)

        clahe_tile_row = QHBoxLayout()
        clahe_tile_row.addWidget(QLabel("CLAHE tile size"))
        self.clahe_tile_spin = QSpinBox()
        self.clahe_tile_spin.setRange(2, 24)
        self.clahe_tile_spin.setValue(6)
        clahe_tile_row.addWidget(self.clahe_tile_spin)
        self.contrast_stage_drawer.content_layout.addLayout(clahe_tile_row)

        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(QLabel("Gaussian sigma"))
        self.smoothing_sigma_spin = QDoubleSpinBox()
        self.smoothing_sigma_spin.setRange(0.1, 4.0)
        self.smoothing_sigma_spin.setSingleStep(0.05)
        self.smoothing_sigma_spin.setDecimals(2)
        self.smoothing_sigma_spin.setValue(0.55)
        smoothing_row.addWidget(self.smoothing_sigma_spin)
        self.smoothing_stage_drawer.content_layout.addLayout(smoothing_row)

        self.gain_auto_target_check.toggled.connect(self._on_gain_auto_target_toggled)
        for spin in [
            self.gain_target_spin,
            self.gain_min_spin,
            self.gain_max_spin,
            self.scanline_clip_spin,
            self.scanline_sigma_spin,
            self.bilateral_diameter_spin,
            self.bilateral_sigma_color_spin,
            self.bilateral_sigma_space_spin,
            self.temporal_sigma_spin,
            self.clahe_clip_spin,
            self.clahe_tile_spin,
            self.smoothing_sigma_spin,
        ]:
            spin.valueChanged.connect(self.on_enhancement_settings_changed)

    def _on_gain_auto_target_toggled(self, checked: bool) -> None:
        self.gain_target_spin.setEnabled(not checked)
        self.on_enhancement_settings_changed()

    def _apply_style(self) -> None:
        pg.setConfigOptions(antialias=True)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1018; color: #e5edf6; font-size: 13px; }
            QMenuBar, QMenu, QToolBar { background: #111827; color: #e5edf6; border: none; }
            QToolBar { padding: 8px; spacing: 8px; }
            QMenu::item:selected { background: #243047; }
            QPushButton { background: #1c2637; border: 1px solid #334155; border-radius: 7px; color: #e5edf6; padding: 7px 12px; }
            QPushButton:hover { background: #263449; }
            QPushButton:disabled { color: #64748b; background: #111827; }
            QPushButton#primaryButton { background: #0f766e; border-color: #14b8a6; font-weight: 700; }
            QPushButton#primaryButton:hover { background: #0d9488; }
            QSlider::groove:horizontal { height: 6px; background: #273449; border-radius: 3px; }
            QSlider::handle:horizontal { background: #67e8f9; width: 16px; margin: -5px 0; border-radius: 8px; }
            QSpinBox, QDoubleSpinBox, QComboBox { background: #111827; border: 1px solid #334155; border-radius: 6px; padding: 5px; color: #e5edf6; }
            QGroupBox { background: #111827; border: 1px solid #253044; border-radius: 8px; margin-top: 12px; padding: 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #f8fafc; }
            QFrame#videoPanel, QFrame#plotPanel { background: #111827; border: 1px solid #253044; border-radius: 8px; }
            QFrame#stageDrawer { background: #0f172a; border: 1px solid #273449; border-radius: 8px; }
            QLabel#panelTitle { font-size: 16px; font-weight: 700; color: #f8fafc; }
            QLabel#pipelineLabel { color: #e2e8f0; font-weight: 700; padding-top: 4px; }
            QLabel#subtleLabel, QLabel#hintLabel { color: #9fb0c6; }
            QLabel#timeLabel { color: #cbd5e1; padding-left: 12px; }
            QFrame#metricCard { background: #0f172a; border: 1px solid #273449; border-radius: 8px; }
            QLabel#metricTitle { color: #9fb0c6; font-size: 12px; }
            QLabel#metricValue { color: #f8fafc; font-size: 22px; font-weight: 800; }
            QLabel#metricDetail { color: #94a3b8; font-size: 12px; }
            QStatusBar { background: #0b1018; color: #9fb0c6; }
            QFrame#loadingOverlay { background: rgba(4, 8, 15, 180); }
            QFrame#loadingOverlayPanel { background: #0f172a; border: 1px solid #334155; border-radius: 10px; min-width: 380px; }
            QLabel#loadingOverlayLabel { color: #e2e8f0; font-size: 14px; font-weight: 700; }
            QProgressBar { border: 1px solid #334155; border-radius: 6px; background: #111827; color: #e5edf6; text-align: center; }
            QProgressBar::chunk { background: #14b8a6; border-radius: 5px; }
            """
        )

    def open_video(self, panel: VideoPanel) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open {panel.label} video",
            str(ROOT),
            "Video files (*.mov *.mp4 *.avi *.mkv);;All files (*)",
        )
        if not path:
            return

        self.pause()
        try:
            panel.set_video(Path(path))
        except RuntimeError as exc:
            QMessageBox.critical(self, "Could not open video", str(exc))
            return

        self.results.clear()
        self.max_frame = min(item.info.frame_count for item in self.panels) - 1
        self.fps = min(item.info.fps for item in self.panels)
        self.play_interval_ms = self._play_interval_ms()
        self.frame_slider.setRange(0, self.max_frame)
        self.frame_spin.setRange(0, self.max_frame)
        self.set_frame_index(0)
        self.clear_plots_and_metrics()

        if self.enhancement_stages().any_enabled:
            self.rebuild_enhancement_pipeline()
        else:
            self.set_display_enhancement(False)

    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        self.is_playing = True
        self.play_button.setText("Pause")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.timer.start(self.play_interval_ms)

    def _play_interval_ms(self) -> int:
        effective_fps = self.fps * self.playback_speed
        if effective_fps <= 0:
            return 1
        return max(1, round(1000 / effective_fps))

    def set_playback_speed(self, slider_value: int) -> None:
        self.playback_speed = slider_value / 100.0
        self.play_interval_ms = self._play_interval_ms()
        self.update_speed_label()
        if self.is_playing:
            self.timer.start(self.play_interval_ms)

    def update_speed_label(self) -> None:
        self.speed_label.setText(f"{self.playback_speed:.2f}x")

    def pause(self) -> None:
        self.is_playing = False
        self.timer.stop()
        self.play_button.setText("Play")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        if any(panel.enhance_display for panel in self.panels):
            for panel in self.panels:
                panel.seek(self.current_frame_index)

    def advance_frame(self) -> None:
        if self.current_frame_index >= self.max_frame:
            self.pause()
            return
        next_frame_index = self.current_frame_index + 1
        if not all(panel.read_next(playback=True) for panel in self.panels):
            self.pause()
            return
        self.current_frame_index = next_frame_index
        for widget in [self.frame_slider, self.frame_spin]:
            if widget.value() != next_frame_index:
                widget.blockSignals(True)
                widget.setValue(next_frame_index)
                widget.blockSignals(False)
        self.update_time_label()

    def set_frame_index(self, frame_index: int) -> None:
        frame_index = max(0, min(frame_index, self.max_frame))
        if frame_index == self.current_frame_index and self.frame_slider.value() == frame_index:
            return
        self.current_frame_index = frame_index
        for panel in self.panels:
            panel.seek(frame_index)
        for widget in [self.frame_slider, self.frame_spin]:
            if widget.value() != frame_index:
                widget.blockSignals(True)
                widget.setValue(frame_index)
                widget.blockSignals(False)
        self.update_time_label()

    def update_time_label(self) -> None:
        current_time = self.current_frame_index / self.fps if self.fps else 0.0
        total_time = self.max_frame / self.fps if self.fps else 0.0
        self.time_label.setText(f"{current_time:05.2f} s / {total_time:05.2f} s")

    def on_roi_changed(self) -> None:
        self.results.clear()
        self.clear_plots_and_metrics()
        ready = all(panel.roi() for panel in self.panels)
        if ready:
            self.statusBar().showMessage("Both ROIs are selected. Run analysis to compare contrast residence.")
        else:
            self.statusBar().showMessage("Draw one ROI on each video.")

    def set_display_enhancement(self, enabled: bool) -> None:
        for panel in self.panels:
            panel.set_enhancement(enabled, self.current_frame_index)
        self.statusBar().showMessage("Video enhancement enabled." if enabled else "Video enhancement disabled.")

    def on_compare_view_toggled(self, enabled: bool) -> None:
        for panel in self.panels:
            panel.set_comparison(enabled, self.current_frame_index)
        if enabled:
            self.statusBar().showMessage("Side-by-side original vs enhanced comparison enabled.")
        else:
            self.statusBar().showMessage("Single-view mode enabled.")

    def on_enhancement_settings_changed(self) -> None:
        active_mode = str(self.enhancement_mode_combo.currentData())
        stages = self.enhancement_stages()
        use_deep_model = stages.denoise and active_mode != "classical"
        uses_ffdnet = stages.denoise and active_mode.startswith("ffdnet")
        self.denoise_strength_label.setEnabled(uses_ffdnet)
        self.denoise_strength_spin.setEnabled(uses_ffdnet)
        self.inference_batch_spin.setEnabled(use_deep_model)
        self.inference_precision_combo.setEnabled(use_deep_model)
        active_key = self._denoiser_key(active_mode) if use_deep_model else ""
        for key, denoiser in list(self.deep_denoisers.items()):
            if key == active_key:
                continue
            close = getattr(denoiser, "close", None)
            if close is not None:
                close()
            del self.deep_denoisers[key]
        if stages.any_enabled:
            self.rebuild_enhancement_pipeline()
        else:
            self.statusBar().showMessage("Enhancement settings updated. Enable one or more stages to preview.")

    def enhancement_stages(self) -> EnhancementStages:
        return EnhancementStages(
            gain_stabilization=self.gain_stage_check.isChecked(),
            scanline_correction=self.scanline_stage_check.isChecked(),
            denoise=self.denoise_stage_check.isChecked(),
            temporal_filter=self.temporal_stage_check.isChecked(),
            local_contrast=self.contrast_stage_check.isChecked(),
            final_smoothing=self.smoothing_stage_check.isChecked(),
            stage_order=tuple(drawer.stage_key for drawer in self.pipeline_stage_drawers),
        )

    def _move_pipeline_stage(self, drawer: StageDrawer, direction: int) -> None:
        current_index = self.pipeline_stage_drawers.index(drawer)
        target_index = current_index + direction
        if target_index < 0 or target_index >= len(self.pipeline_stage_drawers):
            return
        self.pipeline_stage_drawers[current_index], self.pipeline_stage_drawers[target_index] = (
            self.pipeline_stage_drawers[target_index],
            self.pipeline_stage_drawers[current_index],
        )
        self.pipeline_stage_checks = [item.enable_check for item in self.pipeline_stage_drawers]
        self._refresh_pipeline_stage_ui()
        self.on_enhancement_settings_changed()

    def _refresh_pipeline_stage_ui(self) -> None:
        for drawer in self.pipeline_stage_drawers:
            self.enhancement_layout.removeWidget(drawer)
        insert_index = self.enhancement_layout.indexOf(self.reset_pipeline_button)
        if insert_index < 0:
            insert_index = self.enhancement_layout.count()
        for offset, drawer in enumerate(self.pipeline_stage_drawers):
            drawer.set_stage_index(offset + 1)
            drawer.set_move_enabled(offset > 0, offset < len(self.pipeline_stage_drawers) - 1)
            self.enhancement_layout.insertWidget(insert_index + offset, drawer)

    def enhancement_parameters(self) -> EnhancementParameters:
        gain_min = self.gain_min_spin.value()
        gain_max = self.gain_max_spin.value()
        if gain_min > gain_max:
            gain_min, gain_max = gain_max, gain_min
        return EnhancementParameters(
            gain_use_auto_target=self.gain_auto_target_check.isChecked(),
            gain_target_median=self.gain_target_spin.value(),
            gain_min=gain_min,
            gain_max=gain_max,
            scanline_bias_clip=self.scanline_clip_spin.value(),
            scanline_sigma_y=self.scanline_sigma_spin.value(),
            bilateral_diameter=self.bilateral_diameter_spin.value(),
            bilateral_sigma_color=self.bilateral_sigma_color_spin.value(),
            bilateral_sigma_space=self.bilateral_sigma_space_spin.value(),
            temporal_motion_sigma=self.temporal_sigma_spin.value(),
            clahe_clip_limit=self.clahe_clip_spin.value(),
            clahe_tile_size=self.clahe_tile_spin.value(),
            smoothing_sigma_x=self.smoothing_sigma_spin.value(),
        )

    def on_pipeline_stages_changed(self) -> None:
        stages = self.enhancement_stages()
        active_mode = str(self.enhancement_mode_combo.currentData())
        use_deep_model = stages.denoise and active_mode != "classical"
        uses_ffdnet = stages.denoise and active_mode.startswith("ffdnet")
        self.denoise_strength_label.setEnabled(uses_ffdnet)
        self.denoise_strength_spin.setEnabled(uses_ffdnet)
        self.inference_batch_spin.setEnabled(use_deep_model)
        self.inference_precision_combo.setEnabled(use_deep_model)
        if not use_deep_model:
            for key, denoiser in list(self.deep_denoisers.items()):
                close = getattr(denoiser, "close", None)
                if close is not None:
                    close()
                del self.deep_denoisers[key]
        if stages.any_enabled:
            self.rebuild_enhancement_pipeline()
        else:
            for panel in self.panels:
                panel.clear_enhancement_cache()
            self.set_display_enhancement(False)
            self.statusBar().showMessage("Showing original videos.")

    def reset_enhancement_pipeline(self) -> None:
        for check in self.pipeline_stage_checks:
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
        self.on_pipeline_stages_changed()

    def _denoiser_key(self, mode: str) -> str:
        precision = str(self.inference_precision_combo.currentData())
        batch_size = self.inference_batch_spin.value()
        return f"{mode}:{precision}:batch{batch_size}"

    def rebuild_enhancement_pipeline(self) -> None:
        for panel in self.panels:
            panel.clear_enhancement_cache()
        if self.ensure_enhancement_ready():
            self.set_display_enhancement(True)
        else:
            self.set_display_enhancement(False)

    def ensure_enhancement_ready(self) -> bool:
        mode = str(self.enhancement_mode_combo.currentData())
        stages = self.enhancement_stages()
        parameters = self.enhancement_parameters()
        use_deep_model = stages.denoise and mode != "classical"
        noise_sigma = self.denoise_strength_spin.value()
        batch_size = self.inference_batch_spin.value()
        precision = str(self.inference_precision_combo.currentData())
        denoiser = None
        if use_deep_model:
            try:
                denoiser_key = self._denoiser_key(mode)
                if denoiser_key not in self.deep_denoisers:
                    if mode.endswith("-ngc"):
                        self.loading_overlay.begin("Starting the NGC PyTorch worker...", 1)
                        from container_denoiser import ContainerDenoiser

                        model_name = mode.removesuffix("-ngc")
                        weights_name = "ffdnet_gray.pth" if model_name == "ffdnet" else f"{model_name.replace('-', '_')}.pth"
                        self.deep_denoisers[denoiser_key] = ContainerDenoiser(
                            model_name,
                            ROOT / "models" / weights_name,
                            batch_size,
                            precision,
                        )
                    else:
                        self.loading_overlay.begin("Loading FFDNet on the NVIDIA GPU...", 1)
                        from deep_denoiser import FFDNetDenoiser

                        self.deep_denoisers[denoiser_key] = FFDNetDenoiser(
                            ROOT / "models" / "ffdnet_gray.pth",
                            precision,
                        )
                denoiser = self.deep_denoisers[denoiser_key]
            except (ImportError, OSError, RuntimeError) as exc:
                QMessageBox.critical(self, "Deep enhancement unavailable", str(exc))
                return False
            finally:
                self.loading_overlay.finish()

        backend_id = denoiser.backend_id if denoiser is not None else "classical"
        expected_signature = (backend_id, noise_sigma if stages.denoise else 0, stages, parameters)
        pending_panels = [
            panel
            for panel in self.panels
            if panel.enhanced_frames is None or panel.enhancement_signature != expected_signature
        ]
        if not pending_panels:
            return True

        self.pause()
        total_frames = sum(panel.info.frame_count for panel in pending_panels)
        if denoiser is not None:
            model_label = self.enhancement_mode_combo.currentText().split(" (")[0]
            message = f"Running {model_label} on {denoiser.device_name}..."
        else:
            message = "Running classical video enhancement..."
        self.loading_overlay.begin(message, total_frames)

        completed = 0
        try:
            for panel in pending_panels:
                def progress_callback(done: int, _total: int) -> bool:
                    self.loading_overlay.set_progress(completed + done)
                    return True

                prepared = panel.prepare_enhanced_frames(
                    denoiser,
                    noise_sigma,
                    batch_size,
                    stages,
                    parameters,
                    progress_callback,
                )
                if not prepared:
                    return False
                completed += panel.info.frame_count
                self.loading_overlay.set_progress(completed)
            return True
        except RuntimeError as exc:
            QMessageBox.critical(self, "Enhancement failed", str(exc))
            return False
        finally:
            self.loading_overlay.finish()

    def on_analysis_filter_changed(self) -> None:
        if self.results:
            self.results.clear()
            self.clear_plots_and_metrics()
            self.statusBar().showMessage("Analysis filter changed. Run ROI analysis again.")

    def run_analysis(self) -> None:
        missing = [panel.label for panel in self.panels if panel.roi() is None]
        if missing:
            QMessageBox.information(self, "ROIs required", "Draw an ROI on: " + ", ".join(missing))
            return

        self.pause()
        threshold = self.threshold_spin.value()
        progress = QProgressDialog("Measuring ROI intensity...", "Cancel", 0, sum(panel.info.frame_count for panel in self.panels), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(250)
        progress.setValue(0)

        self.results.clear()
        completed = 0
        try:
            for panel in self.panels:
                assert panel.roi() is not None
                result = analyze_video(
                    panel.label,
                    panel.path,
                    panel.info,
                    panel.crop_rect,
                    panel.roi(),
                    threshold,
                    self.gain_correct_check.isChecked(),
                    progress,
                    completed,
                )
                completed += panel.info.frame_count
                progress.setValue(completed)
                if progress.wasCanceled():
                    self.statusBar().showMessage("Analysis canceled.")
                    return
                self.results[panel.label] = result
        finally:
            progress.close()

        self.refresh_plots_and_metrics()
        self.export_action.setEnabled(True)
        self.export_button.setEnabled(True)
        self.statusBar().showMessage("Analysis complete. Residence time is measured above the selected normalized threshold.")

    def refresh_analysis_from_existing(self) -> None:
        if not self.results:
            return
        threshold = self.threshold_spin.value()
        self.results = {
            label: recompute_threshold_metrics(result, threshold)
            for label, result in self.results.items()
        }
        self.refresh_plots_and_metrics()

    def clear_plots_and_metrics(self) -> None:
        self.normalized_plot.clear()
        self.raw_plot.clear()
        self.pre_card.set_metric("--")
        self.post_card.set_metric("--")
        self.delta_card.set_metric("--")
        self.export_action.setEnabled(False)
        self.export_button.setEnabled(False)

    def refresh_plots_and_metrics(self) -> None:
        self.normalized_plot.clear()
        self.raw_plot.clear()
        pens = {
            "Pre-deployment": pg.mkPen("#38bdf8", width=2.5),
            "Post-deployment": pg.mkPen("#f97316", width=2.5),
        }
        threshold = self.threshold_spin.value()
        for label, result in self.results.items():
            self.normalized_plot.plot(result.time, result.normalized_signal, pen=pens[label], name=label)
            self.raw_plot.plot(result.time, result.mean_intensity, pen=pens[label], name=label)

        if self.results:
            max_time = max(result.time[-1] for result in self.results.values() if len(result.time))
            threshold_line = pg.InfiniteLine(pos=threshold, angle=0, pen=pg.mkPen("#e2e8f0", width=1, style=Qt.PenStyle.DashLine))
            self.normalized_plot.addItem(threshold_line)
            self.normalized_plot.setXRange(0, max_time, padding=0)
            self.normalized_plot.setYRange(0, 1.05, padding=0)

        pre = self.results.get("Pre-deployment")
        post = self.results.get("Post-deployment")
        if pre:
            self.pre_card.set_metric(format_seconds(pre.residence_time), self._metric_detail(pre))
        if post:
            self.post_card.set_metric(format_seconds(post.residence_time), self._metric_detail(post))
        if pre and post and pre.residence_time is not None and post.residence_time is not None:
            delta = post.residence_time - pre.residence_time
            self.delta_card.set_metric(f"{delta:+.2f} s", "post minus pre")

    def _metric_detail(self, result: AnalysisResult) -> str:
        return f"arrival {format_seconds(result.arrival_time)} | peak {format_seconds(result.peak_time)} | clear {format_seconds(result.clear_time)}"

    def export_csv(self) -> None:
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis CSV", str(ROOT / "contrast_roi_analysis.csv"), "CSV files (*.csv)")
        if not path:
            return

        labels = list(self.results.keys())
        max_length = max(len(result.time) for result in self.results.values())
        with Path(path).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["label", "roi_x", "roi_y", "roi_width", "roi_height", "gain_corrected", "threshold_fraction", "arrival_s", "peak_s", "clear_s", "residence_s", "peak_signal", "auc"])
            for result in self.results.values():
                writer.writerow([
                    result.label,
                    result.roi.x(),
                    result.roi.y(),
                    result.roi.width(),
                    result.roi.height(),
                    result.gain_corrected,
                    result.threshold_fraction,
                    result.arrival_time,
                    result.peak_time,
                    result.clear_time,
                    result.residence_time,
                    result.peak_signal,
                    result.auc,
                ])
            writer.writerow([])
            header = []
            for label in labels:
                header.extend([f"{label} time_s", f"{label} measurement_intensity", f"{label} reference_intensity", f"{label} contrast_signal", f"{label} normalized_signal"])
            writer.writerow(header)
            for index in range(max_length):
                row = []
                for label in labels:
                    result = self.results[label]
                    if index < len(result.time):
                        row.extend([result.time[index], result.mean_intensity[index], result.reference_intensity[index], result.contrast_signal[index], result.normalized_signal[index]])
                    else:
                        row.extend(["", "", "", "", ""])
                writer.writerow(row)
        self.statusBar().showMessage(f"Exported analysis to {path}")

    def closeEvent(self, event) -> None:  # noqa: ANN001
        for panel in self.panels:
            panel.close()
        for denoiser in self.deep_denoisers.values():
            close = getattr(denoiser, "close", None)
            if close is not None:
                close()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if hasattr(self, "loading_overlay") and self.centralWidget() is not None:
            self.loading_overlay.setGeometry(self.centralWidget().rect())


def analyze_video(
    label: str,
    path: Path,
    info: VideoInfo,
    crop_rect: QRect,
    roi: QRect,
    threshold_fraction: float,
    gain_corrected: bool,
    progress: QProgressDialog,
    progress_offset: int,
) -> AnalysisResult:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    x, y = roi.x(), roi.y()
    width, height = roi.width(), roi.height()
    means: list[float] = []
    references: list[float] = []

    try:
        for frame_index in range(info.frame_count):
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY)
            roi_pixels = gray[y : y + height, x : x + width]
            means.append(float(np.mean(roi_pixels)))
            references.append(reference_mean(gray, roi))
            if frame_index % 15 == 0:
                progress.setValue(progress_offset + frame_index)
                QApplication.processEvents()
                if progress.wasCanceled():
                    break
    finally:
        capture.release()

    roi_intensity = np.asarray(means, dtype=float)
    reference_intensity = np.asarray(references, dtype=float)
    measurement_intensity = roi_intensity
    if gain_corrected and len(roi_intensity):
        baseline_count = baseline_sample_count(info.fps, len(reference_intensity))
        smoothed_reference = smooth_temporal_signal(reference_intensity, info.fps)
        baseline_reference = float(np.median(smoothed_reference[:baseline_count]))
        reference_safe = np.clip(smoothed_reference, 1.0, None)
        gain = np.clip(baseline_reference / reference_safe, 0.55, 1.85)
        measurement_intensity = roi_intensity * gain
    measurement_intensity = smooth_temporal_signal(measurement_intensity, info.fps)

    return build_analysis_result(label, path, info.fps, roi, measurement_intensity, reference_intensity, threshold_fraction, gain_corrected)


def build_analysis_result(
    label: str,
    path: Path,
    fps: float,
    roi: QRect,
    mean_intensity: np.ndarray,
    reference_intensity: np.ndarray,
    threshold_fraction: float,
    gain_corrected: bool,
) -> AnalysisResult:
    time = np.arange(len(mean_intensity), dtype=float) / fps
    if len(mean_intensity) == 0:
        empty = np.asarray([], dtype=float)
        return AnalysisResult(label, path, fps, roi, empty, empty, empty, empty, empty, gain_corrected, threshold_fraction, 0.0, None, None, None, None, 0.0, 0.0)

    baseline_count = baseline_sample_count(fps, len(mean_intensity))
    baseline = float(np.median(mean_intensity[:baseline_count]))
    contrast_signal = np.clip(baseline - mean_intensity, 0, None)
    peak_signal = float(np.max(contrast_signal))
    normalized = contrast_signal / peak_signal if peak_signal > 0 else np.zeros_like(contrast_signal)
    threshold_value = threshold_fraction
    above = normalized >= threshold_value

    arrival_time = peak_time = clear_time = residence_time = None
    if np.any(above):
        arrival_index = int(np.argmax(above))
        peak_index = int(np.argmax(normalized))
        after_peak = above[peak_index:]
        below_after_peak = np.flatnonzero(~after_peak)
        clear_index = peak_index + int(below_after_peak[0]) if len(below_after_peak) else len(normalized) - 1
        arrival_time = time[arrival_index]
        peak_time = time[peak_index]
        clear_time = time[clear_index]
        residence_time = max(0.0, clear_time - arrival_time)

    auc = float(np.trapezoid(normalized, time)) if len(time) > 1 else 0.0
    return AnalysisResult(
        label=label,
        path=path,
        fps=fps,
        roi=QRect(roi),
        time=time,
        mean_intensity=mean_intensity,
        reference_intensity=reference_intensity,
        contrast_signal=contrast_signal,
        normalized_signal=normalized,
        gain_corrected=gain_corrected,
        threshold_fraction=threshold_fraction,
        threshold_value=threshold_value,
        arrival_time=arrival_time,
        peak_time=peak_time,
        clear_time=clear_time,
        residence_time=residence_time,
        peak_signal=peak_signal,
        auc=auc,
    )


def recompute_threshold_metrics(result: AnalysisResult, threshold_fraction: float) -> AnalysisResult:
    return build_analysis_result(result.label, result.path, result.fps, result.roi, result.mean_intensity, result.reference_intensity, threshold_fraction, result.gain_corrected)


def main() -> None:
    app = QApplication(sys.argv)
    window = ContrastWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
