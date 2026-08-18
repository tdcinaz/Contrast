from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty, Full, Queue, SimpleQueue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable, Generator, Protocol, cast

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QImage, QMouseEvent, QPainter, QPdfWriter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QStatusBar,
    QStyle,
    QTabWidget,
    QToolButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from contrast_pipeline import (
    BUILTIN_STAGES,
    EnhancementParameters,
    EnhancementRequest,
    EnhancementStages,
    ExecutionShape,
    FrameContext,
    PipelineStage,
    subtract_fluoroscopy_background,
)
from dicom_source import is_dicom_video, open_video_capture, probe_dicom_video
from frame_scheduler import AdaptiveFrameExecutor
from logging_setup import configure_logging, install_exception_logging
from opencv_denoiser import DEFAULT_NLM_BACKEND_ID, NonLocalMeansDenoiser

ROOT = Path(__file__).resolve().parent
CONFIG_DIRECTORY = ROOT / "configs"
RECENT_CONFIG_FILE = CONFIG_DIRECTORY / "recent.json"
DEFAULT_PIPELINE_SETTINGS_FILE = CONFIG_DIRECTORY / "default_pipeline.json"
CONFIG_VERSION = 1
STREAM_END = object()
PANEL_COLORS = [QColor("#38bdf8"), QColor("#f97316")]
MODE_SINGLE = "single"
MODE_COMPARISON = "comparison"
MODE_LIVE = "live"
MODE_RECENT = "recent"
MODE_SELECT = "select"
LOGGER = logging.getLogger("contrast.main")


def comparison_report_title(paths: list[Path]) -> str:
    prefixes: list[str] = []
    for path in paths:
        parts = path.stem.rsplit("_", 2)
        if len(parts) != 3 or parts[1].lower() not in {"pre", "post"} or not parts[2].isdigit():
            return "Contrast ROI Residence Comparison"
        prefixes.append(parts[0])
    if len(prefixes) == 2 and prefixes[0] == prefixes[1] and prefixes[0]:
        return f"{prefixes[0].replace('_', ' ')} - Contrast Residence Comparison"
    return "Contrast ROI Residence Comparison"


def _report_frame(panel: VideoPanel, frame_index: int) -> np.ndarray | None:
    encoded_frames = getattr(panel, "report_encoded_frames", None) or panel.enhanced_frames
    if encoded_frames is not None and 0 <= frame_index < len(encoded_frames):
        enhanced = cv2.imdecode(encoded_frames[frame_index], cv2.IMREAD_GRAYSCALE)
        if enhanced is not None:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return None if panel.current_frame is None else panel.current_frame.copy()


def _draw_report_image(
    painter: QPainter,
    frame: np.ndarray,
    roi: QRect | None,
    roi_mask: np.ndarray | None,
    color: QColor,
    target: QRect,
) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()
    scaled = image.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    image_rect = QRect(
        target.x() + (target.width() - scaled.width()) // 2,
        target.y() + (target.height() - scaled.height()) // 2,
        scaled.width(),
        scaled.height(),
    )
    painter.drawImage(image_rect, scaled)
    if roi is None or not roi.isValid():
        return
    x_scale = image_rect.width() / max(1, frame.shape[1])
    y_scale = image_rect.height() / max(1, frame.shape[0])
    painter.setPen(QPen(color, 3, Qt.PenStyle.DotLine))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if roi_mask is None:
        painter.drawRect(QRect(round(image_rect.x() + roi.x() * x_scale), round(image_rect.y() + roi.y() * y_scale), round(roi.width() * x_scale), round(roi.height() * y_scale)))
        return
    contours, _ = cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) >= 3:
            painter.drawPolygon(QPolygon([QPoint(round(image_rect.x() + (roi.x() + int(point[0][0])) * x_scale), round(image_rect.y() + (roi.y() + int(point[0][1])) * y_scale)) for point in contour]))


def _print_optimized_heatmap(heatmap: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(grayscale, (2.0, 98.0))
    if high > low:
        grayscale = np.clip((grayscale.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    grayscale = cv2.LUT(grayscale, np.asarray([round(255.0 * (value / 255.0) ** 0.75) for value in range(256)], dtype=np.uint8))
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def maximum_contrast_frame(results: dict[str, AnalysisResult]) -> int | None:
    peak_frame: int | None = None
    peak_signal = -np.inf
    for result in results.values():
        if not len(result.normalized_signal) or not len(result.time) or result.fps <= 0:
            continue
        signal = np.asarray(result.normalized_signal, dtype=float)
        valid = np.isfinite(signal)
        if not np.any(valid):
            continue
        peak_index = int(np.argmax(np.where(valid, signal, -np.inf)))
        if signal[peak_index] > peak_signal:
            peak_signal = signal[peak_index]
            peak_frame = round(float(result.time[peak_index]) * result.fps)
    return peak_frame


def render_comparison_report(
    path: Path,
    panels: list[VideoPanel],
    results: dict[str, AnalysisResult],
    frame_index: int,
    *,
    print_optimized: bool = True,
) -> bool:
    if len(panels) != 2 or not results:
        return False
    frames = [_report_frame(panel, frame_index) for panel in panels]
    heatmaps = [panel.temporal_change_heatmap for panel in panels]
    if any(frame is None for frame in frames) or any(heatmap is None for heatmap in heatmaps):
        return False
    report_title = comparison_report_title([panel.path for panel in panels])
    writer = QPdfWriter(str(path))
    writer.setResolution(144)
    writer.setTitle(report_title)
    painter = QPainter(writer)
    page = writer.pageLayout().paintRectPixels(writer.resolution())
    margin = 72
    content_width = page.width() - margin * 2
    painter.fillRect(page, QColor("#ffffff"))
    title_font = QFont()
    title_font.setPointSize(14)
    title_font.setBold(True)
    painter.setPen(QColor("#172033"))
    painter.setFont(title_font)
    painter.drawText(margin, margin, report_title)
    body_font = QFont()
    body_font.setPointSize(9)
    painter.setFont(body_font)
    painter.setPen(QColor("#526070"))
    painter.drawText(margin, margin + 28, f"Frame {frame_index + 1}")
    image_top = margin + 54
    image_height = int(content_width * 0.50)
    image_gap = 24
    image_width = (content_width - image_gap) // 2
    for index, (panel, frame) in enumerate(zip(panels, frames, strict=True)):
        x = margin + index * (image_width + image_gap)
        painter.setPen(panel.color)
        painter.drawText(x, image_top - 8, f"{panel.label}: {panel.path.name}")
        _draw_report_image(painter, cast(np.ndarray, frame), panel.roi(), panel.roi_mask(), panel.color, QRect(x, image_top, image_width, image_height))
    heatmap_top = image_top + image_height + 76
    painter.setPen(QColor("#172033"))
    painter.setFont(title_font)
    painter.drawText(margin, heatmap_top - 30, "Contrast Residence Heatmap")
    for index, (panel, heatmap) in enumerate(zip(panels, heatmaps, strict=True)):
        x = margin + index * (image_width + image_gap)
        _draw_report_image(
            painter,
            _print_optimized_heatmap(cast(np.ndarray, heatmap)) if print_optimized else cast(np.ndarray, heatmap),
            None,
            None,
            panel.color,
            QRect(x, heatmap_top, image_width, image_height),
        )
    chart_top = heatmap_top + image_height + 76
    chart_title_gap = 30
    tick_label_height = painter.fontMetrics().height() + 4
    axis_label_gap = 38
    chart_footer_padding = 64
    chart_height = page.bottom() - chart_top - margin - chart_footer_padding
    chart = QRect(margin + 56, chart_top, content_width - 72, chart_height)
    painter.setPen(QColor("#172033"))
    painter.setFont(title_font)
    painter.drawText(margin, chart.y() - chart_title_gap, "ROI Residence (Normalized Signal)")
    painter.setFont(body_font)
    painter.setPen(QPen(QColor("#9aa7b5"), 1))
    painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
    painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())
    max_time = max((float(result.time[-1]) for result in results.values() if len(result.time)), default=1.0)
    max_time = max(max_time, 1.0)
    for tick_index in range(9):
        fraction = tick_index / 8
        y = round(chart.bottom() - chart.height() * fraction)
        painter.setPen(QPen(QColor("#d7dde5"), 1, Qt.PenStyle.DotLine))
        painter.drawLine(chart.left(), y, chart.right(), y)
        painter.setPen(QColor("#526070"))
        painter.drawText(chart.left() - 42, y + 4, f"{fraction:.2f}")
    for tick_index in range(7):
        fraction = tick_index / 6
        x = round(chart.left() + chart.width() * fraction)
        painter.setPen(QPen(QColor("#d7dde5"), 1, Qt.PenStyle.DotLine))
        painter.drawLine(x, chart.top(), x, chart.bottom())
        painter.setPen(QColor("#526070"))
        painter.drawText(QRect(x - 28, chart.bottom() + 4, 56, tick_label_height), Qt.AlignmentFlag.AlignHCenter, f"{max_time * fraction:.1f}")
    peak_time = frame_index / max((result.fps for result in results.values()), default=1.0)
    peak_x = round(chart.left() + chart.width() * peak_time / max_time)
    painter.setPen(QPen(QColor("#e11d48"), 2, Qt.PenStyle.DashLine))
    painter.drawLine(peak_x, chart.top(), peak_x, chart.bottom())
    for panel in panels:
        result = results.get(panel.label)
        if result is None or not len(result.time):
            continue
        points = QPolygon([QPoint(round(chart.left() + chart.width() * float(time) / max_time), round(chart.bottom() - chart.height() * float(signal))) for time, signal in zip(result.time, result.normalized_signal, strict=True)])
        painter.setPen(QPen(panel.color, 2))
        painter.drawPolyline(points)
        painter.setFont(body_font)
        painter.drawText(chart.right() - 180, chart.top() + 16 + panels.index(panel) * 16, panel.label)
    painter.setPen(QColor("#526070"))
    painter.drawText(chart.center().x() - 22, chart.bottom() + axis_label_gap, "Time (s)")
    painter.end()
    return True


class ModeSelectionButton(QPushButton):
    def __init__(self, label: str, preview_mode: str) -> None:
        super().__init__()
        self.setText("")
        self.setObjectName("modeSelectionButton")
        self.setFixedSize(196, 196)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        preview = QFrame(self)
        preview.setObjectName("modePreview")
        if preview_mode in (MODE_RECENT, MODE_SELECT):
            preview.setProperty("actionPreview", True)
        preview.setFixedSize(142, 82)
        preview_layout = QHBoxLayout(preview)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)

        if preview_mode == MODE_SINGLE:
            panel = QFrame(preview)
            panel.setObjectName("modePreviewPane")
            panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            preview_layout.addWidget(panel)
        elif preview_mode == MODE_COMPARISON:
            left_panel = QFrame(preview)
            left_panel.setObjectName("modePreviewPane")
            left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            right_panel = QFrame(preview)
            right_panel.setObjectName("modePreviewPane")
            right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            preview_layout.addWidget(left_panel)
            preview_layout.addWidget(right_panel)
        elif preview_mode == MODE_RECENT:
            self._build_recent_icon_preview(preview_layout)
        else:
            self._build_select_icon_preview(preview_layout)

        label_widget = QLabel(label)
        label_widget.setObjectName("modeSelectionLabel")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        layout.addWidget(preview, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_widget, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

    def _build_recent_icon_preview(self, preview_layout: QHBoxLayout) -> None:
        icon_canvas = ModeActionGlyph("recent", self)

        preview_layout.addStretch()
        preview_layout.addWidget(icon_canvas)
        preview_layout.addStretch()

    def _build_select_icon_preview(self, preview_layout: QHBoxLayout) -> None:
        icon_canvas = ModeActionGlyph("select", self)

        preview_layout.addStretch()
        preview_layout.addWidget(icon_canvas)
        preview_layout.addStretch()


class ModeActionGlyph(QWidget):
    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setObjectName("modeActionGlyph")
        self.setFixedSize(88, 56)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        accent = QColor("#0b1018")
        pen = QPen(accent, 3.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self._kind == "recent":
            # Simple CCW rewind-style arrow.
            arc_rect = QRectF(24, 10, 38, 38)
            painter.drawArc(arc_rect, 210 * 16, 300 * 16)
            arrow = QPolygon([QPoint(20, 27), QPoint(23, 15), QPoint(34, 21)])
            painter.setBrush(accent)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(arrow)
            return

        # Simple folder shape.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(17, 17, 54, 32), 4, 4)
        painter.drawRoundedRect(QRectF(23, 9, 22, 11), 3, 3)


@contextmanager
def frame_parallel_opencv() -> Generator[None]:
    previous_thread_count = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous_thread_count)


class FrameDenoiser(Protocol):
    backend_id: str

    @property
    def device_name(self) -> str: ...

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]: ...

class SynchronizedFrameDenoiser:
    def __init__(self, denoiser: FrameDenoiser) -> None:
        self._denoiser = denoiser
        self._lock = Lock()
        self.backend_id = denoiser.backend_id

    @property
    def device_name(self) -> str:
        return self._denoiser.device_name

    def denoise_batch(self, images: list[np.ndarray], noise_sigma: float) -> list[np.ndarray]:
        with self._lock:
            return self._denoiser.denoise_batch(images, noise_sigma)


@dataclass(slots=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    window_center: float | None = None
    window_width: float | None = None

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass(frozen=True, slots=True)
class SourcePipelineState:
    crop_rect: QRect
    trim_start: int
    auto_crop_rect: QRect | None
    trim_cache_key: tuple[int, int, int, int] | None
    detected_trim_start: int | None
    background_reference: np.ndarray | None
    configuration: tuple[object, ...]
    camera_view_mask: np.ndarray | None = None
    camera_field_mask: np.ndarray | None = None
    frame_brightness_onset: int | None = None
    comparison_sync_offset_frames: int = 0
    trim_end: int = 0
    metal_needle_mask: np.ndarray | None = None
    gain_alignment_baseline: float = 0.0
    gain_alignment_span: float = 0.0
    gain_multiplier: float = 1.0
    gain_offset: float = 0.0
    histogram: np.ndarray | None = None
    histogram_match_lut: np.ndarray | None = None


def align_source_gain_states(states: list[SourcePipelineState]) -> list[SourcePipelineState]:
    target_baseline = max((state.gain_alignment_baseline for state in states), default=0.0)
    target_span = max((state.gain_alignment_span for state in states), default=0.0)
    if target_baseline <= 0.0 or target_span <= 0.0:
        return states
    return [
        replace(state, gain_multiplier=gain, gain_offset=target_baseline - gain * state.gain_alignment_baseline)
        for state in states
        for gain in (min(2.5, max(1.0, target_span / max(1.0, state.gain_alignment_span))),)
    ]


def align_source_histogram_states(states: list[SourcePipelineState]) -> list[SourcePipelineState]:
    if len(states) < 2 or states[0].histogram is None:
        return states
    reference_histogram = states[0].histogram.astype(np.float64, copy=False)
    if reference_histogram.sum() <= 0.0:
        return states
    reference_cdf = np.cumsum(reference_histogram) / reference_histogram.sum()
    aligned = [states[0]]
    for state in states[1:]:
        if state.histogram is None or state.histogram.sum() <= 0.0:
            aligned.append(state)
            continue
        source_cdf = np.cumsum(state.histogram.astype(np.float64, copy=False)) / state.histogram.sum()
        lut = np.searchsorted(reference_cdf, source_cdf, side="left").clip(0, 255).astype(np.uint8)
        aligned.append(replace(state, histogram_match_lut=lut))
    return aligned


def align_source_brightness_onsets(states: list[SourcePipelineState]) -> list[SourcePipelineState]:
    if len(states) < 2 or states[0].frame_brightness_onset is None:
        return states
    reference_position = states[0].frame_brightness_onset - states[0].trim_start
    aligned = [states[0]]
    for state in states[1:]:
        if state.frame_brightness_onset is None:
            aligned.append(state)
            continue
        trim_start = max(
            0,
            min(
                state.frame_brightness_onset - reference_position + state.comparison_sync_offset_frames,
                state.frame_brightness_onset,
            ),
        )
        aligned.append(replace(state, trim_start=trim_start))
    return aligned


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


@dataclass(frozen=True, slots=True)
class ROIAlignmentPlotResult:
    time: np.ndarray
    roi_brightness_before: np.ndarray
    roi_brightness_after: np.ndarray
    roi_baseline_before: float
    needle_baseline_before: float
    roi_baseline_after: float
    needle_baseline_after: float


@dataclass(frozen=True, slots=True)
class AnalysisPanelSnapshot:
    label: str
    path: Path
    fps: float
    roi: QRect | None
    roi_mask: np.ndarray | None
    parent_vessel_roi: tuple[int, int, float] | None
    background_roi: tuple[int, int] | None
    camera_view_mask: np.ndarray | None
    encoded_frames: tuple[np.ndarray, ...] | None
    source_frames: tuple[np.ndarray, ...] | None


@dataclass(frozen=True, slots=True)
class PipelineAnalysisRequest:
    generation: int
    panels: tuple[AnalysisPanelSnapshot, ...]
    threshold: float
    gain_corrected: bool
    roi_residence: bool
    frame_brightness: bool
    needle_segmentation: bool
    parent_vessel_roi: bool
    background_roi: bool
    temporal_change: bool
    heatmap_colormap: str


@dataclass(frozen=True, slots=True)
class PipelineAnalysisResult:
    generation: int
    roi_results: dict[str, AnalysisResult]
    frame_brightness_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    needle_brightness_results: dict[str, tuple[np.ndarray, np.ndarray]]
    needle_brightness_baselines: dict[str, float]
    needle_masks: dict[str, np.ndarray | None]
    parent_vessel_roi_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    background_roi_results: dict[str, tuple[np.ndarray, np.ndarray]]
    temporal_change_results: dict[str, tuple[np.ndarray, np.ndarray]]
    temporal_change_heatmaps: dict[str, np.ndarray]


def average_frame_brightness(
    frames: list[np.ndarray],
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    if camera_view_mask is None:
        return np.asarray([float(np.mean(frame)) for frame in frames], dtype=float)
    selected = camera_view_mask.astype(bool)
    if not np.any(selected) or any(frame.shape != selected.shape for frame in frames):
        raise ValueError("Frame brightness requires a non-empty camera-view mask matching the frames.")
    return np.asarray([float(np.mean(frame[selected])) for frame in frames], dtype=float)


def detect_fluoroscope_stabilization_frame(
    gray_frames: list[np.ndarray],
    fps: float,
    camera_view_mask: np.ndarray | None = None,
) -> int:
    """Return the first frame after initial fluoroscope gain/offset settling."""
    if len(gray_frames) < 3:
        return 0
    levels = average_frame_brightness(gray_frames, camera_view_mask)
    scan_count = min(len(levels), max(3, round(max(fps, 1.0) * 8.0)))
    startup_levels = levels[:scan_count]
    reference_window = max(3, round(max(fps, 1.0)))
    reference = float(np.median(startup_levels[-reference_window:]))
    frame_noise = float(np.median(np.abs(np.diff(startup_levels[-reference_window:]))))
    tolerance = max(0.25, frame_noise * 6.0)
    if np.max(np.abs(startup_levels - reference)) <= tolerance:
        return 0
    for frame_index in range(scan_count - reference_window + 1):
        if np.max(np.abs(startup_levels[frame_index:] - reference)) <= tolerance:
            return frame_index
    return 0


def temporal_derivative(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(time) != len(values):
        raise ValueError("Time and value arrays must have the same length")
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values.astype(float), time.astype(float))


def baseline_to_apex_normalized_signal(result: AnalysisResult) -> np.ndarray:
    if result.peak_signal <= 0:
        return np.zeros_like(result.contrast_signal)
    return result.contrast_signal / result.peak_signal


@dataclass(frozen=True, slots=True)
class ROISelection:
    rect: QRect
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        rect = QRect(self.rect).normalized()
        object.__setattr__(self, "rect", rect)
        if self.mask is not None:
            expected_shape = (rect.height(), rect.width())
            if self.mask.shape != expected_shape:
                raise ValueError(f"ROI mask shape {self.mask.shape} does not match ROI rect {expected_shape}")
            object.__setattr__(self, "mask", self.mask.astype(bool, copy=True))

    def contains(self, x: int, y: int) -> bool:
        if not self.rect.contains(x, y):
            return False
        if self.mask is None:
            return True
        return bool(self.mask[y - self.rect.y(), x - self.rect.x()])

    def width(self) -> int:
        return self.rect.width()

    def height(self) -> int:
        return self.rect.height()


def format_seconds(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.2f} s"


def probe_video(path: Path) -> VideoInfo:
    if is_dicom_video(path):
        info = probe_dicom_video(path)
        return VideoInfo(
            path=path,
            fps=info.fps,
            frame_count=info.frame_count,
            width=info.width,
            height=info.height,
            window_center=info.window_center,
            window_width=info.window_width,
        )

    capture = open_video_capture(path)
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


def _contrast_frame_signal(gray: np.ndarray) -> float:
    height, width = gray.shape
    center = gray[height * 3 // 10 : height * 7 // 10, width * 3 // 10 : width * 7 // 10]
    if center.size == 0:
        center = gray
    return float(0.6 * np.mean(center) + 0.4 * np.percentile(center, 30))


def detect_pre_injection_trim_start(
    gray_frames: list[np.ndarray],
    fps: float,
    gain_normalization_enabled: bool = True,
    camera_view_mask: np.ndarray | None = None,
) -> int:
    if len(gray_frames) < 3:
        return 0

    if gain_normalization_enabled:
        selected = None if camera_view_mask is None else camera_view_mask.astype(bool)
        target_median = float(
            np.median([
                np.median(gray) if selected is None else np.median(gray[selected])
                for gray in gray_frames
            ])
        )
        source_frames = [
            stabilize_frame_gain(gray, target_median, 0.70, 1.45, camera_view_mask)
            for gray in gray_frames
        ]
    else:
        source_frames = gray_frames
    signals = average_frame_brightness(source_frames, camera_view_mask)
    smoothed = smooth_temporal_signal(signals, fps)
    baseline_window = min(len(smoothed), max(3, round(fps * 1.5)))
    baseline = float(np.median(smoothed[:baseline_window]))
    baseline_noise = float(np.median(np.abs(smoothed[:baseline_window] - baseline)))
    sustain_window = max(3, round(fps * 0.35))
    drop_threshold = max(2.0, baseline_noise * 6.0)
    threshold = baseline - drop_threshold

    onset_index = None
    search_end = max(baseline_window, len(smoothed) - sustain_window + 1)
    for index in range(baseline_window, search_end):
        if np.all(smoothed[index : index + sustain_window] <= threshold):
            onset_index = index
            break

    if onset_index is None:
        gradient = np.diff(smoothed)
        if len(gradient) > baseline_window:
            onset_index = int(np.argmin(gradient[baseline_window:]) + baseline_window + 1)
        else:
            onset_index = baseline_window

    return max(0, onset_index - round(fps * 0.5))


def detect_frame_brightness_onset(
    gray_frames: list[np.ndarray],
    fps: float,
    camera_view_mask: np.ndarray | None = None,
) -> int | None:
    if len(gray_frames) < 3:
        return None
    brightness = smooth_temporal_signal(average_frame_brightness(gray_frames, camera_view_mask), fps)
    baseline_window = min(len(brightness), max(3, round(fps * 1.5)))
    baseline = float(np.median(brightness[:baseline_window]))
    baseline_noise = float(np.median(np.abs(brightness[:baseline_window] - baseline)))
    threshold = baseline - max(0.5, baseline_noise * 8.0)
    sustain_window = max(3, round(fps * 0.35))
    for index in range(baseline_window, len(brightness) - sustain_window + 1):
        if np.all(brightness[index : index + sustain_window] <= threshold):
            return index
    return None


def _detect_fluoroscope_field_mask(gray_frames: list[np.ndarray]) -> np.ndarray | None:
    if not gray_frames:
        return None
    height, width = gray_frames[0].shape
    if any(frame.shape != (height, width) for frame in gray_frames):
        return None
    temporal_level = np.percentile(np.stack(gray_frames, axis=0), 75, axis=0).astype(np.uint8)
    center = temporal_level[height // 3 : height * 2 // 3, width // 3 : width * 2 // 3]
    edge_depth = max(4, min(height, width) // 32)
    edges = np.concatenate(
        (
            temporal_level[:edge_depth].ravel(),
            temporal_level[-edge_depth:].ravel(),
            temporal_level[:, :edge_depth].ravel(),
            temporal_level[:, -edge_depth:].ravel(),
        )
    )
    edge_level = float(np.median(edges))
    center_level = float(np.median(center))
    contrast = center_level - edge_level
    if abs(contrast) < 12.0:
        return None

    threshold = edge_level + contrast * 0.45
    field_mask = (
        temporal_level > threshold if contrast > 0 else temporal_level < threshold
    ).astype(np.uint8)
    kernel_size = max(5, round(min(height, width) * 0.03) | 1)
    field_mask = cv2.morphologyEx(
        field_mask,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(field_mask)
    if component_count < 2:
        return None
    component = int(labels[height // 2, width // 2])
    if component == 0:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    field_area = int(stats[component, cv2.CC_STAT_AREA])
    field_width = int(stats[component, cv2.CC_STAT_WIDTH])
    field_height = int(stats[component, cv2.CC_STAT_HEIGHT])
    bounding_area = field_width * field_height
    fill_fraction = field_area / max(1, bounding_area)
    if field_area < width * height * 0.2 or not 0.55 <= fill_fraction <= 0.90:
        return None

    return (labels == component).astype(np.uint8)


def camera_view_mask_for_crop(
    field_mask: np.ndarray | None,
    crop_rect: QRect,
    edge_margin_ratio: float = 0.01,
) -> np.ndarray:
    shape = (max(1, crop_rect.height()), max(1, crop_rect.width()))
    if field_mask is None:
        fallback = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(
            fallback,
            (shape[1] // 2, shape[0] // 2),
            (max(1, shape[1] // 2 - 2), max(1, shape[0] // 2 - 2)),
            0,
            0,
            360,
            1,
            thickness=-1,
        )
        return fallback.astype(bool)
    padded = np.pad((field_mask > 0).astype(np.uint8), 1, mode="constant")
    field_distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    cropped_distance = crop_frame(field_distance, crop_rect)
    if cropped_distance.shape != shape:
        return np.ones(shape, dtype=bool)
    edge_margin = max(2.0, min(shape) * max(0.0, float(edge_margin_ratio)))
    interior = cropped_distance > edge_margin
    if np.any(interior):
        return interior
    return crop_frame(field_mask, crop_rect).astype(bool)


def _detect_aligned_field_crop(gray_frames: list[np.ndarray]) -> QRect | None:
    component_mask = _detect_fluoroscope_field_mask(gray_frames)
    if component_mask is None:
        return None
    return _aligned_field_crop_from_mask(component_mask)


def _aligned_field_crop_from_mask(component_mask: np.ndarray) -> QRect | None:
    height, width = component_mask.shape

    distance = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    center_y, center_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    points = cv2.findNonZero(component_mask)
    if points is None:
        return None
    _field_x, _field_y, field_width, field_height = cv2.boundingRect(points)
    alignment = 32
    maximum_size = min(field_width, field_height) // alignment * alignment
    minimum_size = max(alignment, int(min(width, height) * 0.45) // alignment * alignment)
    for size in range(maximum_size, minimum_size - 1, -alignment):
        x = max(0, min(width - size, round(center_x - size / 2)))
        y = max(0, min(height - size, round(center_y - size / 2)))
        occupancy = float(np.mean(component_mask[y : y + size, x : x + size]))
        if occupancy >= 0.995:
            return QRect(x, y, size, size)
    return None


def _adjust_auto_crop_square(crop_rect: QRect, width: int, height: int, size_offset: int) -> QRect:
    """Resize an auto-crop around its center while retaining an in-frame square."""
    alignment = 32
    maximum_size = min(width, height) // alignment * alignment
    if maximum_size <= 0:
        size = max(1, min(width, height))
        return QRect(0, 0, size, size)

    base_size = min(crop_rect.width(), crop_rect.height()) // alignment * alignment
    size = max(alignment, min(maximum_size, base_size + size_offset))
    center_x = crop_rect.x() + crop_rect.width() / 2
    center_y = crop_rect.y() + crop_rect.height() / 2
    x = max(0, min(width - size, round(center_x - size / 2)))
    y = max(0, min(height - size, round(center_y - size / 2)))
    return QRect(x, y, size, size)


def _detect_pillarbox_crop(gray_frames: list[np.ndarray], width: int, height: int) -> QRect:
    full_frame = QRect(0, 0, width, height)
    column_profiles = [np.percentile(gray, 90, axis=0).astype(np.float32) for gray in gray_frames]
    profile = np.median(np.stack(column_profiles, axis=0), axis=0)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    smooth = np.convolve(profile, kernel / np.sum(kernel), mode="same")

    center_start = width // 4
    center_end = (width * 3) // 4
    center_slice = smooth[center_start:center_end]
    center_level = float(np.median(center_slice)) if center_slice.size else float(np.median(smooth))
    edge_width = max(1, width // 12)
    edge_left = float(np.median(smooth[:edge_width]))
    edge_right = float(np.median(smooth[-edge_width:]))
    edge_level = min(edge_left, edge_right)
    threshold = max(3.0, edge_level + (center_level - edge_level) * 0.35)
    content = smooth > threshold
    if not np.any(content):
        return full_frame

    left = max(0, int(np.argmax(content)) - 2)
    right = min(width - 1, int(width - 1 - np.argmax(content[::-1])) + 2)
    cropped_width = right - left + 1
    if cropped_width < int(width * 0.45) or left < 2 or width - right - 1 < 2:
        return full_frame
    return QRect(left, 0, cropped_width, height)


def detect_fluoroscope_crop_from_frames(
    gray_frames: list[np.ndarray],
    width: int,
    height: int,
) -> QRect:
    return detect_fluoroscope_geometry_from_frames(gray_frames, width, height)[0]


def detect_fluoroscope_geometry_from_frames(
    gray_frames: list[np.ndarray],
    width: int,
    height: int,
) -> tuple[QRect, np.ndarray]:
    full_frame = QRect(0, 0, width, height)
    field_mask = _detect_fluoroscope_field_mask(gray_frames) if gray_frames else None
    field_crop = _aligned_field_crop_from_mask(field_mask) if field_mask is not None else None
    if field_crop is not None:
        return field_crop, field_mask.astype(bool)
    pillarbox_crop = _detect_pillarbox_crop(gray_frames, width, height) if gray_frames else full_frame
    fallback_mask = np.zeros((height, width), dtype=np.uint8)
    x, y, crop_width, crop_height = pillarbox_crop.getRect()
    cv2.ellipse(
        fallback_mask,
        (x + crop_width // 2, y + crop_height // 2),
        (max(1, crop_width // 2 - 2), max(1, crop_height // 2 - 2)),
        0,
        0,
        360,
        1,
        thickness=-1,
    )
    return pillarbox_crop, fallback_mask.astype(bool)


def detect_fluoroscope_crop(
    path: Path,
    info: VideoInfo,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> QRect:
    return detect_fluoroscope_geometry(path, info, progress_callback)[0]


def detect_fluoroscope_geometry(
    path: Path,
    info: VideoInfo,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> tuple[QRect, np.ndarray]:
    full_frame = QRect(0, 0, info.width, info.height)
    if info.width <= 0 or info.height <= 0:
        return full_frame, np.ones((max(1, info.height), max(1, info.width)), dtype=bool)

    capture = open_video_capture(path)
    if not capture.isOpened():
        return full_frame, np.ones((info.height, info.width), dtype=bool)

    try:
        sample_count = min(24, max(6, info.frame_count if info.frame_count > 0 else 6))
        frame_indexes = np.unique(np.linspace(0, max(0, info.frame_count - 1), num=sample_count, dtype=int))
        gray_frames: list[np.ndarray] = []
        for sample_index, frame_index in enumerate(frame_indexes):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                continue
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if progress_callback is not None and not progress_callback(sample_index + 1, len(frame_indexes)):
                return full_frame, np.ones((info.height, info.width), dtype=bool)

        return detect_fluoroscope_geometry_from_frames(gray_frames, info.width, info.height)
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


def sanitize_camera_view(frame: np.ndarray, camera_view_mask: np.ndarray | None) -> np.ndarray:
    if camera_view_mask is None:
        return frame
    selected = camera_view_mask.astype(bool)
    if selected.shape != frame.shape or not np.any(selected):
        raise ValueError("Camera-view sanitization requires a non-empty mask matching the frame.")
    if np.all(selected):
        return frame
    sanitized = frame.copy()
    sanitized[~selected] = np.median(frame[selected])
    return sanitized


def estimate_video_median(
    path: Path,
    crop_rect: QRect,
    start_frame: int,
    frame_count: int,
    camera_view_mask: np.ndarray | None = None,
) -> float:
    capture = open_video_capture(path)
    try:
        medians: list[float] = []
        if frame_count <= 0:
            return 128.0
        sample_indexes = np.unique(np.linspace(0, max(0, frame_count - 1), num=min(24, max(1, frame_count)), dtype=int))
        for frame_index in sample_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(frame_index))
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY)
                pixels = gray.ravel() if camera_view_mask is None else gray[camera_view_mask.astype(bool)]
                medians.append(float(np.median(pixels)))
        return float(np.median(medians)) if medians else 128.0
    finally:
        capture.release()


def estimate_video_contrast_response(
    path: Path,
    crop_rect: QRect,
    start_frame: int,
    frame_count: int,
    fps: float,
    camera_view_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    capture = open_video_capture(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        signals: list[float] = []
        for _ in range(frame_count):
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY)
                gray = sanitize_camera_view(gray, camera_view_mask)
                signals.append(_contrast_frame_signal(gray))
            else:
                break
        if not signals:
            return 128.0, 1.0
        pre_injection_count = max(3, min(round(fps * 0.4), len(signals)))
        baseline = float(np.median(signals[:pre_injection_count]))
        dark_level = float(np.min(smooth_temporal_signal(np.asarray(signals), fps)))
        return baseline, max(1.0, baseline - dark_level)
    finally:
        capture.release()


def estimate_video_histogram(
    path: Path,
    crop_rect: QRect,
    start_frame: int,
    frame_count: int,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    capture = open_video_capture(path)
    try:
        histogram = np.zeros(256, dtype=np.int64)
        sample_indexes = np.unique(np.linspace(0, max(0, frame_count - 1), num=min(24, max(1, frame_count)), dtype=int))
        for frame_index in sample_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(frame_index))
            ok, frame = capture.read()
            if ok:
                gray = cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY)
                pixels = gray.ravel() if camera_view_mask is None else gray[camera_view_mask.astype(bool)]
                histogram += np.bincount(pixels, minlength=256)
        return histogram
    finally:
        capture.release()


def stabilize_frame_gain(
    gray: np.ndarray,
    target_median: float,
    min_gain: float,
    max_gain: float,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    pixels = gray.ravel() if camera_view_mask is None else gray[camera_view_mask.astype(bool)]
    current_median = max(1.0, float(np.median(pixels)))
    gain = float(np.clip(target_median / current_median, min_gain, max_gain))
    return np.clip(gray.astype(np.float32) * gain, 0, 255)


def _fit_robust_intensity_map(current: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    current_values = current.astype(np.float64, copy=False).ravel()
    reference_values = reference.astype(np.float64, copy=False).ravel()
    keep = np.isfinite(current_values) & np.isfinite(reference_values)
    gain = 1.0
    offset = 0.0

    for _ in range(4):
        kept_current = current_values[keep]
        kept_reference = reference_values[keep]
        if kept_current.size < 24:
            break
        current_center = float(np.mean(kept_current))
        reference_center = float(np.mean(kept_reference))
        centered_current = kept_current - current_center
        denominator = float(np.dot(centered_current, centered_current))
        if denominator < 1e-6:
            gain = 1.0
            offset = float(np.median(kept_reference - kept_current))
            break
        gain = float(np.dot(centered_current, kept_reference - reference_center) / denominator)
        offset = reference_center - gain * current_center
        residual = reference_values - (gain * current_values + offset)
        residual_center = float(np.median(residual[keep]))
        residual_scale = 1.4826 * float(np.median(np.abs(residual[keep] - residual_center))) + 0.5
        next_keep = np.abs(residual - residual_center) <= 2.8 * residual_scale
        if int(np.count_nonzero(next_keep)) < 24:
            break
        keep = next_keep

    clipped_gain = float(np.clip(gain, 0.75, 1.33))
    if clipped_gain != gain:
        offset = float(np.median(reference_values[keep] - clipped_gain * current_values[keep]))
    return clipped_gain, float(np.clip(offset, -48.0, 48.0))


def estimate_intensity_corrections(
    gray_frames: list[np.ndarray],
    analysis_size: int = 192,
    quantile_count: int = 48,
    camera_view_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not gray_frames:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    frame_height, frame_width = gray_frames[0].shape
    scale = min(1.0, float(analysis_size) / max(frame_height, frame_width))
    analysis_width = max(16, int(round(frame_width * scale)))
    analysis_height = max(16, int(round(frame_height * scale)))
    analysis_mask = None
    if camera_view_mask is not None:
        if camera_view_mask.shape != (frame_height, frame_width) or not np.any(camera_view_mask):
            raise ValueError("Brightness stabilization requires a non-empty camera-view mask matching the frames.")
        analysis_mask = cv2.resize(
            camera_view_mask.astype(np.uint8),
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    percentiles = np.linspace(70.0, 98.0, num=max(24, quantile_count), dtype=np.float32)
    frame_quantiles = np.empty((len(gray_frames), percentiles.size), dtype=np.float32)
    for index, frame in enumerate(gray_frames):
        analysis_frame = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
        pixels = analysis_frame.ravel() if analysis_mask is None else analysis_frame[analysis_mask]
        frame_quantiles[index] = np.percentile(pixels, percentiles)

    reference_quantiles = np.median(frame_quantiles, axis=0)
    gains = np.empty(len(gray_frames), dtype=np.float32)
    offsets = np.empty(len(gray_frames), dtype=np.float32)
    for index, current_quantiles in enumerate(frame_quantiles):
        gains[index], offsets[index] = _fit_robust_intensity_map(
            current_quantiles,
            reference_quantiles,
        )
    return gains, offsets


def stabilize_frame_intensity(gray: np.ndarray, gain: float, offset: float) -> np.ndarray:
    stabilized = gray.astype(np.float32) * float(gain) + float(offset)
    return np.clip(stabilized, 0, 255).astype(np.uint8)


def _camera_registration_image(
    gray: np.ndarray,
    analysis_width: int,
    analysis_height: int,
) -> np.ndarray:
    resized = cv2.resize(gray, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    low_frequency = cv2.GaussianBlur(resized, (0, 0), sigmaX=3.0, sigmaY=3.0)
    return resized.astype(np.float32) - low_frequency.astype(np.float32)


def estimate_camera_translations(
    gray_frames: list[np.ndarray],
    max_shift: float = 12.0,
    analysis_size: int = 320,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not gray_frames:
        return np.empty((0, 2), dtype=np.float32)
    frame_height, frame_width = gray_frames[0].shape
    if any(frame.shape != (frame_height, frame_width) for frame in gray_frames):
        raise ValueError("Camera motion stabilization requires consistently sized grayscale frames.")

    scale = min(1.0, float(analysis_size) / max(frame_height, frame_width))
    analysis_width = max(32, int(round(frame_width * scale)))
    analysis_height = max(32, int(round(frame_height * scale)))
    analysis_mask = None
    if camera_view_mask is not None:
        if camera_view_mask.shape != (frame_height, frame_width) or not np.any(camera_view_mask):
            raise ValueError("Camera motion stabilization requires a non-empty camera-view mask matching the frames.")
        resized_mask = cv2.resize(
            camera_view_mask.astype(np.uint8),
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_NEAREST,
        )
        erosion_radius = max(2, int(round(max_shift * scale)) + 1)
        kernel_size = erosion_radius * 2 + 1
        analysis_mask = cv2.erode(
            resized_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(np.float32)

    reference_count = min(len(gray_frames), max(1, min(15, len(gray_frames) // 10)))
    reference_gray = np.median(np.stack(gray_frames[:reference_count]), axis=0).astype(np.uint8)
    reference = _camera_registration_image(
        reference_gray,
        analysis_width,
        analysis_height,
    )
    tile_mask = np.ones((analysis_height, analysis_width), dtype=np.float32)
    if analysis_mask is not None:
        tile_mask = analysis_mask
    tiles: list[tuple[slice, slice, np.ndarray, np.ndarray]] = []
    tile_count = 4
    for row in range(tile_count):
        y_slice = slice(row * analysis_height // tile_count, (row + 1) * analysis_height // tile_count)
        for column in range(tile_count):
            x_slice = slice(
                column * analysis_width // tile_count,
                (column + 1) * analysis_width // tile_count,
            )
            local_mask = tile_mask[y_slice, x_slice] > 0
            local_reference = reference[y_slice, x_slice]
            if float(np.mean(local_mask)) < 0.72 or float(np.std(local_reference[local_mask])) < 1.0:
                continue
            tiles.append(
                (
                    y_slice,
                    x_slice,
                    local_mask.astype(np.float32),
                    cv2.createHanningWindow(
                        (local_reference.shape[1], local_reference.shape[0]),
                        cv2.CV_32F,
                    ),
                )
            )
    if len(tiles) < 3:
        return np.zeros((len(gray_frames), 2), dtype=np.float32)

    translations = np.zeros((len(gray_frames), 2), dtype=np.float32)
    scaled_limit = max(1.0, float(max_shift) * scale)
    for index, frame in enumerate(gray_frames):
        current = _camera_registration_image(
            frame,
            analysis_width,
            analysis_height,
        )
        estimates: list[tuple[float, float]] = []
        for y_slice, x_slice, local_mask, window in tiles:
            shift, response = cv2.phaseCorrelate(
            reference[y_slice, x_slice] * local_mask,
            current[y_slice, x_slice] * local_mask,
                window,
            )
            if response >= 0.05 and abs(shift[0]) <= scaled_limit and abs(shift[1]) <= scaled_limit:
                estimates.append(shift)
        if not estimates:
            continue
        values = np.asarray(estimates, dtype=np.float32)
        center = np.median(values, axis=0)
        residuals = np.linalg.norm(values - center, axis=1)
        inliers = residuals <= 0.75
        minimum_consensus = max(3, len(tiles) // 3)
        if int(np.count_nonzero(inliers)) < minimum_consensus:
            continue
        center = np.median(values[inliers], axis=0)
        translation = -center / scale
        if float(np.linalg.norm(translation)) >= 0.75:
            translations[index] = translation

    for index, translation in enumerate(translations.copy()):
        magnitude = float(np.linalg.norm(translation))
        if magnitude == 0.0 or magnitude >= 2.0:
            continue
        neighbor_start = max(0, index - 1)
        neighboring_translations = translations[neighbor_start : index + 2]
        has_consistent_neighbor = any(
            neighbor_start + offset != index and float(np.linalg.norm(neighbor)) > 0.0
            and float(np.linalg.norm(neighbor - translation)) <= 0.75
            for offset, neighbor in enumerate(neighboring_translations)
        )
        if not has_consistent_neighbor:
            translations[index] = 0.0
    return translations


def stabilize_frame_translation(gray: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if float(np.linalg.norm(translation)) < 1e-6:
        return gray.copy()
    transform = np.asarray(
        [[1.0, 0.0, float(translation[0])], [0.0, 1.0, float(translation[1])]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        gray,
        transform,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def correct_scanlines(gray: np.ndarray, bias_clip: float, sigma_y: float) -> np.ndarray:
    corrected = gray.astype(np.float32)
    vertical_smooth = cv2.GaussianBlur(corrected, (1, 9), sigmaX=0, sigmaY=sigma_y)
    row_bias = np.median(corrected - vertical_smooth, axis=1)
    row_bias -= np.median(row_bias)
    corrected -= np.clip(row_bias, -bias_clip, bias_clip)[:, np.newaxis]
    return np.clip(corrected, 0, 255).astype(np.uint8)


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


def reduce_quantum_mottle(
    frames: list[np.ndarray],
    similarity_sigma: float,
) -> np.ndarray:
    if not frames:
        raise ValueError("Quantum mottle reduction requires at least one frame.")

    center_index = len(frames) // 2
    center = frames[center_index].astype(np.float32)
    weighted_sum = center.copy()
    weight_sum = np.ones_like(center)
    safe_sigma = max(0.1, float(similarity_sigma))
    for index, frame in enumerate(frames):
        if index == center_index:
            continue
        neighbor = frame.astype(np.float32)
        weight = np.exp(-np.abs(neighbor - center) / safe_sigma)
        weighted_sum += neighbor * weight
        weight_sum += weight
    return np.clip(np.rint(weighted_sum / weight_sum), 0, 255).astype(np.uint8)


def enhance_local_contrast(gray: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def smooth_final_frame(gray: np.ndarray, sigma_x: float) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_x)


def apply_image_adjustments(
    gray: np.ndarray,
    brightness_offset: int,
    contrast_gain: float,
    sharpen_amount: float,
    gamma: float,
) -> np.ndarray:
    adjusted = gray.astype(np.float32)
    adjusted = adjusted * float(contrast_gain) + float(brightness_offset)

    safe_gamma = max(0.1, float(gamma))
    if abs(safe_gamma - 1.0) > 1e-4:
        normalized = np.clip(adjusted, 0, 255) / 255.0
        adjusted = np.power(normalized, 1.0 / safe_gamma).astype(np.float32) * 255.0

    amount = max(0.0, float(sharpen_amount))
    if amount > 1e-4:
        blurred = cv2.GaussianBlur(adjusted, (0, 0), sigmaX=1.0, sigmaY=1.0)
        adjusted = cv2.addWeighted(adjusted, 1.0 + amount, blurred, -amount, 0.0)

    return np.clip(adjusted, 0, 255).astype(np.uint8)


ROI_VESSEL_LABEL = 96
ROI_ANEURYSM_LABEL = 224


def compute_temporal_change_map(
    gray_frames: list[np.ndarray],
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not gray_frames:
        return np.zeros((1, 1), dtype=np.float32)

    source_frames = [np.clip(frame, 0, 255).astype(np.uint8) for frame in gray_frames]
    if len(source_frames) == 1:
        return np.zeros_like(source_frames[0], dtype=np.float32)

    stack = np.stack(source_frames, axis=0).astype(np.float32)
    baseline_count = min(len(source_frames) - 1, max(3, min(24, len(source_frames) // 8)))
    baseline = np.percentile(stack[:baseline_count], 75.0, axis=0)
    darkest = np.percentile(stack, 10.0, axis=0)
    temporal_change = np.clip(baseline - darkest, 0.0, None)
    if camera_view_mask is not None:
        if camera_view_mask.shape != temporal_change.shape:
            raise ValueError("Temporal segmentation requires a camera-view mask matching the frames.")
        temporal_change[~camera_view_mask.astype(bool)] = 0.0
    return temporal_change


def detect_stationary_metal_mask(
    gray_frames: list[np.ndarray],
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Find persistent dark metal objects while rejecting moving dark contrast."""
    if len(gray_frames) < 3:
        return None
    shape = gray_frames[0].shape
    if any(frame.shape != shape for frame in gray_frames):
        raise ValueError("Metal needle detection requires consistently sized frames.")
    if camera_view_mask is not None and camera_view_mask.shape != shape:
        raise ValueError("Metal needle detection requires a camera-view mask matching the frames.")

    stack = np.stack(gray_frames, axis=0).astype(np.float32)
    median_intensity = np.median(stack, axis=0)
    mean_frame_change = np.mean(np.abs(np.diff(stack, axis=0)), axis=0)
    candidates = ((median_intensity <= 40.0) & (mean_frame_change <= 5.0)).astype(np.uint8)
    if camera_view_mask is not None:
        candidates &= camera_view_mask.astype(np.uint8)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    mask = np.zeros(shape, dtype=np.uint8)
    for label in range(1, component_count):
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width / max(1, height), height / max(1, width))
        if stats[label, cv2.CC_STAT_AREA] >= 256 and aspect_ratio >= 2.0:
            mask[labels == label] = 255
    if not np.any(mask):
        return None
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    if camera_view_mask is not None:
        mask[~camera_view_mask.astype(bool)] = 0
    return mask if np.any(mask) else None


def segment_pre_injection_needle(
    gray_frames: list[np.ndarray],
    fps: float,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Segment the single darkest connected region using only pre-injection frames."""
    if len(gray_frames) < 3:
        return None
    shape = gray_frames[0].shape
    if any(frame.shape != shape for frame in gray_frames):
        raise ValueError("Needle segmentation requires consistently sized frames.")
    if camera_view_mask is not None and camera_view_mask.shape != shape:
        raise ValueError("Needle segmentation requires a camera-view mask matching the frames.")

    pre_injection_count = min(
        len(gray_frames),
        max(
            3,
            detect_pre_injection_trim_start(
                gray_frames,
                fps,
                camera_view_mask=camera_view_mask,
            )
            + max(1, round(float(fps) * 0.5)),
        ),
    )
    temporal_median = np.median(
        np.stack(gray_frames[:pre_injection_count], axis=0),
        axis=0,
    ).astype(np.uint8)
    valid_pixels = (
        temporal_median[camera_view_mask.astype(bool)]
        if camera_view_mask is not None
        else temporal_median.ravel()
    )
    if valid_pixels.size == 0:
        return None
    otsu_threshold, _ = cv2.threshold(
        valid_pixels,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    darkest_level = float(np.percentile(valid_pixels, 0.5))
    candidates = (temporal_median <= min(float(otsu_threshold), darkest_level)).astype(np.uint8)
    if camera_view_mask is not None:
        candidates &= camera_view_mask.astype(np.uint8)
    candidates = cv2.morphologyEx(
        candidates,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    if camera_view_mask is not None:
        candidates &= camera_view_mask.astype(np.uint8)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    minimum_area = max(8, round(temporal_median.size * 0.00002))
    maximum_area = max(minimum_area, round(temporal_median.size * 0.25))
    components: list[tuple[float, int, int]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not minimum_area <= area <= maximum_area:
            continue
        points_yx = np.column_stack(np.nonzero(labels == label))
        oriented_width, oriented_height = cv2.minAreaRect(points_yx[:, ::-1].astype(np.float32))[1]
        elongation = max(oriented_width, oriented_height) / max(1.0, min(oriented_width, oriented_height))
        if elongation < 2.0:
            continue
        component_pixels = temporal_median[labels == label]
        components.append((float(np.mean(component_pixels)), -area, label))
    if not components:
        return None

    selected_label = min(components)[2]
    mask = np.zeros(shape, dtype=np.uint8)
    mask[labels == selected_label] = 255
    return mask


def masked_average_brightness(gray_frames: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    selected = mask > 0
    if not np.any(selected):
        raise ValueError("Needle brightness requires a non-empty segmentation mask.")
    if any(frame.shape != mask.shape for frame in gray_frames):
        raise ValueError("Needle brightness requires frames matching the segmentation mask.")
    return np.asarray([float(np.mean(frame[selected])) for frame in gray_frames], dtype=float)


def masked_affine_average_brightness(
    gray_frames: list[np.ndarray],
    mask: np.ndarray,
    gain: float,
    offset: float,
) -> np.ndarray:
    selected = mask > 0
    if not np.any(selected):
        raise ValueError("Aligned brightness requires a non-empty mask.")
    if any(frame.shape != mask.shape for frame in gray_frames):
        raise ValueError("Aligned brightness requires frames matching the mask.")
    return np.asarray(
        [
            float(np.mean(np.clip(frame[selected].astype(np.float32) * gain + offset, 0.0, 255.0)))
            for frame in gray_frames
        ],
        dtype=float,
    )


def needle_core_mask(mask: np.ndarray) -> np.ndarray:
    selected = (mask > 0).astype(np.uint8)
    if not np.any(selected):
        raise ValueError("Needle brightness requires a non-empty segmentation mask.")
    distance = cv2.distanceTransform(selected, cv2.DIST_L2, 5)
    return (distance >= float(np.max(distance)) * 0.55).astype(np.uint8) * 255


def needle_average_brightness(gray_frames: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Measure the stable metal core without subpixel contamination at mask edges."""
    return masked_average_brightness(gray_frames, needle_core_mask(mask))


def intensity_alignment_parameters(
    source_roi_level: float,
    source_needle_level: float,
    target_roi_level: float,
    target_needle_level: float,
) -> tuple[float, float]:
    source_span = float(source_roi_level) - float(source_needle_level)
    target_span = float(target_roi_level) - float(target_needle_level)
    if abs(source_span) < 1.0 or abs(target_span) < 1.0 or source_span * target_span <= 0.0:
        raise ValueError("ROI and needle baseline levels must define distinct, consistently ordered anchors.")
    gain = target_span / source_span
    offset = float(target_needle_level) - gain * float(source_needle_level)
    return gain, offset


def align_frame_intensity(frame: np.ndarray, gain: float, offset: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * float(gain) + float(offset), 0.0, 255.0).astype(np.uint8)


def measure_roi_needle_baselines(
    gray_frames: list[np.ndarray],
    fps: float,
    roi_mask: np.ndarray,
    camera_view_mask: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    if not gray_frames:
        raise ValueError("ROI / needle level alignment requires video frames.")
    if roi_mask.shape != gray_frames[0].shape or not np.any(roi_mask):
        raise ValueError("ROI / needle level alignment requires a non-empty ROI mask matching the frames.")
    needle_mask = segment_pre_injection_needle(gray_frames, fps, camera_view_mask)
    if needle_mask is None:
        raise ValueError("ROI / needle level alignment could not segment the stationary needle.")
    baseline_count = min(
        len(gray_frames),
        max(
            1,
            detect_pre_injection_trim_start(
                gray_frames,
                fps,
                camera_view_mask=camera_view_mask,
            )
            + max(1, round(float(fps) * 0.5)),
        ),
    )
    baseline_frames = gray_frames[:baseline_count]
    roi_level = float(np.mean(masked_average_brightness(baseline_frames, roi_mask)))
    needle_level = float(np.mean(needle_average_brightness(baseline_frames, needle_mask)))
    return roi_level, needle_level, needle_mask


def roi_selection_mask(roi: ROISelection, frame_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    x, y, width, height = roi.rect.getRect()
    if x < 0 or y < 0 or x + width > frame_shape[1] or y + height > frame_shape[0]:
        raise ValueError("ROI / needle level alignment requires an ROI contained within the video frame.")
    selected = np.ones((height, width), dtype=bool) if roi.mask is None else roi.mask
    mask[y : y + height, x : x + width][selected] = 255
    return mask


def compute_temporal_change_summary(
    gray_frames: list[np.ndarray],
    fps: float,
    camera_view_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time-integrated dark contrast and peak darkening for every pixel."""
    if not gray_frames:
        empty = np.zeros((1, 1), dtype=np.float32)
        return empty, empty

    shape = gray_frames[0].shape
    if any(frame.shape != shape for frame in gray_frames):
        raise ValueError("Contrast residence analysis requires consistently sized frames.")
    if len(gray_frames) == 1:
        empty = np.zeros(shape, dtype=np.float32)
        return empty, empty
    selected = np.ones(shape, dtype=bool) if camera_view_mask is None else camera_view_mask.astype(bool)
    if selected.shape != shape or not np.any(selected):
        raise ValueError("Contrast residence analysis requires a non-empty camera-view mask matching the frames.")

    baseline_count = min(len(gray_frames) - 1, max(3, round(max(float(fps), 1.0) * 0.75)))

    # Remove frame-wide fluoroscope gain changes before measuring local darkening.
    baseline_source = [np.clip(frame, 0, 255).astype(np.float32) for frame in gray_frames[:baseline_count]]
    baseline_levels = np.asarray([np.median(frame[selected]) for frame in baseline_source], dtype=np.float32)
    baseline_level = float(np.median(baseline_levels))
    baseline_frames = np.stack(
        [
            np.clip(frame + (baseline_level - level), 0.0, 255.0)
            for frame, level in zip(baseline_source, baseline_levels, strict=True)
        ],
        axis=0,
    )
    baseline = np.percentile(baseline_frames, 75.0, axis=0)
    baseline_noise = np.percentile(np.abs(baseline_frames - baseline), 95.0, axis=0)
    noise_floor = np.maximum(2.0, baseline_noise * 2.0)

    contrast_burden = np.zeros(shape, dtype=np.float32)
    peak_contrast = np.zeros(shape, dtype=np.float32)
    for frame in gray_frames[baseline_count:]:
        source = np.clip(frame, 0, 255).astype(np.float32)
        corrected = np.clip(source + (baseline_level - float(np.median(source[selected]))), 0.0, 255.0)
        dark_contrast = np.clip(baseline - corrected - noise_floor, 0.0, None).astype(np.float32)
        dark_contrast[~selected] = 0.0
        contrast_burden += dark_contrast
        np.maximum(peak_contrast, dark_contrast, out=peak_contrast)
    contrast_burden /= max(float(fps), 1e-6)
    return contrast_burden, peak_contrast


def _robust_positive_peak(values: list[np.ndarray], percentile: float = 99.0) -> float:
    positive_values = [value[value > 0] for value in values if np.any(value > 0)]
    if not positive_values:
        return 0.0
    combined = np.concatenate(positive_values)
    if combined.size < 100:
        return float(np.max(combined))
    return float(np.percentile(combined, percentile))


HEATMAP_COLORMAPS = {
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "cividis": cv2.COLORMAP_CIVIDIS,
}


def render_temporal_change_heatmap(
    contrast_burden: np.ndarray,
    peak_contrast: np.ndarray,
    burden_peak: float | None = None,
    contrast_peak: float | None = None,
    colormap: str = "hot",
) -> np.ndarray:
    """Render contrast residence as heat and peak vessel contrast as brightness."""
    burden_peak = (
        _robust_positive_peak([contrast_burden])
        if burden_peak is None
        else float(burden_peak)
    )
    contrast_peak = (
        _robust_positive_peak([peak_contrast])
        if contrast_peak is None
        else float(contrast_peak)
    )
    burden_normalized = (
        np.clip(contrast_burden / burden_peak, 0.0, 1.0)
        if burden_peak > 0
        else np.zeros(contrast_burden.shape, dtype=np.float32)
    )
    contrast_normalized = (
        np.clip(peak_contrast / contrast_peak, 0.0, 1.0)
        if contrast_peak > 0
        else np.zeros(peak_contrast.shape, dtype=np.float32)
    )
    heat_levels = np.round(np.power(burden_normalized, 0.55) * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat_levels, HEATMAP_COLORMAPS.get(colormap, cv2.COLORMAP_HOT))
    brightness = 0.08 + 0.92 * np.power(contrast_normalized, 0.45)
    rendered = np.clip(heatmap.astype(np.float32) * brightness[..., np.newaxis], 0, 255).astype(np.uint8)
    rendered[contrast_burden <= 0] = 0
    return rendered


def temporal_change_heatmap_peaks(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    burden_peak = _robust_positive_peak([burden for burden, _peak in results.values()])
    contrast_peak = _robust_positive_peak([peak for _burden, peak in results.values()])
    return burden_peak, contrast_peak


def detect_aneurysm_roi(
    gray_frames: list[np.ndarray],
    fps: float,
    *,
    camera_view_mask: np.ndarray | None = None,
    soften_mask: bool = False,
    soften_radius_ratio: float = 0.12,
    soften_threshold: float = 0.10,
) -> ROISelection | None:
    if len(gray_frames) < 3:
        return None

    detection_frame_count = max(3, round(max(1.0, fps) * 10.0))
    gray_frames = gray_frames[:detection_frame_count]

    height, width = gray_frames[0].shape
    if height < 16 or width < 16 or any(frame.shape != (height, width) for frame in gray_frames):
        return None
    valid_camera_pixels = np.ones((height, width), dtype=bool) if camera_view_mask is None else camera_view_mask.astype(bool)
    if valid_camera_pixels.shape != (height, width) or not np.any(valid_camera_pixels):
        return None

    baseline_count = min(len(gray_frames) - 1, max(3, round(max(1.0, fps) * 0.6)))
    temporal_indexes = np.linspace(
        baseline_count,
        len(gray_frames) - 1,
        num=min(72, len(gray_frames) - baseline_count),
        dtype=int,
    )
    sample_indexes = np.unique(np.concatenate((np.arange(baseline_count), temporal_indexes)))
    target_median = float(np.median([np.median(gray_frames[index][valid_camera_pixels]) for index in sample_indexes]))
    stabilized = np.empty((len(sample_indexes), height, width), dtype=np.uint8)
    for output_index, frame_index in enumerate(sample_indexes):
        stabilized[output_index] = stabilize_frame_gain(
            gray_frames[frame_index],
            target_median,
            0.75,
            1.33,
            valid_camera_pixels,
        )
    sampled_baseline_count = int(np.searchsorted(sample_indexes, baseline_count))
    baseline = np.percentile(stabilized[:sampled_baseline_count], 75.0, axis=0)
    darkest = np.percentile(stabilized, 10.0, axis=0)
    darkening = cv2.GaussianBlur(np.clip(baseline - darkest, 0, None), (5, 5), sigmaX=0)

    active_values = darkening[(darkening >= 3.0) & valid_camera_pixels]
    if active_values.size < max(20, round(height * width * 0.0005)):
        return None

    minimum_area = max(80, round(height * width * 0.0008))
    maximum_area = round(height * width * 0.25)
    thresholds = sorted(
        {
            max(3.0, float(np.percentile(active_values, percentile)))
            for percentile in (45.0, 60.0, 75.0, 87.5, 95.0, 97.0, 99.0)
        }
    )
    best_candidate: tuple[float, np.ndarray] | None = None
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = np.ones((5, 5), dtype=np.uint8)

    for threshold in thresholds:
        mask = ((darkening >= threshold) & valid_camera_pixels).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not minimum_area <= area <= maximum_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            x, y, component_width, component_height = cv2.boundingRect(contour)
            aspect_ratio = max(component_width, component_height) / max(1, min(component_width, component_height))
            if circularity < 0.32 or aspect_ratio > 2.2:
                continue

            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 1, thickness=-1)
            component_pixels = component_mask.astype(bool)
            response = float(np.median(darkening[component_pixels]))
            fill_fraction = area / max(1, component_width * component_height)
            edge_margin = min(x, y, width - (x + component_width), height - (y + component_height))
            edge_factor = 0.65 if edge_margin <= 1 else 1.0
            score = response * math.sqrt(area) * circularity**2 * (0.5 + fill_fraction) * edge_factor
            if best_candidate is None or score > best_candidate[0]:
                best_candidate = (score, contour.copy())

    if best_candidate is None:
        return None

    _, contour = best_candidate
    roi_x, roi_y, roi_width, roi_height = cv2.boundingRect(contour)
    soften_radius = max(0, round(min(roi_width, roi_height) * max(0.0, soften_radius_ratio))) if soften_mask else 0
    local_contour = contour.copy()
    local_contour[:, 0, 0] -= roi_x - soften_radius
    local_contour[:, 0, 1] -= roi_y - soften_radius
    roi_mask = np.zeros((roi_height + soften_radius * 2, roi_width + soften_radius * 2), dtype=np.uint8)
    cv2.drawContours(roi_mask, [local_contour], -1, 1, thickness=-1)
    if soften_radius > 0:
        soften_kernel_size = soften_radius * 2 + 1
        soften_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (soften_kernel_size, soften_kernel_size))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, soften_kernel)
        roi_mask = cv2.dilate(roi_mask, soften_kernel, iterations=1)
        roi_mask = cv2.GaussianBlur(roi_mask.astype(np.float32), (soften_kernel_size, soften_kernel_size), sigmaX=0)
        roi_mask = (roi_mask >= float(np.clip(soften_threshold, 0.0, 1.0))).astype(np.uint8)

    mask_origin_x = roi_x - soften_radius
    mask_origin_y = roi_y - soften_radius
    valid_local = np.zeros(roi_mask.shape, dtype=bool)
    source_left = max(0, mask_origin_x)
    source_top = max(0, mask_origin_y)
    source_right = min(width, mask_origin_x + roi_mask.shape[1])
    source_bottom = min(height, mask_origin_y + roi_mask.shape[0])
    if source_left < source_right and source_top < source_bottom:
        target_left = source_left - mask_origin_x
        target_top = source_top - mask_origin_y
        valid_local[
            target_top : target_top + source_bottom - source_top,
            target_left : target_left + source_right - source_left,
        ] = valid_camera_pixels[source_top:source_bottom, source_left:source_right]
    roi_mask &= valid_local.astype(np.uint8)

    mask_points = cv2.findNonZero(roi_mask)
    if mask_points is None:
        return None
    mask_x, mask_y, mask_width, mask_height = cv2.boundingRect(mask_points)
    roi = QRect(int(roi_x - soften_radius + mask_x), int(roi_y - soften_radius + mask_y), int(mask_width), int(mask_height))
    cropped_mask = roi_mask[mask_y : mask_y + mask_height, mask_x : mask_x + mask_width]
    return ROISelection(roi, cropped_mask.astype(bool))


def extract_aneurysm_roi(
    gray_frames: list[np.ndarray],
    fps: float,
    parameters: EnhancementParameters,
    camera_view_mask: np.ndarray | None = None,
) -> ROISelection | None:
    if parameters.roi_mode == "manual":
        circle_values = parameters.roi_manual_circle
        if circle_values is None:
            return None
        center_x, center_y, radius = circle_values
        if radius < 2:
            return None
        rect = QRect(center_x - radius, center_y - radius, 2 * radius + 1, 2 * radius + 1)
        mask = np.zeros((rect.height(), rect.width()), dtype=np.uint8)
        cv2.circle(mask, (radius, radius), radius, 1, thickness=-1)
        if camera_view_mask is not None:
            valid_local = np.zeros(mask.shape, dtype=bool)
            source_left = max(0, rect.x())
            source_top = max(0, rect.y())
            source_right = min(camera_view_mask.shape[1], rect.x() + rect.width())
            source_bottom = min(camera_view_mask.shape[0], rect.y() + rect.height())
            if source_left < source_right and source_top < source_bottom:
                target_left = source_left - rect.x()
                target_top = source_top - rect.y()
                valid_local[
                    target_top : target_top + source_bottom - source_top,
                    target_left : target_left + source_right - source_left,
                ] = camera_view_mask[source_top:source_bottom, source_left:source_right]
            mask &= valid_local.astype(np.uint8)
        if not np.any(mask):
            return None
        return ROISelection(rect, mask.astype(bool))
    return detect_aneurysm_roi(
        gray_frames,
        fps,
        camera_view_mask=camera_view_mask,
        soften_mask=bool(parameters.roi_softening_enabled),
        soften_radius_ratio=float(parameters.roi_softening_radius_ratio),
        soften_threshold=float(parameters.roi_softening_threshold),
    )


def roi_selection_from_mask(mask: np.ndarray) -> ROISelection | None:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return ROISelection(
        QRect(x, y, width, height),
        mask[y : y + height, x : x + width].astype(bool, copy=True),
    )


def fit_circle_to_convex_hull(mask: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < 3:
        return mask.astype(bool, copy=True)
    hull = cv2.convexHull(points).reshape(-1, 2).astype(np.float64)
    coefficients = np.column_stack((2.0 * hull[:, 0], 2.0 * hull[:, 1], np.ones(len(hull))))
    solution, *_ = np.linalg.lstsq(coefficients, np.sum(hull * hull, axis=1), rcond=None)
    center_x, center_y, offset = solution
    radius_squared = offset + center_x * center_x + center_y * center_y
    if radius_squared <= 0.0:
        return mask.astype(bool, copy=True)
    circle = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(circle, (round(center_x), round(center_y)), max(1, round(math.sqrt(radius_squared))), 1, thickness=-1)
    return circle.astype(bool)


def extract_aneurysm_regions(
    gray_frames: list[np.ndarray],
    fps: float,
    parameters: EnhancementParameters,
    camera_view_mask: np.ndarray | None = None,
) -> tuple[ROISelection | None, np.ndarray]:
    roi = extract_aneurysm_roi(gray_frames, fps, parameters, camera_view_mask)
    if roi is None or parameters.roi_mode == "manual":
        shape = gray_frames[0].shape if gray_frames else (1, 1)
        return roi, np.zeros(shape, dtype=np.uint8)

    temporal_change = compute_temporal_change_map(gray_frames, camera_view_mask)
    aneurysm_mask = np.zeros(temporal_change.shape, dtype=bool)
    x, y, width, height = roi.rect.getRect()
    aneurysm_mask[y : y + height, x : x + width] = roi.mask
    regions = segment_temporal_change_map(
        temporal_change,
        change_threshold=12.0,
        level_tolerance=0,
        minimum_area=80,
        smoothing_window=51,
        aneurysm_mask=aneurysm_mask,
        fill_aneurysm_hull=parameters.roi_convex_hull_enabled,
        camera_view_mask=camera_view_mask,
    )
    if parameters.roi_circle_fit_enabled:
        aneurysm_region = fit_circle_to_convex_hull(regions == ROI_ANEURYSM_LABEL)
        if camera_view_mask is not None:
            aneurysm_region &= camera_view_mask.astype(bool)
        regions[regions == ROI_ANEURYSM_LABEL] = ROI_VESSEL_LABEL
        regions[aneurysm_region] = ROI_ANEURYSM_LABEL
    segmented_roi = roi_selection_from_mask(regions == ROI_ANEURYSM_LABEL)
    if segmented_roi is not None:
        return segmented_roi, regions
    if parameters.roi_circle_fit_enabled:
        fitted_mask = fit_circle_to_convex_hull(roi.mask)
        if camera_view_mask is not None:
            x, y, width, height = roi.rect.getRect()
            fitted_mask &= camera_view_mask[y : y + height, x : x + width]
        return ROISelection(roi.rect, fitted_mask), regions
    return roi, regions


def segment_temporal_change_map(
    temporal_change: np.ndarray,
    change_threshold: float,
    level_tolerance: int,
    minimum_area: int,
    smoothing_window: int,
    aneurysm_mask: np.ndarray | None = None,
    fill_aneurysm_hull: bool = True,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    smoothed_change = temporal_change.astype(np.float32, copy=True)
    kernel_size = max(3, int(smoothing_window) | 1)
    smoothed_change = cv2.GaussianBlur(smoothed_change, (kernel_size, kernel_size), sigmaX=0)

    threshold = max(0.0, float(change_threshold))
    mask = (smoothed_change >= threshold).astype(np.uint8)
    selected = None
    if camera_view_mask is not None:
        if camera_view_mask.shape != mask.shape:
            raise ValueError("Temporal segmentation requires a camera-view mask matching the change map.")
        selected = camera_view_mask.astype(bool)
        mask[~selected] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if selected is not None:
        mask[~selected] = 0

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    change_map = np.zeros_like(mask, dtype=np.uint8)
    selected_components: list[int] = []
    for component in range(1, component_count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= minimum_area:
            selected_components.append(component)

    for component in selected_components:
        change_map[labels == component] = ROI_VESSEL_LABEL

    if aneurysm_mask is not None and aneurysm_mask.shape == mask.shape:
        aneurysm_pixels = (change_map > 0) & aneurysm_mask.astype(bool)
        change_map[aneurysm_pixels] = ROI_ANEURYSM_LABEL
        rows, columns = np.nonzero(aneurysm_pixels)
        points = np.column_stack((columns, rows)).astype(np.int32)
        if fill_aneurysm_hull and len(points) >= 3:
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(change_map, hull, ROI_ANEURYSM_LABEL)
    if selected is not None:
        change_map[~selected] = 0

    return change_map


def segment_temporal_change_contrast(
    gray_frames: list[np.ndarray],
    change_threshold: float,
    level_tolerance: int,
    minimum_area: int,
    smoothing_window: int,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    temporal_change = compute_temporal_change_map(gray_frames, camera_view_mask)
    roi = detect_aneurysm_roi(gray_frames, fps=15.0, camera_view_mask=camera_view_mask)
    aneurysm_mask = None
    if roi is not None:
        aneurysm_mask = np.zeros(temporal_change.shape, dtype=bool)
        x, y, width, height = roi.rect.getRect()
        aneurysm_mask[y : y + height, x : x + width] = roi.mask
    return segment_temporal_change_map(
        temporal_change,
        change_threshold,
        level_tolerance,
        minimum_area,
        smoothing_window,
        aneurysm_mask,
        camera_view_mask=camera_view_mask,
    )


def overlay_roi_regions(
    frame: np.ndarray,
    mask: np.ndarray,
    aneurysm_color: tuple[int, int, int] = (255, 0, 0),
    opacity: float = 0.42,
) -> np.ndarray:
    result = frame.copy()
    selected = mask > 0
    if not np.any(selected):
        return result
    overlay_colors = np.zeros_like(result)
    overlay_colors[mask == ROI_VESSEL_LABEL] = (0, 0, 255)
    overlay_colors[mask == ROI_ANEURYSM_LABEL] = aneurysm_color
    unknown = selected & (mask != ROI_VESSEL_LABEL) & (mask != ROI_ANEURYSM_LABEL)
    overlay_colors[unknown] = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)[unknown]
    result[selected] = np.clip(
        result[selected].astype(np.float32) * (1.0 - opacity)
        + overlay_colors[selected].astype(np.float32) * opacity,
        0,
        255,
    ).astype(np.uint8)
    return result


def overlay_needle_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    opacity: float = 0.55,
) -> np.ndarray:
    if frame.shape[:2] != mask.shape:
        raise ValueError("Needle overlay mask must match the video frame size.")
    result = frame.copy()
    selected = needle_core_mask(mask) > 0
    if not np.any(selected):
        return result
    result[selected] = np.clip(
        result[selected].astype(np.float32) * (1.0 - opacity)
        + np.asarray(color, dtype=np.float32) * opacity,
        0,
        255,
    ).astype(np.uint8)
    return result


def reference_mean(
    gray: np.ndarray,
    roi: QRect,
    roi_mask: np.ndarray | None = None,
    camera_view_mask: np.ndarray | None = None,
) -> float:
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
    if camera_view_mask is not None:
        if camera_view_mask.shape != gray.shape:
            raise ValueError("ROI analysis requires a camera-view mask matching the frame.")
        mask &= camera_view_mask[top:bottom, left:right].astype(bool)
    if roi_mask is None:
        mask[roi_top:roi_bottom, roi_left:roi_right] = False
    else:
        mask_slice = roi_mask[: roi_bottom - roi_top, : roi_right - roi_left]
        mask[roi_top:roi_bottom, roi_left:roi_right] &= ~mask_slice
    pixels = reference[mask]

    if pixels.size < 200:
        mask = np.ones(gray.shape, dtype=bool) if camera_view_mask is None else camera_view_mask.astype(bool).copy()
        if roi_mask is None:
            mask[roi.y() : roi.y() + roi.height(), roi.x() : roi.x() + roi.width()] = False
        else:
            mask[roi.y() : roi.y() + roi.height(), roi.x() : roi.x() + roi.width()] &= ~roi_mask
        pixels = gray[mask]
    return float(np.median(pixels)) if pixels.size else math.nan


def roi_mean(
    gray: np.ndarray,
    roi: QRect,
    roi_mask: np.ndarray | None = None,
    camera_view_mask: np.ndarray | None = None,
) -> float:
    roi_pixels = gray[roi.y() : roi.y() + roi.height(), roi.x() : roi.x() + roi.width()]
    selected = np.ones(roi_pixels.shape, dtype=bool) if roi_mask is None else roi_mask.astype(bool).copy()
    if camera_view_mask is not None:
        if camera_view_mask.shape != gray.shape:
            raise ValueError("ROI analysis requires a camera-view mask matching the frame.")
        selected &= camera_view_mask[roi.y() : roi.y() + roi.height(), roi.x() : roi.x() + roi.width()]
    pixels = roi_pixels[selected]
    return float(np.mean(pixels)) if pixels.size else math.nan


PARENT_VESSEL_ROI_SIZE = 50
BACKGROUND_ROI_SIZE = 150


def parent_vessel_dark_median(
    frames: list[np.ndarray],
    center_x: int,
    center_y: int,
    rotation_degrees: float,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    return parent_vessel_brightness_measurements(
        frames, center_x, center_y, rotation_degrees, camera_view_mask
    )[0]


def parent_vessel_average_brightness(
    frames: list[np.ndarray],
    center_x: int,
    center_y: int,
    rotation_degrees: float,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    return parent_vessel_brightness_measurements(
        frames, center_x, center_y, rotation_degrees, camera_view_mask
    )[1]


def parent_vessel_brightness_measurements(
    frames: list[np.ndarray],
    center_x: int,
    center_y: int,
    rotation_degrees: float,
    camera_view_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        empty = np.asarray([], dtype=float)
        return empty, empty
    shape = frames[0].shape
    if camera_view_mask is not None and camera_view_mask.shape != shape:
        raise ValueError("Parent vessel ROI analysis requires a camera-view mask matching the frame.")
    corners = cv2.boxPoints(
        ((float(center_x), float(center_y)), (PARENT_VESSEL_ROI_SIZE, PARENT_VESSEL_ROI_SIZE), float(rotation_degrees))
    ).round().astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners, 1)
    selected = mask.astype(bool)
    if camera_view_mask is not None:
        selected &= camera_view_mask
    darkest_values: list[float] = []
    average_values: list[float] = []
    for frame in frames:
        if frame.shape != shape:
            raise ValueError("Parent vessel ROI analysis requires equally sized frames.")
        pixels = np.asarray(frame)[selected]
        if not pixels.size:
            darkest_values.append(math.nan)
            average_values.append(math.nan)
            continue
        darkest_count = max(1, math.ceil(pixels.size * 0.10))
        darkest_values.append(float(np.median(np.partition(pixels, darkest_count - 1)[:darkest_count])))
        average_values.append(float(np.mean(pixels)))
    return np.asarray(darkest_values, dtype=float), np.asarray(average_values, dtype=float)


def background_roi_average_brightness(
    frames: list[np.ndarray],
    center_x: int,
    center_y: int,
    camera_view_mask: np.ndarray | None = None,
) -> np.ndarray:
    if not frames:
        return np.asarray([], dtype=float)
    shape = frames[0].shape
    if camera_view_mask is not None and camera_view_mask.shape != shape:
        raise ValueError("Background ROI analysis requires a camera-view mask matching the frame.")
    half_size = BACKGROUND_ROI_SIZE // 2
    left = max(0, center_x - half_size)
    right = min(shape[1], center_x + half_size)
    top = max(0, center_y - half_size)
    bottom = min(shape[0], center_y + half_size)
    selected = np.zeros(shape, dtype=bool)
    selected[top:bottom, left:right] = True
    if camera_view_mask is not None:
        selected &= camera_view_mask
    return np.asarray(
        [float(np.mean(np.asarray(frame)[selected])) if np.any(selected) else math.nan for frame in frames],
        dtype=float,
    )


class VideoDisplay(QLabel):
    roiChanged = Signal(QRect)
    roiDrawn = Signal(object)
    parentVesselRoiDrawn = Signal(object)
    backgroundRoiDrawn = Signal(object)
    MANUAL_ROI_RADIUS = 70

    def __init__(self, title: str, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.roi_color = color
        self.frame_size = (0, 0)
        self._left_pixmap = QPixmap()
        self._right_pixmap = QPixmap()
        self._comparison_enabled = True
        self._left_label = "Source"
        self._roi: QRect | None = None
        self._roi_mask: np.ndarray | None = None
        self._parent_vessel_roi: tuple[int, int, float] | None = None
        self._background_roi: tuple[int, int] | None = None
        self._display_rect = QRect()
        self._right_display_rect = QRect()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Keep video areas landscape by preventing the display from becoming taller than wide.
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0e1116; border: 1px solid #253044; border-radius: 8px;")
        self._update_landscape_height_limit()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._update_landscape_height_limit()

    def _update_landscape_height_limit(self) -> None:
        max_height = max(1, self.width())
        if self.maximumHeight() != max_height:
            self.setMaximumHeight(max_height)

    def _to_pixmap(self, frame: np.ndarray) -> QPixmap:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(image)

    def set_frames(
        self,
        original_frame: np.ndarray,
        enhanced_frame: np.ndarray | None = None,
        left_label: str = "Source",
    ) -> None:
        self._left_pixmap = self._to_pixmap(original_frame)
        self.frame_size = (original_frame.shape[1], original_frame.shape[0])
        self._left_label = left_label
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

    def roi_mask(self) -> np.ndarray | None:
        return None if self._roi_mask is None else self._roi_mask.copy()

    def clear_roi(self) -> None:
        self._roi = None
        self._roi_mask = None
        self.update()

    def set_roi(self, roi: QRect | None, mask: np.ndarray | None = None) -> None:
        self._roi = QRect(roi) if roi is not None and roi.isValid() else None
        self._roi_mask = None if self._roi is None or mask is None else mask.astype(bool, copy=True)
        self.update()
        self.roiChanged.emit(QRect(self._roi) if self._roi is not None else QRect())

    def set_parent_vessel_roi(self, roi: tuple[int, int, float] | None) -> None:
        self._parent_vessel_roi = roi
        self.update()

    def set_background_roi(self, roi: tuple[int, int] | None) -> None:
        self._background_roi = roi
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
                painter.drawText(left_slot.adjusted(8, 6, -8, -6), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self._left_label)
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
            display_roi = self._frame_to_display_rect(self._roi, self._roi_display_rect())
            pen = QPen(self.roi_color, 2)
            painter.setPen(pen)
            painter.setBrush(QColor(self.roi_color.red(), self.roi_color.green(), self.roi_color.blue(), 35))
            if self._roi_mask is None:
                painter.drawRect(display_roi)
            else:
                contours, _ = cv2.findContours(self._roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                drew_contour = False
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    polygon = QPolygon(
                        [
                            self._frame_to_display_point(
                                QPoint(self._roi.x() + int(point[0][0]), self._roi.y() + int(point[0][1])),
                                self._roi_display_rect(),
                            )
                            for point in contour
                        ]
                    )
                    painter.drawPolygon(polygon)
                    drew_contour = True
                if not drew_contour:
                    painter.drawRect(display_roi)

            painter.setPen(QColor("#f8fafc"))
            painter.drawText(display_roi.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop, "ROI")

        if self._parent_vessel_roi is not None and self.frame_size != (0, 0):
            center_x, center_y, rotation = self._parent_vessel_roi
            half_size = PARENT_VESSEL_ROI_SIZE / 2.0
            radians = math.radians(rotation)
            corners = []
            for x_offset, y_offset in ((-half_size, -half_size), (half_size, -half_size), (half_size, half_size), (-half_size, half_size)):
                x = center_x + x_offset * math.cos(radians) - y_offset * math.sin(radians)
                y = center_y + x_offset * math.sin(radians) + y_offset * math.cos(radians)
                corners.append(self._frame_to_display_point(QPoint(round(x), round(y)), self._roi_display_rect()))
            painter.setPen(QPen(QColor("#facc15"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QPolygon(corners))
            painter.setPen(QColor("#fefce8"))
            painter.drawText(QRect(corners[0], corners[2]).normalized().adjusted(4, 4, -4, -4), Qt.AlignmentFlag.AlignTop, "Parent vessel")

        if self._background_roi is not None and self.frame_size != (0, 0):
            center_x, center_y = self._background_roi
            half_size = BACKGROUND_ROI_SIZE // 2
            roi = QRect(center_x - half_size, center_y - half_size, BACKGROUND_ROI_SIZE, BACKGROUND_ROI_SIZE)
            display_roi = self._frame_to_display_rect(roi, self._roi_display_rect())
            painter.setPen(QPen(QColor("#22c55e"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(display_roi)
            painter.setPen(QColor("#f0fdf4"))
            painter.drawText(display_roi.adjusted(4, 4, -4, -4), Qt.AlignmentFlag.AlignTop, "Background")

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if self.frame_size == (0, 0):
            return
        point = event.position().toPoint()
        if not self._roi_display_rect().contains(point):
            return
        frame_point = self._display_to_frame_point(point)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.backgroundRoiDrawn.emit((frame_point.x(), frame_point.y()))
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.parentVesselRoiDrawn.emit((frame_point.x(), frame_point.y()))
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        radius = self.MANUAL_ROI_RADIUS
        self._roi = QRect(frame_point.x() - radius, frame_point.y() - radius, 2 * radius + 1, 2 * radius + 1)
        self._roi_mask = np.zeros((self._roi.height(), self._roi.width()), dtype=np.uint8)
        cv2.circle(self._roi_mask, (radius, radius), radius, 1, thickness=-1)
        self._roi_mask = self._roi_mask.astype(bool)
        self.roiChanged.emit(QRect(self._roi))
        self.roiDrawn.emit((frame_point.x(), frame_point.y(), radius))
        self.update()

    def _display_to_frame_point(self, point: QPoint) -> QPoint:
        display_rect = self._roi_display_rect()
        width, height = self.frame_size
        x_fraction = (point.x() - display_rect.left()) / max(1, display_rect.width())
        y_fraction = (point.y() - display_rect.top()) / max(1, display_rect.height())
        x = round(max(0.0, min(1.0, x_fraction)) * (width - 1))
        y = round(max(0.0, min(1.0, y_fraction)) * (height - 1))
        return QPoint(x, y)

    def _roi_display_rect(self) -> QRect:
        return self._right_display_rect if self._comparison_enabled else self._display_rect

    def _frame_to_display_rect(self, rect: QRect, display_rect: QRect | None = None) -> QRect:
        display_rect = display_rect or self._display_rect
        width, height = self.frame_size
        x_scale = display_rect.width() / max(1, width)
        y_scale = display_rect.height() / max(1, height)
        return QRect(
            round(display_rect.left() + rect.left() * x_scale),
            round(display_rect.top() + rect.top() * y_scale),
            round(rect.width() * x_scale),
            round(rect.height() * y_scale),
        )

    def _frame_to_display_point(self, point: QPoint, display_rect: QRect | None = None) -> QPoint:
        display_rect = display_rect or self._display_rect
        width, height = self.frame_size
        x_scale = display_rect.width() / max(1, width)
        y_scale = display_rect.height() / max(1, height)
        return QPoint(
            round(display_rect.left() + point.x() * x_scale),
            round(display_rect.top() + point.y() * y_scale),
        )


class VideoDropGlyph(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("videoDropGlyph")
        self.setFixedSize(102, 78)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event) -> None:  # noqa: ANN001
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        drag_active = bool(self.parentWidget() and self.parentWidget().property("dragActive"))
        accent = QColor("#5eead4") if drag_active else QColor("#94a3b8")
        tray_pen = QPen(accent, 2)
        tray_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        tray_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        painter.setPen(tray_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRect(13, 47, 76, 18), 6, 6)
        painter.drawLine(24, 47, 24, 39)
        painter.drawLine(78, 47, 78, 39)

        painter.drawLine(51, 14, 51, 38)
        arrow = QPolygon([QPoint(40, 31), QPoint(51, 42), QPoint(62, 31)])
        painter.setBrush(accent)
        painter.drawPolygon(arrow)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRect(37, 5, 28, 14), 3, 3)


class LandscapeSurface(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._update_landscape_height_limit()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._update_landscape_height_limit()

    def _update_landscape_height_limit(self) -> None:
        max_height = max(1, self.width())
        if self.maximumHeight() != max_height:
            self.setMaximumHeight(max_height)


class CurrentPageStack(QStackedWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._height_refresh_pending = False
        self.currentChanged.connect(self._update_current_page_geometry)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.request_height_refresh()

    def hasHeightForWidth(self) -> bool:
        page = self.currentWidget()
        return page is not None and (page.hasHeightForWidth() or bool(self._current_page_cards()))

    def heightForWidth(self, width: int) -> int:
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        card_height = self._card_height_for_width(width)
        if card_height is not None:
            return card_height
        return page.heightForWidth(width) if page.hasHeightForWidth() else page.sizeHint().height()

    def sizeHint(self) -> QSize:
        page = self.currentWidget()
        if page is None:
            return super().sizeHint()
        card_height = self._card_height_for_width(self.width()) if self.width() > 0 else None
        if card_height is None:
            return page.sizeHint()
        return QSize(page.sizeHint().width(), card_height)

    def _update_current_page_geometry(self) -> None:
        self.updateGeometry()
        if self.parentWidget() is not None and self.parentWidget().layout() is not None:
            self.parentWidget().layout().invalidate()
        self.request_height_refresh()

    def _current_page_cards(self) -> list[QWidget]:
        page = self.currentWidget()
        layout = page.layout() if page is not None else None
        if layout is None:
            return []
        return [
            widget
            for index in range(layout.count())
            if (widget := layout.itemAt(index).widget()) is not None and widget.hasHeightForWidth()
        ]

    def _card_height_for_width(self, width: int) -> int | None:
        page = self.currentWidget()
        layout = page.layout() if page is not None else None
        cards = self._current_page_cards()
        if layout is None or not cards:
            return None

        margins = layout.contentsMargins()
        available_width = max(
            0,
            width - margins.left() - margins.right() - layout.spacing() * (len(cards) - 1),
        )
        card_width = available_width // len(cards)
        return margins.top() + max(card.heightForWidth(card_width) for card in cards) + margins.bottom()

    def request_height_refresh(self) -> None:
        if self._height_refresh_pending:
            return
        self._height_refresh_pending = True
        QTimer.singleShot(0, self._apply_current_page_height)

    def _apply_current_page_height(self) -> None:
        self._height_refresh_pending = False
        page = self.currentWidget()
        if page is None:
            return
        card_height = self._card_height_for_width(self.width())
        if card_height is None:
            self.setMaximumHeight(16_777_215)
            return
        self.setMaximumHeight(card_height)
        self.updateGeometry()
        if self.parentWidget() is not None and self.parentWidget().layout() is not None:
            self.parentWidget().layout().activate()


class UniformScaleView(QGraphicsView):
    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._content_size = content.size()
        self._scene = QGraphicsScene(self)
        self._proxy = self._scene.addWidget(content)
        self._scene.setSceneRect(0, 0, self._content_size.width(), self._content_size.height())
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return self._content_size

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        viewport_size = self.viewport().size()
        available_width = max(0, viewport_size.width() - 2)
        available_height = max(0, viewport_size.height() - 2)
        scale = min(
            1.0,
            available_width / max(1, self._content_size.width()),
            available_height / max(1, self._content_size.height()),
        )
        self.resetTransform()
        self.scale(scale, scale)


class VideoDropPlaceholder(QFrame):
    fileDialogRequested = Signal()
    fileDropped = Signal(object)
    SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}
    PANEL_MARGIN = 14
    PANEL_SPACING = 10
    TITLE_HEIGHT = 30

    def __init__(self, label: str, color: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = label
        self.color = color
        self.selected_path: Path | None = None
        self.setObjectName("videoDropPlaceholder")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setProperty("dragActive", False)
        # Match VideoPanel sizing so replacing one slot does not reflow panel widths/heights.
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.title_label = QLabel(label)
        self.title_label.setObjectName("panelTitle")

        self.title_container = QWidget()
        self.title_container.setObjectName("videoPanelTitleContainer")
        self.title_container.setFixedHeight(self.TITLE_HEIGHT)
        title_layout = QHBoxLayout(self.title_container)
        title_layout.setContentsMargins(8, 2, 8, 2)
        title_layout.addWidget(self.title_label)

        self.video_surface = LandscapeSurface()
        self.video_surface.setObjectName("videoDropSurface")
        self.video_surface.setProperty("dragActive", False)
        # Mirror VideoDisplay sizing so placeholders reserve the same visual area.
        self.video_surface.setMinimumSize(0, 0)
        self.video_surface.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.hint_button = QPushButton("")
        self.hint_button.setObjectName("videoDropHintButton")
        self.hint_button.setProperty("dragActive", False)
        self.hint_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hint_button.setFixedSize(170, 170)
        hint_layout = QVBoxLayout(self.hint_button)
        hint_layout.setContentsMargins(0, 0, 0, 0)
        hint_layout.setSpacing(0)

        self.hint_glyph = VideoDropGlyph(self.hint_button)
        hint_layout.addWidget(self.hint_glyph, 0, Qt.AlignmentFlag.AlignCenter)
        self.hint_button.clicked.connect(self.fileDialogRequested.emit)

        surface_layout = QVBoxLayout(self.video_surface)
        surface_layout.setContentsMargins(16, 16, 16, 16)
        surface_layout.setSpacing(0)
        surface_layout.addStretch()
        surface_layout.addWidget(self.hint_button, 0, Qt.AlignmentFlag.AlignCenter)
        surface_layout.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.PANEL_MARGIN, self.PANEL_MARGIN, self.PANEL_MARGIN, self.PANEL_MARGIN)
        layout.setSpacing(self.PANEL_SPACING)
        layout.addWidget(self.title_container)
        layout.addWidget(self.video_surface, 1)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.setMaximumHeight(self.heightForWidth(self.width()))
        QTimer.singleShot(0, self._refresh_surface_layout)

    def _refresh_surface_layout(self) -> None:
        layout = self.video_surface.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        surface_width = max(0, width - 2 * self.PANEL_MARGIN)
        return 2 * self.PANEL_MARGIN + self.TITLE_HEIGHT + self.PANEL_SPACING + surface_width

    def set_selected_path(self, path: Path) -> None:
        self.selected_path = path

    def dragEnterEvent(self, event) -> None:  # noqa: ANN001
        if self._extract_video_path(event) is not None:
            self._set_drag_active(True)
            event.acceptProposedAction()
            return
        self._set_drag_active(False)
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._extract_video_path(event) is not None:
            self._set_drag_active(True)
            event.acceptProposedAction()
            return
        self._set_drag_active(False)
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: ANN001
        self._set_drag_active(False)
        event.accept()

    def dropEvent(self, event) -> None:  # noqa: ANN001
        self._set_drag_active(False)
        path = self._extract_video_path(event)
        if path is None:
            event.ignore()
            return
        self.fileDropped.emit(path)
        event.acceptProposedAction()

    def _set_drag_active(self, active: bool) -> None:
        if bool(self.property("dragActive")) == active:
            return
        self.setProperty("dragActive", active)
        self.video_surface.setProperty("dragActive", active)
        self.hint_button.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.style().unpolish(self.video_surface)
        self.style().polish(self.video_surface)
        self.style().unpolish(self.hint_button)
        self.style().polish(self.hint_button)
        self.update()
        self.video_surface.update()
        self.hint_button.update()
        self.hint_glyph.update()

    def _extract_video_path(self, event) -> Path | None:  # noqa: ANN001
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return None
        for url in mime_data.urls():
            local_path = url.toLocalFile()
            if not local_path:
                continue
            path = Path(local_path)
            if not path.exists() or not path.is_file():
                continue
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS or is_dicom_video(path):
                return path
        return None


class VideoPanel(QFrame):
    roiChanged = Signal()
    roiDrawn = Signal(object)
    parentVesselRoiDrawn = Signal(object)
    backgroundRoiDrawn = Signal(object)
    PANEL_MARGIN = 14
    PANEL_SPACING = 10
    TITLE_HEIGHT = 30

    def __init__(
        self,
        label: str,
        color: QColor,
        path: Path,
        parent: QWidget | None = None,
        live_input: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label = label
        self.color = color
        self.path = path
        self.live_input = live_input
        self.info = probe_video(path)
        self.crop_rect = QRect(0, 0, self.info.width, self.info.height)
        self._auto_crop_rect_cache: QRect | None = None
        self._camera_field_mask_cache: np.ndarray | None = None
        self._trim_start_cache: dict[tuple[int, int, int, int], int] = {}
        self.source_pipeline_configuration: tuple[object, ...] | None = None
        self.background_subtraction_enabled = False
        self.background_level = 0
        self.background_reference: np.ndarray | None = None
        self.metal_needle_mask: np.ndarray | None = None
        self.camera_view_mask: np.ndarray | None = None
        self.source_gain_multiplier = 1.0
        self.source_gain_offset = 0.0
        self.source_histogram_match_lut: np.ndarray | None = None
        self.capture = open_video_capture(path)
        self.trim_start_frame = 0
        self.trim_frame_count = self.info.frame_count
        self.current_frame: np.ndarray | None = None
        self.current_frame_index = -1
        self.enhance_display = False
        self.comparison_display = True
        self.temporal_change_display = False
        self.roi_region_overlay_display = True
        self.needle_segmentation_mask: np.ndarray | None = None
        self.target_median = 128.0
        self.roi_needle_alignment_gain = 1.0
        self.roi_needle_alignment_offset = 0.0
        self.enhanced_frames: list[np.ndarray] | None = None
        self.report_encoded_frames: list[np.ndarray] | None = None
        self.temporal_change_heatmap: np.ndarray | None = None
        self.roi_region_masks: list[np.ndarray] | None = None
        self.source_gray_frames: list[np.ndarray] | None = None
        self.stage_frame_cache: dict[tuple[tuple[str, tuple[object, ...]], ...], list[np.ndarray]] = {}
        self.encoded_frame_cache: dict[tuple[tuple[str, tuple[object, ...]], ...], list[np.ndarray]] = {}
        self.roi_region_mask_cache: dict[tuple[tuple[str, tuple[object, ...]], ...], list[np.ndarray]] = {}
        self.roi_selection_cache: dict[tuple[tuple[str, tuple[object, ...]], ...], ROISelection | None] = {}
        self.temporal_change_map_cache: dict[tuple[tuple[str, tuple[object, ...]], ...], np.ndarray] = {}
        self.active_sequence_key: tuple[tuple[str, tuple[object, ...]], ...] | None = None
        self.inactive_sequence_key: tuple[tuple[str, tuple[object, ...]], ...] | None = None
        self.stage_duration_per_frame: dict[tuple[str, tuple[object, ...]], float] = {}
        self.stage_roi_selection: ROISelection | None = None

        self.display = VideoDisplay(label, color)
        self.display.set_comparison_enabled(self.comparison_display)
        self.display.roiChanged.connect(lambda _roi: self.roiChanged.emit())
        self.display.roiDrawn.connect(self.roiDrawn.emit)
        self.display.parentVesselRoiDrawn.connect(self.parentVesselRoiDrawn.emit)
        self.display.backgroundRoiDrawn.connect(self.backgroundRoiDrawn.emit)

        self.title_label = QLabel(label)
        self.title_label.setObjectName("panelTitle")
        self.path_label = QLabel(path.name)
        self.path_label.setObjectName("subtleLabel")
        self.meta_label = QLabel(self._metadata_text())
        self.meta_label.setObjectName("subtleLabel")

        self.title_container = QWidget()
        self.title_container.setObjectName("videoPanelTitleContainer")
        self.title_container.setFixedHeight(self.TITLE_HEIGHT)
        header = QHBoxLayout(self.title_container)
        header.setContentsMargins(8, 2, 8, 2)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.meta_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.PANEL_MARGIN, self.PANEL_MARGIN, self.PANEL_MARGIN, self.PANEL_MARGIN)
        layout.setSpacing(self.PANEL_SPACING)
        layout.addWidget(self.title_container)
        layout.addWidget(self.path_label)
        layout.addWidget(self.display, 1)

        self.setObjectName("videoPanel")
        self.set_trim_window(0)
        self.seek(0)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self.setMaximumHeight(self.heightForWidth(self.width()))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        surface_width = max(0, width - 2 * self.PANEL_MARGIN)
        path_height = self.path_label.sizeHint().height()
        return 2 * self.PANEL_MARGIN + self.TITLE_HEIGHT + path_height + 2 * self.PANEL_SPACING + surface_width

    def _full_frame_rect(self) -> QRect:
        return QRect(0, 0, self.info.width, self.info.height)

    def _crop_rect_key(self, crop_rect: QRect) -> tuple[int, int, int, int]:
        return crop_rect.x(), crop_rect.y(), crop_rect.width(), crop_rect.height()

    def calculate_source_pipeline(
        self,
        auto_crop_enabled: bool,
        temporal_alignment_enabled: bool,
        progress_callback: Callable[[str, float, float], bool] | None = None,
        auto_crop_size_offset: int = 0,
        temporal_trim_offset_seconds: float = 0.0,
        temporal_end_trim_seconds: float = 0.0,
        comparison_sync_offset_seconds: float = 0.0,
        contrast_gain_alignment_enabled: bool = False,
        histogram_matching_enabled: bool = False,
        background_subtraction_enabled: bool = False,
        background_level: int = 0,
        metal_needle_removal_enabled: bool = False,
        startup_stabilization_enabled: bool = False,
        temporal_gain_normalization_enabled: bool = True,
    ) -> SourcePipelineState:
        temporal_alignment_enabled = temporal_alignment_enabled and not self.live_input
        background_subtraction_enabled = background_subtraction_enabled and not self.live_input
        full_rect = self._full_frame_rect()
        next_crop_rect = full_rect
        source_stage_count = (
            int(auto_crop_enabled)
            + int(startup_stabilization_enabled)
            + int(temporal_alignment_enabled)
            + int(contrast_gain_alignment_enabled)
            + int(histogram_matching_enabled)
            + int(background_subtraction_enabled)
            + int(metal_needle_removal_enabled)
        )
        completed_stages = 0

        def report(stage: str, done: float, total: float) -> bool:
            return progress_callback is None or progress_callback(
                stage,
                completed_stages + done / max(1.0, total),
                source_stage_count,
            )

        auto_crop_rect = getattr(self, "_auto_crop_rect_cache", None)
        camera_field_mask = getattr(self, "_camera_field_mask_cache", None)
        if auto_crop_rect is None:
            if hasattr(self, "path"):
                auto_crop_rect, camera_field_mask = detect_fluoroscope_geometry(
                    self.path,
                    self.info,
                    lambda done, total: report("Detecting fluoroscope field", done, total),
                )
            else:
                auto_crop_rect = full_rect
                camera_field_mask = np.ones((self.info.height, self.info.width), dtype=bool)
            if self.live_input:
                self._auto_crop_rect_cache = QRect(auto_crop_rect)
                self._camera_field_mask_cache = camera_field_mask.copy()
        if auto_crop_enabled:
            next_crop_rect = _adjust_auto_crop_square(
                auto_crop_rect,
                self.info.width,
                self.info.height,
                auto_crop_size_offset,
            )
            report("Auto-cropping", 1.0, 1.0)
            completed_stages += 1
        camera_view_mask = camera_view_mask_for_crop(camera_field_mask, next_crop_rect)

        next_trim_start = 0
        next_trim_end = 0
        trim_cache_key: tuple[int, int, int, int] | None = None
        detected_trim_start: int | None = None
        frame_brightness_onset: int | None = None
        if startup_stabilization_enabled:
            gray_frames = self._sample_cropped_gray_frames(
                next_crop_rect,
                lambda done, total: report("Detecting fluoroscope startup stabilization", done, total),
            )
            next_trim_start = min(
                max(0, self.info.frame_count - 1),
                detect_fluoroscope_stabilization_frame(gray_frames, self.info.fps, camera_view_mask),
            )
            report("Detecting fluoroscope startup stabilization", 1.0, 1.0)
            completed_stages += 1
        if temporal_alignment_enabled:
            crop_key = self._crop_rect_key(next_crop_rect)
            trim_cache_key = crop_key
            cached_trim_start = self._trim_start_cache.get(crop_key)
            gray_frames = self._sample_cropped_gray_frames(
                next_crop_rect,
                lambda done, total: report("Aligning contrast timing", done, total),
            )
            if cached_trim_start is None:
                cached_trim_start = (
                    detect_pre_injection_trim_start(
                        gray_frames[next_trim_start:],
                        self.info.fps,
                        temporal_gain_normalization_enabled,
                        camera_view_mask,
                    )
                    + next_trim_start
                )
            detected_trim_start = cached_trim_start
            visible_onset = detect_frame_brightness_onset(
                gray_frames[next_trim_start:],
                self.info.fps,
                camera_view_mask,
            )
            if visible_onset is not None:
                frame_brightness_onset = next_trim_start + visible_onset
            comparison_sync_offset_frames = round(comparison_sync_offset_seconds * self.info.fps)
            trim_offset_frames = round(temporal_trim_offset_seconds * self.info.fps) + comparison_sync_offset_frames
            next_trim_start = max(0, min(self.info.frame_count - 1, cached_trim_start + trim_offset_frames))
            end_trim_frames = round(max(0.0, temporal_end_trim_seconds) * self.info.fps)
            next_trim_end = max(0, min(self.info.frame_count - next_trim_start - 1, end_trim_frames))
            report("Aligning contrast timing", 1.0, 1.0)

        gain_alignment_baseline = 0.0
        gain_alignment_span = 0.0
        histogram = None
        if contrast_gain_alignment_enabled:
            gain_alignment_baseline, gain_alignment_span = estimate_video_contrast_response(
                self.path,
                next_crop_rect,
                next_trim_start,
                max(1, self.info.frame_count - next_trim_start),
                self.info.fps,
                camera_view_mask,
            )
            report("Measuring fluoroscope contrast response", 1.0, 1.0)
            completed_stages += 1

        if histogram_matching_enabled:
            histogram = estimate_video_histogram(
                self.path,
                next_crop_rect,
                next_trim_start,
                max(1, self.info.frame_count - next_trim_start - next_trim_end),
                camera_view_mask,
            )
            report("Measuring comparison histogram", 1.0, 1.0)
            completed_stages += 1

        background_reference = None
        if background_subtraction_enabled:
            pre_injection_trim_start = detected_trim_start
            if pre_injection_trim_start is None:
                gray_frames = self._sample_cropped_gray_frames(next_crop_rect)
                pre_injection_trim_start = detect_pre_injection_trim_start(
                    gray_frames,
                    self.info.fps,
                    camera_view_mask=camera_view_mask,
                )
            mask_end_frame = min(
                self.info.frame_count,
                pre_injection_trim_start + round(self.info.fps * 0.5),
            )
            background_reference = self._acquire_dsa_mask(
                next_crop_rect,
                mask_end_frame,
                lambda done, total: report("Acquiring DSA mask", done, total),
            )
            report("Acquiring DSA mask", 1.0, 1.0)
            completed_stages += 1

        metal_needle_mask = None
        if metal_needle_removal_enabled:
            metal_needle_mask = self._acquire_stationary_metal_mask(
                next_crop_rect,
                lambda done, total: report("Detecting stationary metal", done, total),
                camera_view_mask,
            )
            report("Detecting stationary metal", 1.0, 1.0)
            completed_stages += 1

        return SourcePipelineState(
            crop_rect=next_crop_rect,
            trim_start=next_trim_start,
            trim_end=next_trim_end,
            auto_crop_rect=auto_crop_rect,
            trim_cache_key=trim_cache_key,
            detected_trim_start=detected_trim_start,
            frame_brightness_onset=frame_brightness_onset,
            comparison_sync_offset_frames=comparison_sync_offset_frames if temporal_alignment_enabled else 0,
            background_reference=background_reference,
            metal_needle_mask=metal_needle_mask,
            camera_view_mask=camera_view_mask,
            camera_field_mask=camera_field_mask,
            configuration=(
                auto_crop_enabled,
                temporal_alignment_enabled,
                contrast_gain_alignment_enabled,
                background_subtraction_enabled,
                background_level,
                auto_crop_size_offset,
                temporal_trim_offset_seconds,
                temporal_end_trim_seconds,
                comparison_sync_offset_seconds,
                metal_needle_removal_enabled,
                histogram_matching_enabled,
            ) + ((True,) if startup_stabilization_enabled else ()) + ((False,) if not temporal_gain_normalization_enabled else ()),
            gain_alignment_baseline=gain_alignment_baseline,
            gain_alignment_span=gain_alignment_span,
            histogram=histogram,
        )

    def apply_source_pipeline_state(self, state: SourcePipelineState) -> bool:
        configuration_changed = (
            self.source_pipeline_configuration != state.configuration
            or self.source_gain_multiplier != state.gain_multiplier
            or self.source_gain_offset != state.gain_offset
            or not np.array_equal(self.source_histogram_match_lut, state.histogram_match_lut)
            or not np.array_equal(self.camera_view_mask, state.camera_view_mask)
        )
        self.source_pipeline_configuration = state.configuration
        self.source_gain_multiplier = state.gain_multiplier
        self.source_gain_offset = state.gain_offset
        self.source_histogram_match_lut = state.histogram_match_lut
        self.background_subtraction_enabled = state.configuration[3]
        self.background_level = state.configuration[4]
        self.background_reference = state.background_reference
        self.metal_needle_mask = state.metal_needle_mask
        self.camera_view_mask = None if state.camera_view_mask is None else state.camera_view_mask.copy()
        if state.auto_crop_rect is not None:
            self._auto_crop_rect_cache = QRect(state.auto_crop_rect)
        if state.camera_field_mask is not None:
            self._camera_field_mask_cache = state.camera_field_mask.copy()
        if state.trim_cache_key is not None and state.detected_trim_start is not None:
            self._trim_start_cache[state.trim_cache_key] = state.detected_trim_start

        available_frames = max(1, self.info.frame_count - state.trim_start - state.trim_end)
        if (
            state.crop_rect == self.crop_rect
            and state.trim_start == self.trim_start_frame
            and self.trim_frame_count == available_frames
        ):
            if not configuration_changed:
                return False
            self.clear_enhancement_cache()
            return True

        self.crop_rect = QRect(state.crop_rect)
        self.set_trim_window(state.trim_start, available_frames)
        self._activate_stage_roi_selection(None)
        return True

    def apply_source_pipeline(
        self,
        auto_crop_enabled: bool,
        temporal_alignment_enabled: bool,
        auto_crop_size_offset: int = 0,
        temporal_trim_offset_seconds: float = 0.0,
        temporal_end_trim_seconds: float = 0.0,
        comparison_sync_offset_seconds: float = 0.0,
        contrast_gain_alignment_enabled: bool = False,
        histogram_matching_enabled: bool = False,
        background_subtraction_enabled: bool = False,
        background_level: int = 0,
        metal_needle_removal_enabled: bool = False,
    ) -> bool:
        return self.apply_source_pipeline_state(
            self.calculate_source_pipeline(
                auto_crop_enabled,
                temporal_alignment_enabled,
                auto_crop_size_offset=auto_crop_size_offset,
                temporal_trim_offset_seconds=temporal_trim_offset_seconds,
                temporal_end_trim_seconds=temporal_end_trim_seconds,
                comparison_sync_offset_seconds=comparison_sync_offset_seconds,
                contrast_gain_alignment_enabled=contrast_gain_alignment_enabled,
                histogram_matching_enabled=histogram_matching_enabled,
                background_subtraction_enabled=background_subtraction_enabled,
                background_level=background_level,
                metal_needle_removal_enabled=metal_needle_removal_enabled,
            )
        )

    def _sample_cropped_gray_frames(
        self,
        crop_rect: QRect | None = None,
        progress_callback: Callable[[int, int], bool] | None = None,
    ) -> list[np.ndarray]:
        capture = open_video_capture(self.path)
        gray_frames: list[np.ndarray] = []
        active_crop_rect = crop_rect if crop_rect is not None else self.crop_rect
        frame_count = max(1, self.info.frame_count)
        try:
            for frame_index in range(frame_count):
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(crop_frame(frame, active_crop_rect), cv2.COLOR_BGR2GRAY)
                gray_frames.append(gray)
                if progress_callback is not None and not progress_callback(frame_index + 1, frame_count):
                    break
        finally:
            capture.release()
        return gray_frames

    def _acquire_dsa_mask(
        self,
        crop_rect: QRect,
        mask_end_frame: int,
        progress_callback: Callable[[int, int], bool] | None = None,
    ) -> np.ndarray | None:
        sample_count = min(max(3, round(self.info.fps)), max(0, mask_end_frame))
        if sample_count < 3:
            return None
        capture = open_video_capture(self.path)
        samples: list[np.ndarray] = []
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, mask_end_frame - sample_count)
            for frame_index in range(sample_count):
                ok, frame = capture.read()
                if not ok:
                    break
                samples.append(cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY))
                if progress_callback is not None and not progress_callback(frame_index + 1, sample_count):
                    break
        finally:
            capture.release()
        if not samples:
            return None
        return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)

    def _acquire_stationary_metal_mask(
        self,
        crop_rect: QRect,
        progress_callback: Callable[[int, int], bool] | None = None,
        camera_view_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
        sample_count = min(max(3, round(self.info.fps * 2.0)), self.info.frame_count)
        capture = open_video_capture(self.path)
        samples: list[np.ndarray] = []
        try:
            for frame_index in range(sample_count):
                ok, frame = capture.read()
                if not ok:
                    break
                samples.append(cv2.cvtColor(crop_frame(frame, crop_rect), cv2.COLOR_BGR2GRAY))
                if progress_callback is not None and not progress_callback(frame_index + 1, sample_count):
                    break
        finally:
            capture.release()
        return detect_stationary_metal_mask(samples, camera_view_mask)

    def _subtract_background(self, gray: np.ndarray) -> np.ndarray:
        if not self.background_subtraction_enabled or self.background_reference is None:
            return gray
        return subtract_fluoroscopy_background(gray, self.background_reference, self.background_level)

    def _remove_stationary_metal(self, gray: np.ndarray) -> np.ndarray:
        if self.metal_needle_mask is None:
            return gray
        output = gray.copy()
        component_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(self.metal_needle_mask, connectivity=8)
        boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
        for label in range(1, component_count):
            component = (labels == label).astype(np.uint8)
            boundary = cv2.dilate(component, boundary_kernel) & (1 - component)
            if np.any(boundary):
                output[component.astype(bool)] = np.median(gray[boundary.astype(bool)])
        return output

    def _align_source_gain(self, gray: np.ndarray) -> np.ndarray:
        if self.source_gain_multiplier == 1.0 and self.source_gain_offset == 0.0:
            return gray
        aligned = gray.astype(np.float32) * self.source_gain_multiplier + self.source_gain_offset
        return np.clip(aligned, 0, 255).astype(np.uint8)

    def _match_source_histogram(self, gray: np.ndarray) -> np.ndarray:
        if self.source_histogram_match_lut is None:
            return gray
        return cv2.LUT(gray, self.source_histogram_match_lut)

    def set_trim_window(self, start_frame: int, frame_count: int | None = None) -> None:
        max_start = max(0, self.info.frame_count - 1)
        self.trim_start_frame = max(0, min(start_frame, max_start))
        available_frames = max(1, self.info.frame_count - self.trim_start_frame)
        self.trim_frame_count = max(1, min(frame_count if frame_count is not None else available_frames, available_frames))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start_frame)
        self.target_median = estimate_video_median(
            self.path,
            self.crop_rect,
            self.trim_start_frame,
            self.trim_frame_count,
            getattr(self, "camera_view_mask", None),
        )
        self.current_frame_index = -1
        self.clear_enhancement_cache()
        self.meta_label.setText(self._metadata_text())

    @property
    def playback_frame_count(self) -> int:
        return self.trim_frame_count

    @property
    def playback_duration(self) -> float:
        return self.playback_frame_count / self.info.fps if self.info.fps else 0.0

    def _stage_token(
        self,
        stage_key: str,
        backend_id: str,
        noise_sigma: int,
        parameters: EnhancementParameters,
    ) -> tuple[str, tuple[object, ...]]:
        if stage_key == "gain_stabilization":
            target_median = self.target_median if parameters.gain_use_auto_target else float(parameters.gain_target_median)
            return (
                stage_key,
                (
                    round(float(target_median), 4),
                    round(float(parameters.gain_min), 4),
                    round(float(parameters.gain_max), 4),
                ),
            )
        if stage_key == "roi_needle_level_alignment":
            return (
                stage_key,
                (
                    round(float(self.roi_needle_alignment_gain), 6),
                    round(float(self.roi_needle_alignment_offset), 4),
                ),
            )
        return BUILTIN_STAGES.require(stage_key).cache_token(parameters, backend_id, noise_sigma)

    def _sequence_key(
        self,
        stages: EnhancementStages,
        backend_id: str,
        noise_sigma: int,
        parameters: EnhancementParameters,
    ) -> tuple[tuple[str, tuple[object, ...]], ...]:
        return tuple(
            self._stage_token(
                stage.key,
                backend_id,
                stage.noise_sigma if stage.noise_sigma is not None else noise_sigma,
                stage.parameters or parameters,
            )
            for stage in stages.enabled_stage_instances(parameters)
        )

    def _frame_sequence_key(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> tuple[tuple[str, tuple[object, ...]], ...]:
        return tuple(
            token
            for token in sequence_key
            if BUILTIN_STAGES.require(token[0]).modifies_frame_data
        )

    def _frame_prefixes(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> set[tuple[tuple[str, tuple[object, ...]], ...]]:
        prefixes: set[tuple[tuple[str, tuple[object, ...]], ...]] = set()
        prefix: tuple[tuple[str, tuple[object, ...]], ...] = tuple()
        for token in sequence_key:
            if not BUILTIN_STAGES.require(token[0]).modifies_frame_data:
                continue
            prefix += (token,)
            prefixes.add(prefix)
        return prefixes

    def _prune_cache_branches(
        self,
        sequence_keys: tuple[tuple[tuple[str, tuple[object, ...]], ...], ...],
    ) -> None:
        retained_frame_keys = {self._frame_sequence_key(key) for key in sequence_keys}
        retained_frame_prefixes: set[tuple[tuple[str, tuple[object, ...]], ...]] = set()
        for key in sequence_keys:
            retained_frame_prefixes.update(self._frame_prefixes(key))

        self.encoded_frame_cache = {
            key: frames
            for key, frames in self.encoded_frame_cache.items()
            if key in retained_frame_keys
        }
        self.stage_frame_cache = {
            key: frames
            for key, frames in self.stage_frame_cache.items()
            if key in retained_frame_prefixes
        }

        retained_artifact_keys = {
            self._artifact_cache_key(sequence_key[: index + 1])
            for sequence_key in sequence_keys
            for index, token in enumerate(sequence_key)
            if not BUILTIN_STAGES.require(token[0]).modifies_frame_data
        }

        self.roi_region_mask_cache = {
            key: masks
            for key, masks in self.roi_region_mask_cache.items()
            if key in retained_artifact_keys
        }
        self.roi_selection_cache = {
            key: selection
            for key, selection in self.roi_selection_cache.items()
            if key in retained_artifact_keys
        }
        self.temporal_change_map_cache = {
            key: change_map
            for key, change_map in self.temporal_change_map_cache.items()
            if key in retained_frame_prefixes or not key
        }

    def _begin_cache_branch(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> None:
        if self.active_sequence_key is None or sequence_key in {
            self.active_sequence_key,
            self.inactive_sequence_key,
        }:
            return
        self._prune_cache_branches((self.active_sequence_key,))
        self.inactive_sequence_key = None

    def _activate_cache_branch(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> None:
        if sequence_key != self.active_sequence_key:
            self.inactive_sequence_key = self.active_sequence_key
            self.active_sequence_key = sequence_key
        retained = tuple(
            key
            for key in (self.active_sequence_key, self.inactive_sequence_key)
            if key is not None
        )
        self._prune_cache_branches(retained)

    def _source_stage_token(self) -> tuple[str, tuple[object, ...]]:
        return ("source_decode", tuple())

    def _encode_stage_token(self) -> tuple[str, tuple[object, ...]]:
        return ("encode_enhanced", tuple())

    def _roi_region_masks_for_sequence(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> list[np.ndarray] | None:
        for index, token in enumerate(sequence_key):
            if token[0] == "roi_extraction":
                return self.roi_region_mask_cache.get(self._artifact_cache_key(sequence_key[: index + 1]))
        return None

    def _roi_selection_for_sequence(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> ROISelection | None:
        for index, token in enumerate(sequence_key):
            if token[0] == "roi_extraction":
                return self.roi_selection_cache.get(self._artifact_cache_key(sequence_key[: index + 1]))
        return None

    def _artifact_cache_key(
        self,
        stage_prefix: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> tuple[tuple[str, tuple[object, ...]], ...]:
        return self._frame_sequence_key(stage_prefix[:-1]) + (stage_prefix[-1],)

    def _ensure_cached_artifacts(
        self,
        enabled_stages: tuple[PipelineStage, ...],
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
        default_parameters: EnhancementParameters,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> bool:
        frame_prefix: tuple[tuple[str, tuple[object, ...]], ...] = tuple()
        for index, (stage, token) in enumerate(zip(enabled_stages, sequence_key)):
            definition = BUILTIN_STAGES.require(stage.key)
            if definition.modifies_frame_data:
                frame_prefix += (token,)
                continue
            artifact_key = self._artifact_cache_key(sequence_key[: index + 1])
            frames = self.source_gray_frames if not frame_prefix else self.stage_frame_cache.get(frame_prefix)
            if frames is None:
                return False
            if cancel_callback is not None and cancel_callback():
                return False
            parameters = stage.parameters or default_parameters
            if stage.key == "roi_extraction" and artifact_key not in self.roi_selection_cache:
                roi, regions = extract_aneurysm_regions(
                    frames,
                    self.info.fps,
                    parameters,
                    getattr(self, "camera_view_mask", None),
                )
                self.roi_selection_cache[artifact_key] = roi
                encoded_ok, encoded_regions = cv2.imencode(".png", regions)
                if not encoded_ok:
                    raise RuntimeError(f"Could not cache ROI regions: {self.path}")
                self.roi_region_mask_cache[artifact_key] = [encoded_regions] * len(frames)
        return True

    def _sequence_has_roi_extraction(
        self,
        sequence_key: tuple[tuple[str, tuple[object, ...]], ...],
    ) -> bool:
        return any(token[0] == "roi_extraction" for token in sequence_key)

    def _default_stage_seconds_per_frame(self, stage_token: tuple[str, tuple[object, ...]]) -> float:
        stage_key = stage_token[0]
        if stage_key == "source_decode":
            return 0.0025
        if stage_key == "encode_enhanced":
            return 0.0018
        if stage_key == "denoise" and str(stage_token[1][0]).startswith("non-local-means"):
            return 0.0400
        if stage_key == "denoise" and str(stage_token[1][0]).startswith("tensor-nlm"):
            return 0.0550
        return BUILTIN_STAGES.require(stage_key).default_seconds_per_frame

    def _conservative_stage_seconds_per_frame(self, stage_token: tuple[str, tuple[object, ...]]) -> float:
        stage_key = stage_token[0]
        if stage_key == "source_decode":
            return 0.0035
        if stage_key == "encode_enhanced":
            return 0.0025
        if stage_key == "denoise" and str(stage_token[1][0]).startswith("non-local-means"):
            return 0.0600
        if stage_key == "denoise" and str(stage_token[1][0]).startswith("tensor-nlm"):
            return 0.0750
        return BUILTIN_STAGES.require(stage_key).conservative_seconds_per_frame

    def _estimated_stage_duration(self, stage_token: tuple[str, tuple[object, ...]], frame_count: int) -> float:
        seconds_per_frame = self.stage_duration_per_frame.get(stage_token)
        if seconds_per_frame is None:
            seconds_per_frame = max(
                self._default_stage_seconds_per_frame(stage_token),
                self._conservative_stage_seconds_per_frame(stage_token),
            )
        return seconds_per_frame * max(1, frame_count)

    def _record_stage_duration(self, stage_token: tuple[str, tuple[object, ...]], duration_seconds: float, frame_count: int) -> None:
        measured = duration_seconds / max(1, frame_count)
        previous = self.stage_duration_per_frame.get(stage_token)
        if previous is None:
            self.stage_duration_per_frame[stage_token] = measured
            return
        self.stage_duration_per_frame[stage_token] = previous * 0.4 + measured * 0.6

    def estimate_prepare_work(
        self,
        backend_id: str,
        noise_sigma: int,
        stages: EnhancementStages,
        parameters: EnhancementParameters,
    ) -> float:
        sequence_key = self._sequence_key(stages, backend_id, noise_sigma, parameters)
        frame_sequence_key = self._frame_sequence_key(sequence_key)
        frame_count = self.playback_frame_count
        work_units = 0.0
        if frame_sequence_key in self.encoded_frame_cache:
            return work_units
        if self.source_gray_frames is None:
            work_units += self._estimated_stage_duration(self._source_stage_token(), frame_count)
        reusable_prefix: tuple[tuple[str, tuple[object, ...]], ...] = tuple()
        for token in frame_sequence_key:
            candidate = reusable_prefix + (token,)
            if candidate not in self.stage_frame_cache:
                break
            reusable_prefix = candidate
        for token in frame_sequence_key[len(reusable_prefix) :]:
            work_units += self._estimated_stage_duration(token, frame_count)
        work_units += self._estimated_stage_duration(self._encode_stage_token(), frame_count)
        return work_units

    def _stage_display_name(self, stage_key: str) -> str:
        return BUILTIN_STAGES.require(stage_key).display_name

    def _apply_frame_stage(
        self,
        stage_key: str,
        frame: np.ndarray,
        parameters: EnhancementParameters,
    ) -> np.ndarray:
        if stage_key == "roi_needle_level_alignment":
            return align_frame_intensity(
                frame,
                self.roi_needle_alignment_gain,
                self.roi_needle_alignment_offset,
            )
        definition = BUILTIN_STAGES.require(stage_key)
        if definition.processor is None:
            return frame
        return definition.process_frame(
            frame,
            parameters,
            FrameContext(
                target_median=self.target_median,
                camera_view_mask=getattr(self, "camera_view_mask", None),
            ),
        )

    def prepare_enhanced_frames(
        self,
        denoiser: FrameDenoiser | None = None,
        noise_sigma: int = 10,
        batch_size: int = 4,
        stages: EnhancementStages = EnhancementStages(),
        parameters: EnhancementParameters = EnhancementParameters(),
        progress_callback: Callable[[float, float], bool] | None = None,
        stage_progress_callback: Callable[[str, int, int], bool] | None = None,
        encoded_frame_callback: Callable[[int, np.ndarray], None] | None = None,
        roi_region_mask_callback: Callable[[int, np.ndarray], None] | None = None,
        activate_result: bool = True,
        cancel_callback: Callable[[], bool] | None = None,
        frame_executor: AdaptiveFrameExecutor | None = None,
    ) -> bool:
        enabled_stages = stages.enabled_stage_instances(parameters)
        if denoiser is None and any(stage.key == "denoise" for stage in enabled_stages):
            raise ValueError("Spatial denoising requires an FFDNet backend.")
        backend_id = denoiser.backend_id if denoiser is not None else "none"
        sequence_key = self._sequence_key(stages, backend_id, noise_sigma, parameters)
        frame_sequence_key = self._frame_sequence_key(sequence_key)
        self._begin_cache_branch(sequence_key)
        stage_parameters_by_prefix: dict[tuple[tuple[str, tuple[object, ...]], ...], EnhancementParameters] = {}
        prefix: tuple[tuple[str, tuple[object, ...]], ...] = tuple()
        stage_noise_by_prefix: dict[tuple[tuple[str, tuple[object, ...]], ...], int] = {}
        for stage, token in zip(enabled_stages, sequence_key):
            prefix += (token,)
            stage_parameters_by_prefix[prefix] = stage.parameters or parameters
            stage_noise_by_prefix[prefix] = stage.noise_sigma if stage.noise_sigma is not None else noise_sigma
        artifacts_ready = self._ensure_cached_artifacts(
            enabled_stages,
            sequence_key,
            parameters,
            cancel_callback,
        )
        if frame_sequence_key in self.encoded_frame_cache and artifacts_ready:
            encoded_frames = self.encoded_frame_cache[frame_sequence_key]
            roi_region_masks = self._roi_region_masks_for_sequence(sequence_key)
            roi_selection = self._roi_selection_for_sequence(sequence_key)
            if encoded_frame_callback is not None or roi_region_mask_callback is not None:
                for index, encoded in enumerate(encoded_frames):
                    if cancel_callback is not None and cancel_callback():
                        return False
                    if roi_region_mask_callback is not None and roi_region_masks is not None:
                        roi_region_mask_callback(index, roi_region_masks[index])
                    if encoded_frame_callback is not None:
                        encoded_frame_callback(index, encoded)
            if activate_result:
                self.enhanced_frames = encoded_frames
            self.roi_region_masks = roi_region_masks
            self._activate_cache_branch(sequence_key)
            if activate_result and self._sequence_has_roi_extraction(sequence_key):
                self._activate_stage_roi_selection(roi_selection)
            elif activate_result:
                self.stage_roi_selection = None
            return True

        if frame_executor is None:
            with frame_parallel_opencv(), AdaptiveFrameExecutor() as owned_executor:
                return self.prepare_enhanced_frames(
                    denoiser,
                    noise_sigma,
                    batch_size,
                    stages,
                    parameters,
                    progress_callback,
                    stage_progress_callback,
                    encoded_frame_callback,
                    roi_region_mask_callback,
                    activate_result,
                    cancel_callback,
                    owned_executor,
                )

        frame_count = self.playback_frame_count
        reusable_frames = self.source_gray_frames
        missing_stages: list[
            tuple[
                tuple[str, tuple[object, ...]],
                tuple[tuple[str, tuple[object, ...]], ...],
                tuple[tuple[str, tuple[object, ...]], ...],
            ]
        ] = []
        frame_prefix: tuple[tuple[str, tuple[object, ...]], ...] = tuple()
        start_index = 0
        for index, token in enumerate(sequence_key):
            definition = BUILTIN_STAGES.require(token[0])
            stage_prefix = sequence_key[: index + 1]
            if definition.modifies_frame_data:
                candidate = frame_prefix + (token,)
                cached_frames = self.stage_frame_cache.get(candidate)
                if cached_frames is None:
                    break
                frame_prefix = candidate
                reusable_frames = cached_frames
            else:
                artifact_key = self._artifact_cache_key(stage_prefix)
                if (
                    token[0] == "roi_extraction"
                    and (artifact_key not in self.roi_selection_cache or artifact_key not in self.roi_region_mask_cache)
                ):
                    break
            start_index = index + 1

        frame_prefix = self._frame_sequence_key(sequence_key[:start_index])
        for index in range(start_index, len(sequence_key)):
            token = sequence_key[index]
            if BUILTIN_STAGES.require(token[0]).modifies_frame_data:
                frame_prefix += (token,)
            missing_stages.append((token, sequence_key[: index + 1], frame_prefix))

        source_missing = reusable_frames is None

        work_tokens = ([self._source_stage_token()] if source_missing else []) + [
            token for token, _, _ in missing_stages
        ] + [self._encode_stage_token()]
        estimates = [self._estimated_stage_duration(token, frame_count) for token in work_tokens]
        completed_frames = [0] * len(work_tokens)
        total_estimate = max(sum(estimates), 0.001)
        progress_lock = Lock()
        callback_lock = Lock()
        cancelled = Event()

        def report_frame(work_index: int, stage_name: str, done: int) -> bool:
            if cancel_callback is not None and cancel_callback():
                cancelled.set()
                return False
            with progress_lock:
                completed_frames[work_index] = done
                overall_done = sum(
                    estimate * min(count, frame_count) / max(1, frame_count)
                    for estimate, count in zip(estimates, completed_frames)
                )
            with callback_lock:
                stage_ok = stage_progress_callback is None or stage_progress_callback(stage_name, done, frame_count)
                progress_ok = progress_callback is None or progress_callback(overall_done, total_estimate)
            if not stage_ok or not progress_ok:
                cancelled.set()
                return False
            return True

        queues = [Queue[object](maxsize=frame_executor.max_pending) for _ in range(len(missing_stages) + 1)]
        source_output: list[np.ndarray] = []
        stage_outputs: list[list[np.ndarray]] = [[] for _ in missing_stages]
        roi_region_outputs: dict[tuple[tuple[str, tuple[object, ...]], ...], list[np.ndarray]] = {}
        roi_outputs: dict[tuple[tuple[str, tuple[object, ...]], ...], ROISelection | None] = {}
        encoded_frames: list[np.ndarray] = []

        def enqueue(queue: Queue[object], item: object) -> bool:
            while not cancelled.is_set():
                try:
                    queue.put(item, timeout=0.05)
                    return True
                except Full:
                    continue
            return False

        def close_queue(queue: Queue[object]) -> None:
            if cancelled.is_set():
                while True:
                    try:
                        queue.get_nowait()
                    except Empty:
                        break
                queue.put_nowait(STREAM_END)
                return
            enqueue(queue, STREAM_END)

        def produce_frames() -> None:
            if not source_missing:
                for index, frame in enumerate(reusable_frames or []):
                    if not enqueue(queues[0], (index, frame)):
                        break
                close_queue(queues[0])
                return

            capture = open_video_capture(self.path)
            if not capture.isOpened():
                close_queue(queues[0])
                raise RuntimeError(f"Could not precompute enhancement for video: {self.path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start_frame)
            active_seconds = 0.0
            try:
                for index in range(frame_count):
                    if cancelled.is_set():
                        break
                    started_at = perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Could not precompute enhancement for video: {self.path}")
                    gray = cv2.cvtColor(crop_frame(frame, self.crop_rect), cv2.COLOR_BGR2GRAY)
                    gray = sanitize_camera_view(gray, getattr(self, "camera_view_mask", None))
                    gray = self._remove_stationary_metal(gray)
                    gray = self._align_source_gain(gray)
                    gray = self._match_source_histogram(gray)
                    gray = self._subtract_background(gray)
                    active_seconds += perf_counter() - started_at
                    source_output.append(gray)
                    if not enqueue(queues[0], (index, gray)):
                        break
                    if not report_frame(0, "Decode source frames", index + 1):
                        break
            except Exception:
                cancelled.set()
                raise
            finally:
                capture.release()
                close_queue(queues[0])
                if not cancelled.is_set():
                    self._record_stage_duration(self._source_stage_token(), active_seconds, frame_count)

        stage_work_offset = 1 if source_missing else 0

        def run_stage(stage_index: int) -> None:
            token, stage_prefix, stage_frame_prefix = missing_stages[stage_index]
            stage_key = token[0]
            stage_parameters = stage_parameters_by_prefix[stage_prefix]
            stage_noise_sigma = stage_noise_by_prefix[stage_prefix]
            stage_name = self._stage_display_name(stage_key)
            input_queue = queues[stage_index]
            output_queue = queues[stage_index + 1]
            output = stage_outputs[stage_index]
            work_index = stage_work_offset + stage_index
            active_seconds = 0.0
            pending_frames: deque[tuple[int, Future[tuple[np.ndarray, float]]]] = deque()

            def emit(frame_index: int, frame: np.ndarray) -> bool:
                if cancelled.is_set():
                    return False
                output.append(frame)
                if not enqueue(output_queue, (frame_index, frame)):
                    return False
                return report_frame(work_index, stage_name, len(output))

            def apply_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
                started_at = perf_counter()
                transformed = self._apply_frame_stage(stage_key, frame, stage_parameters)
                return transformed, perf_counter() - started_at

            def apply_temporal(
                previous: np.ndarray,
                current: np.ndarray,
                following: np.ndarray,
            ) -> tuple[np.ndarray, float]:
                started_at = perf_counter()
                transformed = motion_aware_temporal_filter(
                    previous,
                    current,
                    following,
                    stage_parameters.temporal_motion_sigma,
                )
                return transformed, perf_counter() - started_at

            def apply_mottle(frames: list[np.ndarray]) -> tuple[np.ndarray, float]:
                started_at = perf_counter()
                transformed = reduce_quantum_mottle(
                    frames,
                    stage_parameters.mottle_similarity_sigma,
                )
                return transformed, perf_counter() - started_at

            def drain_frame() -> bool:
                nonlocal active_seconds
                frame_index, future = pending_frames.popleft()
                transformed, duration = future.result()
                active_seconds += duration
                return cancelled.is_set() or emit(frame_index, transformed)

            try:
                if stage_key == "temporal_filter":
                    previous: np.ndarray | None = None
                    current_item: tuple[int, np.ndarray] | None = None
                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            break
                        if cancelled.is_set():
                            continue
                        next_item = cast(tuple[int, np.ndarray], item)
                        if current_item is None:
                            current_item = next_item
                            continue
                        current_index, current = current_item
                        following = next_item[1]
                        pending_frames.append(
                            (
                                current_index,
                                frame_executor.submit(
                                    apply_temporal,
                                    previous if previous is not None else current,
                                    current,
                                    following,
                                ),
                            )
                        )
                        if len(pending_frames) >= frame_executor.max_workers:
                            drain_frame()
                        previous = current
                        current_item = next_item
                    if current_item is not None and not cancelled.is_set():
                        current_index, current = current_item
                        pending_frames.append(
                            (
                                current_index,
                                frame_executor.submit(
                                    apply_temporal,
                                    previous if previous is not None else current,
                                    current,
                                    current,
                                ),
                            )
                        )
                    while pending_frames:
                        drain_frame()
                elif stage_key == "quantum_mottle_filter":
                    radius = max(1, int(stage_parameters.mottle_window_radius))
                    window_size = radius * 2 + 1
                    mottle_window: deque[tuple[int, np.ndarray]] = deque()

                    def submit_mottle_window() -> None:
                        if len(mottle_window) < window_size:
                            return
                        center_index = mottle_window[radius][0]
                        frames = [frame for _, frame in mottle_window]
                        pending_frames.append(
                            (center_index, frame_executor.submit(apply_mottle, frames))
                        )

                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            break
                        if cancelled.is_set():
                            continue
                        next_item = cast(tuple[int, np.ndarray], item)
                        if not mottle_window:
                            mottle_window.extend([next_item] * radius)
                        mottle_window.append(next_item)
                        if len(mottle_window) >= window_size:
                            submit_mottle_window()
                            mottle_window.popleft()
                        if len(pending_frames) >= frame_executor.max_workers:
                            drain_frame()

                    if mottle_window and not cancelled.is_set():
                        last_item = mottle_window[-1]
                        for _ in range(radius):
                            mottle_window.append(last_item)
                            if len(mottle_window) >= window_size:
                                submit_mottle_window()
                                mottle_window.popleft()
                            if len(pending_frames) >= frame_executor.max_workers:
                                drain_frame()
                    while pending_frames:
                        drain_frame()
                elif stage_key == "denoise" and denoiser is not None:
                    batch: list[tuple[int, np.ndarray]] = []

                    def flush_batch() -> None:
                        nonlocal active_seconds
                        if not batch or cancelled.is_set():
                            return
                        denoise_input = [np.clip(frame, 0, 255).astype(np.uint8) for _, frame in batch]

                        def denoise_frames() -> tuple[list[np.ndarray], float]:
                            started_at = perf_counter()
                            result = denoiser.denoise_batch(denoise_input, stage_noise_sigma)
                            return result, perf_counter() - started_at

                        denoised_batch, duration = frame_executor.submit(denoise_frames).result()
                        active_seconds += duration
                        if len(denoised_batch) != len(batch):
                            raise RuntimeError("Denoiser returned an unexpected number of frames.")
                        for (frame_index, _), denoised in zip(batch, denoised_batch):
                            if not emit(frame_index, denoised):
                                break
                        batch.clear()

                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            flush_batch()
                            break
                        if cancelled.is_set():
                            continue
                        batch.append(cast(tuple[int, np.ndarray], item))
                        if len(batch) >= batch_size:
                            flush_batch()
                elif stage_key == "brightness_stabilization":
                    stage_frames: list[tuple[int, np.ndarray]] = []
                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            break
                        if cancelled.is_set():
                            continue
                        stage_frames.append(cast(tuple[int, np.ndarray], item))

                    if stage_frames and not cancelled.is_set():
                        frames_only = [frame for _, frame in stage_frames]
                        estimate_started_at = perf_counter()
                        gains, offsets = estimate_intensity_corrections(
                            frames_only,
                            camera_view_mask=getattr(self, "camera_view_mask", None),
                        )
                        active_seconds += perf_counter() - estimate_started_at

                        pending_stabilized: deque[tuple[int, Future[tuple[np.ndarray, float]]]] = deque()

                        def stabilize_frame(frame: np.ndarray, gain: float, offset: float) -> tuple[np.ndarray, float]:
                            started_at = perf_counter()
                            stabilized = stabilize_frame_intensity(frame, gain, offset)
                            return stabilized, perf_counter() - started_at

                        for (frame_index, frame), gain, offset in zip(stage_frames, gains, offsets):
                            pending_stabilized.append(
                                (
                                    frame_index,
                                    frame_executor.submit(stabilize_frame, frame, float(gain), float(offset)),
                                )
                            )
                            if len(pending_stabilized) >= frame_executor.max_workers:
                                frame_number, future = pending_stabilized.popleft()
                                transformed, duration = future.result()
                                active_seconds += duration
                                if cancelled.is_set() or not emit(frame_number, transformed):
                                    break

                        while pending_stabilized and not cancelled.is_set():
                            frame_number, future = pending_stabilized.popleft()
                            transformed, duration = future.result()
                            active_seconds += duration
                            if not emit(frame_number, transformed):
                                break
                elif stage_key == "camera_motion_stabilization":
                    stage_frames: list[tuple[int, np.ndarray]] = []
                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            break
                        if cancelled.is_set():
                            continue
                        stage_frames.append(cast(tuple[int, np.ndarray], item))

                    if stage_frames and not cancelled.is_set():
                        frames_only = [frame for _, frame in stage_frames]
                        estimate_started_at = perf_counter()
                        translations = estimate_camera_translations(
                            frames_only,
                            camera_view_mask=getattr(self, "camera_view_mask", None),
                        )
                        active_seconds += perf_counter() - estimate_started_at
                        pending_stabilized: deque[tuple[int, Future[tuple[np.ndarray, float]]]] = deque()

                        def stabilize_motion(frame: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, float]:
                            started_at = perf_counter()
                            stabilized = stabilize_frame_translation(frame, translation)
                            stabilized = sanitize_camera_view(
                                stabilized,
                                getattr(self, "camera_view_mask", None),
                            )
                            return stabilized, perf_counter() - started_at

                        for (frame_index, frame), translation in zip(stage_frames, translations):
                            pending_stabilized.append(
                                (
                                    frame_index,
                                    frame_executor.submit(stabilize_motion, frame, translation),
                                )
                            )
                            if len(pending_stabilized) >= frame_executor.max_workers:
                                frame_number, future = pending_stabilized.popleft()
                                transformed, duration = future.result()
                                active_seconds += duration
                                if cancelled.is_set() or not emit(frame_number, transformed):
                                    break

                        while pending_stabilized and not cancelled.is_set():
                            frame_number, future = pending_stabilized.popleft()
                            transformed, duration = future.result()
                            active_seconds += duration
                            if not emit(frame_number, transformed):
                                break
                elif stage_key == "roi_extraction":
                    stage_frames: list[tuple[int, np.ndarray]] = []
                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            break
                        if cancelled.is_set():
                            continue
                        frame_index, frame = cast(tuple[int, np.ndarray], item)
                        stage_frames.append((frame_index, frame))

                    if stage_frames and not cancelled.is_set():
                        started_at = perf_counter()
                        artifact_key = self._artifact_cache_key(stage_prefix)
                        roi, regions = extract_aneurysm_regions(
                            [frame for _, frame in stage_frames],
                            self.info.fps,
                            stage_parameters,
                            getattr(self, "camera_view_mask", None),
                        )
                        encoded_ok, encoded_regions = cv2.imencode(".png", regions)
                        if not encoded_ok:
                            raise RuntimeError(f"Could not cache ROI regions: {self.path}")
                        roi_outputs[artifact_key] = roi
                        roi_region_outputs[artifact_key] = [encoded_regions] * len(stage_frames)
                        active_seconds += perf_counter() - started_at
                        for frame_index, frame in stage_frames:
                            if roi_region_mask_callback is not None:
                                roi_region_mask_callback(frame_index, encoded_regions)
                            if not emit(frame_index, frame):
                                break
                else:
                    while True:
                        item = input_queue.get()
                        if item is STREAM_END:
                            break
                        if cancelled.is_set():
                            continue
                        frame_index, frame = cast(tuple[int, np.ndarray], item)
                        pending_frames.append((frame_index, frame_executor.submit(apply_frame, frame)))
                        if len(pending_frames) >= frame_executor.max_workers:
                            drain_frame()
                    while pending_frames:
                        drain_frame()
            except Exception:
                cancelled.set()
                raise
            finally:
                close_queue(output_queue)
                if not cancelled.is_set():
                    self._record_stage_duration(token, active_seconds, frame_count)

        encode_work_index = len(work_tokens) - 1

        def encode_frames() -> None:
            active_seconds = 0.0
            pending_frames: deque[tuple[int, Future[tuple[np.ndarray, float]]]] = deque()

            def encode_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
                enhanced = frame if frame.dtype == np.uint8 else np.clip(frame, 0, 255).astype(np.uint8)
                started_at = perf_counter()
                encoded_ok, encoded = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
                duration = perf_counter() - started_at
                if not encoded_ok:
                    raise RuntimeError(f"Could not cache enhanced video frame: {self.path}")
                return encoded, duration

            def drain_frame() -> None:
                nonlocal active_seconds
                frame_index, future = pending_frames.popleft()
                encoded, duration = future.result()
                active_seconds += duration
                if cancelled.is_set():
                    return
                encoded_frames.append(encoded)
                if encoded_frame_callback is not None:
                    encoded_frame_callback(frame_index, encoded)
                report_frame(encode_work_index, "Encode enhanced frames", len(encoded_frames))

            try:
                while True:
                    item = queues[-1].get()
                    if item is STREAM_END:
                        break
                    if cancelled.is_set():
                        continue
                    frame_index, frame = cast(tuple[int, np.ndarray], item)
                    pending_frames.append((frame_index, frame_executor.submit(encode_frame, frame)))
                    if len(pending_frames) >= frame_executor.max_workers:
                        drain_frame()
                while pending_frames:
                    drain_frame()
            except Exception:
                cancelled.set()
                raise
            finally:
                if not cancelled.is_set():
                    self._record_stage_duration(self._encode_stage_token(), active_seconds, frame_count)

        worker_count = len(missing_stages) + 2
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="enhancement-stage") as executor:
            futures = [executor.submit(produce_frames)]
            futures.extend(executor.submit(run_stage, index) for index in range(len(missing_stages)))
            futures.append(executor.submit(encode_frames))
            for future in futures:
                future.result()

        if cancelled.is_set():
            return False
        if source_missing:
            self.source_gray_frames = source_output
        for (token, _, stage_frame_prefix), output in zip(missing_stages, stage_outputs):
            if BUILTIN_STAGES.require(token[0]).modifies_frame_data:
                self.stage_frame_cache[stage_frame_prefix] = output
        self.roi_region_mask_cache.update(roi_region_outputs)
        self.roi_selection_cache.update(roi_outputs)
        self.encoded_frame_cache[frame_sequence_key] = encoded_frames

        if activate_result:
            self.enhanced_frames = self.encoded_frame_cache[frame_sequence_key]
        self.roi_region_masks = self._roi_region_masks_for_sequence(sequence_key)
        self._activate_cache_branch(sequence_key)
        if activate_result and self._sequence_has_roi_extraction(sequence_key):
            self._activate_stage_roi_selection(self._roi_selection_for_sequence(sequence_key))
        elif activate_result:
            self.stage_roi_selection = None
        return True

    def activate_prepared_sequence(
        self,
        backend_id: str,
        noise_sigma: int,
        stages: EnhancementStages,
        parameters: EnhancementParameters,
    ) -> bool:
        sequence_key = self._sequence_key(stages, backend_id, noise_sigma, parameters)
        encoded_frames = self.encoded_frame_cache.get(self._frame_sequence_key(sequence_key))
        if encoded_frames is None:
            return False
        self.enhanced_frames = encoded_frames
        self.roi_region_masks = self._roi_region_masks_for_sequence(sequence_key)
        self._activate_cache_branch(sequence_key)
        if self._sequence_has_roi_extraction(sequence_key):
            self._activate_stage_roi_selection(self._roi_selection_for_sequence(sequence_key))
        else:
            self.stage_roi_selection = None
        return True

    def _metadata_text(self) -> str:
        crop_width = self.crop_rect.width()
        crop_height = self.crop_rect.height()
        trimmed_duration = self.playback_duration
        source_suffix = " | looping live input" if self.live_input else f" | {trimmed_duration:.1f} s trimmed"
        if self.info.window_center is not None and self.info.window_width is not None:
            source_suffix += f" | DICOM WL {self.info.window_center:g} / WW {self.info.window_width:g}"
        if crop_width != self.info.width or crop_height != self.info.height:
            return f"{crop_width}x{crop_height} (auto-cropped) | {self.info.fps:.1f} fps{source_suffix}"
        return f"{self.info.width}x{self.info.height} | {self.info.fps:.1f} fps{source_suffix}"

    def _display_frame(self, frame: np.ndarray, apply_enhancement: bool | None = None) -> None:
        self.current_frame = frame
        source_frame = frame
        if (
            self.source_gain_multiplier != 1.0
            or self.source_gain_offset != 0.0
            or self.source_histogram_match_lut is not None
            or self.background_subtraction_enabled
            or self.metal_needle_mask is not None
        ):
            source_frame = self._align_source_gain(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            source_frame = self._match_source_histogram(source_frame)
            source_frame = self._remove_stationary_metal(source_frame)
            source_frame = self._subtract_background(source_frame)
            source_frame = cv2.cvtColor(source_frame, cv2.COLOR_GRAY2BGR)
        if apply_enhancement is None:
            apply_enhancement = self.enhance_display
        can_enhance = self.enhanced_frames is not None and 0 <= self.current_frame_index < len(self.enhanced_frames)
        enhanced_frame = source_frame
        if apply_enhancement and can_enhance:
            enhanced = cv2.imdecode(self.enhanced_frames[self.current_frame_index], cv2.IMREAD_GRAYSCALE)
            if enhanced is not None:
                enhanced_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                can_overlay = (
                    self.roi_region_overlay_display
                    and self.roi_region_masks is not None
                    and self.current_frame_index < len(self.roi_region_masks)
                )
                if can_overlay:
                    mask = cv2.imdecode(self.roi_region_masks[self.current_frame_index], cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        enhanced_frame = overlay_roi_regions(
                            enhanced_frame,
                            mask,
                            (self.color.blue(), self.color.green(), self.color.red()),
                        )
                if self.roi_region_overlay_display and self.needle_segmentation_mask is not None:
                    enhanced_frame = overlay_needle_mask(enhanced_frame, self.needle_segmentation_mask)
        left_frame = self.temporal_change_heatmap if self.temporal_change_display and self.temporal_change_heatmap is not None else source_frame
        left_label = "Contrast residence" if self.temporal_change_display and self.temporal_change_heatmap is not None else "Source"
        self.display.set_frames(left_frame, enhanced_frame, left_label)

    def read_next(self, playback: bool = False) -> bool:
        if not self.live_input and self.current_frame_index >= self.playback_frame_count - 1:
            return False
        ok, frame = self.capture.read()
        if not ok and self.live_input:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if ok:
            self.current_frame_index += 1
            self._display_frame(crop_frame(frame, self.crop_rect), apply_enhancement=self.enhance_display)
        return ok

    def apply_live_enhancement(
        self,
        stages: EnhancementStages,
        default_parameters: EnhancementParameters,
        denoiser: FrameDenoiser | None = None,
        noise_sigma: int = 10,
    ) -> None:
        if not self.live_input or self.current_frame is None:
            return
        source = self.current_frame
        enhanced: np.ndarray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        for stage in stages.enabled_stage_instances(default_parameters):
            parameters = stage.parameters or default_parameters
            if stage.key in {
                "temporal_filter",
                "quantum_mottle_filter",
                "brightness_stabilization",
                "camera_motion_stabilization",
                "roi_extraction",
            }:
                continue
            if stage.key == "denoise" and denoiser is not None:
                result = denoiser.denoise_batch([np.clip(enhanced, 0, 255).astype(np.uint8)], stage.noise_sigma or noise_sigma)
                if result:
                    enhanced = result[0]
                continue
            enhanced = self._apply_frame_stage(stage.key, enhanced, parameters)

        output = cv2.cvtColor(np.clip(enhanced, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.display.set_frames(source, output)

    def seek(self, frame_index: int) -> bool:
        frame_index = max(0, min(frame_index, self.playback_frame_count - 1))
        if frame_index == self.current_frame_index and self.current_frame is not None:
            self._display_frame(self.current_frame)
            return True
        if frame_index == self.current_frame_index + 1:
            return self.read_next()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.trim_start_frame + frame_index)
        ok, frame = self.capture.read()
        if ok:
            self.current_frame_index = frame_index
            self._display_frame(crop_frame(frame, self.crop_rect))
        return ok

    def roi(self) -> QRect | None:
        return self.display.roi()

    def roi_mask(self) -> np.ndarray | None:
        return self.display.roi_mask()

    def _activate_stage_roi_selection(self, roi_selection: ROISelection | None) -> None:
        self.stage_roi_selection = roi_selection
        block_signals = getattr(self.display, "blockSignals", None)
        previous = block_signals(True) if block_signals is not None else None
        try:
            if roi_selection is None:
                self.display.set_roi(None)
                return
            self.display.set_roi(roi_selection.rect, roi_selection.mask)
        finally:
            if block_signals is not None:
                block_signals(previous)

    def has_stage_roi_mask(self) -> bool:
        return self.stage_roi_selection is not None and self.stage_roi_selection.mask is not None

    def auto_detect_aneurysm(self, gray_frames: list[np.ndarray] | None = None) -> QRect | None:
        source_frames = gray_frames if gray_frames is not None else self._sample_cropped_gray_frames()
        start = self.trim_start_frame
        end = min(len(source_frames), start + self.playback_frame_count)
        roi = detect_aneurysm_roi(
            source_frames[start:end],
            self.info.fps,
            camera_view_mask=getattr(self, "camera_view_mask", None),
        )
        if roi is None:
            self.display.set_roi(None)
            return None
        self.display.set_roi(roi.rect, roi.mask)
        return roi.rect

    def set_video(self, path: Path) -> None:
        self.capture.release()
        self.path = path
        self.info = probe_video(path)
        self.crop_rect = self._full_frame_rect()
        self._auto_crop_rect_cache = None
        self._camera_field_mask_cache = None
        self._trim_start_cache.clear()
        self.source_pipeline_configuration = None
        self.background_reference = None
        self.metal_needle_mask = None
        self.camera_view_mask = None
        self.source_gain_multiplier = 1.0
        self.source_gain_offset = 0.0
        self.source_histogram_match_lut = None
        self.capture = open_video_capture(path)
        self.path_label.setText(path.name)
        self.set_trim_window(0)
        self.meta_label.setText(self._metadata_text())
        self._activate_stage_roi_selection(None)
        self.display.set_comparison_enabled(self.comparison_display)
        self.seek(0)

    def set_enhancement(self, enabled: bool, frame_index: int) -> None:
        self.enhance_display = enabled
        self.seek(frame_index)

    def set_comparison(self, enabled: bool, frame_index: int) -> None:
        self.comparison_display = enabled
        self.temporal_change_display = False
        self.display.set_comparison_enabled(enabled)
        self.seek(frame_index)

    def set_temporal_change_heatmap(self, heatmap: np.ndarray | None) -> None:
        self.temporal_change_heatmap = None if heatmap is None else heatmap.copy()

    def set_temporal_change_display(self, enabled: bool, frame_index: int) -> None:
        self.temporal_change_display = enabled and self.temporal_change_heatmap is not None
        self.comparison_display = self.temporal_change_display
        self.display.set_comparison_enabled(self.comparison_display)
        self.seek(frame_index)

    def set_roi_region_overlay(self, enabled: bool, frame_index: int) -> None:
        self.roi_region_overlay_display = enabled
        self.seek(frame_index)

    def set_needle_segmentation_mask(self, mask: np.ndarray | None, frame_index: int) -> None:
        self.needle_segmentation_mask = None if mask is None else mask.copy()
        self.seek(frame_index)

    def clear_enhancement_cache(self) -> None:
        self.enhanced_frames = None
        self.report_encoded_frames = None
        self.temporal_change_heatmap = None
        self.temporal_change_display = False
        self.roi_region_masks = None
        self.needle_segmentation_mask = None
        self.source_gray_frames = None
        self.stage_roi_selection = None
        self.stage_frame_cache.clear()
        self.encoded_frame_cache.clear()
        self.roi_region_mask_cache.clear()
        self.roi_selection_cache.clear()
        self.temporal_change_map_cache.clear()
        self.active_sequence_key = None
        self.inactive_sequence_key = None
        self.stage_duration_per_frame.clear()

    def encoded_analysis_frames(
        self,
        backend_id: str,
        noise_sigma: int,
        stages: EnhancementStages,
        parameters: EnhancementParameters,
    ) -> list[np.ndarray] | None:
        sequence_key = self._sequence_key(stages, backend_id, noise_sigma, parameters)
        frame_sequence_key = self._frame_sequence_key(sequence_key)
        if not frame_sequence_key:
            return None
        return self.encoded_frame_cache.get(frame_sequence_key)

    def analysis_frames(
        self,
        backend_id: str,
        noise_sigma: int,
        stages: EnhancementStages,
        parameters: EnhancementParameters,
    ) -> list[np.ndarray] | None:
        encoded_frames = self.encoded_analysis_frames(backend_id, noise_sigma, stages, parameters)
        if encoded_frames is None:
            sequence_key = self._sequence_key(stages, backend_id, noise_sigma, parameters)
            if not self._frame_sequence_key(sequence_key):
                return self.source_gray_frames
            return None
        frames = [cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) for encoded in encoded_frames]
        return frames if all(frame is not None for frame in frames) else None

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


class EnhancementProgressPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("enhancementProgressPanel")
        self.setFrameShape(QFrame.Shape.NoFrame)

        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(12, 10, 12, 10)
        panel_layout.setSpacing(7)

        self.message_label = QLabel("Preparing enhanced video...")
        self.message_label.setObjectName("enhancementProgressLabel")
        self.message_label.setWordWrap(True)
        self.total_label = QLabel("Overall progress")
        self.total_label.setObjectName("subtleLabel")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.panel_labels = [QLabel("Video 1"), QLabel("Video 2")]
        self.panel_progress_bars = [QProgressBar(), QProgressBar()]
        self._active_labels = ["Video 1", "Video 2"]
        self._panel_count = 2
        for label, progress_bar in zip(self.panel_labels, self.panel_progress_bars):
            label.setObjectName("subtleLabel")
            progress_bar.setRange(0, 1000)
            progress_bar.setValue(0)
            progress_bar.setTextVisible(True)

        panel_layout.addWidget(self.message_label)
        panel_layout.addWidget(self.total_label)
        panel_layout.addWidget(self.progress_bar)
        for label, progress_bar in zip(self.panel_labels, self.panel_progress_bars):
            panel_layout.addWidget(label)
            panel_layout.addWidget(progress_bar)

        self.hide()

    def _overall_units(self, value: float) -> int:
        return max(0, round(max(0.0, value) * 1000.0))

    def configure_panels(self, labels: list[str]) -> None:
        self._panel_count = max(1, min(2, len(labels))) if labels else 1
        defaults = [f"Video {index + 1}" for index in range(self._panel_count)]
        self._active_labels = [
            labels[index] if index < len(labels) else defaults[index]
            for index in range(self._panel_count)
        ]
        for index, (label, progress_bar) in enumerate(zip(self.panel_labels, self.panel_progress_bars)):
            is_visible = index < self._panel_count
            label.setVisible(is_visible)
            progress_bar.setVisible(is_visible)
            if is_visible:
                label.setText(self._active_labels[index])

    def begin(self, message: str) -> None:
        self.message_label.setText(message)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        for index in range(self._panel_count):
            progress_bar = self.panel_progress_bars[index]
            self.panel_labels[index].setText(self._active_labels[index])
            progress_bar.setRange(0, 0)
            progress_bar.setValue(0)
        self.show()

    def set_progress(self, value: float, maximum: float) -> None:
        self.progress_bar.setRange(0, max(1, self._overall_units(maximum)))
        self.progress_bar.setValue(min(self._overall_units(value), self.progress_bar.maximum()))

    def set_panel_progress(self, panel_index: int, stage_message: str, value: float, maximum: float) -> None:
        if panel_index < 0 or panel_index >= self._panel_count:
            return
        label = self._active_labels[panel_index]
        self.panel_labels[panel_index].setText(f"{label}: {stage_message}" if stage_message else label)
        progress_bar = self.panel_progress_bars[panel_index]
        progress_bar.setRange(0, max(1, self._overall_units(maximum)))
        progress_bar.setValue(min(self._overall_units(value), progress_bar.maximum()))

    def finish(self) -> None:
        self.hide()


class StageDrawer(QFrame):
    enabledChanged = Signal(int)
    reorderRequested = Signal(str, str)
    dragStarted = Signal(object, QPoint)
    dragMoved = Signal(QPoint)
    dragFinished = Signal(QPoint)
    optionsRequested = Signal(object, QPoint)

    def __init__(self, stage_key: str, title: str, stage_index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stageDrawer")
        self.stage_key = stage_key
        self.stage_title = title

        self.enable_button = QToolButton()
        self.enable_button.setObjectName("stageEnableButton")
        self.enable_button.setCheckable(True)
        self.enable_button.setFixedSize(32, 32)
        self.enable_button.setToolTip("Enable stage")
        self.stage_label = QLabel()
        self.stage_label.setObjectName("stageLabel")
        self.grab_handle = QLabel("||")
        self.grab_handle.setObjectName("stageGrabHandle")
        self.grab_handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grab_handle.setFixedSize(24, 32)
        self.grab_handle.setToolTip("Drag to reorder stage")
        self.expand_button = QToolButton()
        self.expand_button.setObjectName("stageExpandButton")
        self.expand_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.expand_button.setArrowType(Qt.ArrowType.RightArrow)
        self.expand_button.setCheckable(True)
        self.expand_button.setChecked(False)
        self.expand_button.setFixedSize(32, 32)
        self.expand_button.setToolTip("Show stage options")
        self.options_button = QToolButton()
        self.options_button.setObjectName("stageOptionsButton")
        self.options_button.setText("...")
        self.options_button.setFixedSize(32, 32)
        self.options_button.setToolTip("Stage actions")
        self._set_enabled_icon(False)
        self.set_stage_index(stage_index)
        self._drag_start_pos = QPoint()
        self._drag_from_handle = False
        self._is_dragging = False
        self._drag_enabled = True

        self.header = QWidget()
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.header.setToolTip("Show or hide stage options")
        self.header.installEventFilter(self)
        self.stage_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header = QHBoxLayout(self.header)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.grab_handle)
        header.addWidget(self.enable_button)
        header.addWidget(self.stage_label, 1)
        header.addWidget(self.options_button)
        header.addWidget(self.expand_button)

        self.content = QWidget()
        self.content.setVisible(False)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(6)

        self.status_label = QLabel()
        self.status_label.setVisible(False)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("stageStatusLabel")
        self.status_label.setContentsMargins(8, 0, 8, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(self.header)
        layout.addWidget(self.status_label)
        layout.addWidget(self.content)

        self.enable_button.toggled.connect(self._set_enabled_icon)
        self.enable_button.toggled.connect(lambda enabled: self.enabledChanged.emit(int(enabled)))
        self.expand_button.toggled.connect(self._set_expanded)
        self.options_button.clicked.connect(
            lambda: self.optionsRequested.emit(
                self,
                self.options_button.mapToGlobal(self.options_button.rect().bottomLeft()),
            )
        )

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            watched is self.header
            and event.type() == QEvent.Type.MouseButtonRelease
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.MouseButton.LeftButton
            and not self._is_header_control_at(event.position().toPoint())
        ):
            self.expand_button.toggle()
            return True
        return super().eventFilter(watched, event)

    def _is_header_control_at(self, point: QPoint) -> bool:
        widget = self.header.childAt(point)
        while widget is not None:
            if widget in (self.grab_handle, self.enable_button, self.options_button, self.expand_button):
                return True
            widget = widget.parentWidget()
        return False

    def _set_enabled_icon(self, enabled: bool) -> None:
        color = QColor("#14b8a6" if enabled else "#64748b")
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(5, 5, 22, 22))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setBrush(Qt.GlobalColor.transparent)
        painter.drawEllipse(QRectF(9, 9, 14, 14))
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(color, 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(16, 4, 16, 14)
        painter.end()
        self.enable_button.setIcon(QIcon(pixmap))
        self.enable_button.setToolTip("Disable stage" if enabled else "Enable stage")

    def _is_on_grab_handle(self, point: QPoint) -> bool:
        widget = self.childAt(point)
        while widget is not None:
            if widget is self.grab_handle:
                return True
            widget = widget.parentWidget()
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._is_on_grab_handle(event.position().toPoint()):
            self._drag_start_pos = event.position().toPoint()
            self._drag_from_handle = True
            event.accept()
            return
        self._drag_from_handle = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_enabled
            and self._drag_from_handle
            and (event.buttons() & Qt.MouseButton.LeftButton)
            and (event.position().toPoint() - self._drag_start_pos).manhattanLength() >= QApplication.startDragDistance()
        ):
            if not self._is_dragging:
                self._is_dragging = True
                self.grabMouse()
                self.grab_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.dragStarted.emit(self, event.globalPosition().toPoint())
            self.dragMoved.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_dragging:
            self.dragFinished.emit(event.globalPosition().toPoint())
            self.releaseMouse()
            self._is_dragging = False
        self._drag_from_handle = False
        self.grab_handle.setCursor(Qt.CursorShape.OpenHandCursor if self._drag_enabled else Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _set_expanded(self, expanded: bool) -> None:
        self.expand_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)
        if self.layout() is not None:
            self.layout().invalidate()
        self.updateGeometry()
        parent = self.parentWidget()
        while parent is not None:
            if parent.layout() is not None:
                parent.layout().invalidate()
                parent.layout().activate()
            parent.updateGeometry()
            parent = parent.parentWidget()

    def set_stage_index(self, stage_index: int) -> None:
        self.stage_label.setText(self.stage_title)

    def set_reorder_enabled(self, drag_enabled: bool) -> None:
        self._drag_enabled = drag_enabled
        self.grab_handle.setEnabled(drag_enabled)
        self.grab_handle.setCursor(Qt.CursorShape.OpenHandCursor if drag_enabled else Qt.CursorShape.ArrowCursor)
        self.grab_handle.setToolTip("Drag to reorder stage" if drag_enabled else "This stage is fixed")

    def set_status(self, message: str | None, is_error: bool = False) -> None:
        if not message:
            self.status_label.clear()
            self.status_label.setVisible(False)
            return
        color = "#fca5a5" if is_error else "#9fb0c6"
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setVisible(True)


class CollapsibleDrawer(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("pipelineDrawer")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("pipelineDrawerTitle")

        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("pipelineDrawerToggle")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.ArrowType.LeftArrow)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setFixedSize(32, 32)
        self.toggle_button.setToolTip("Hide pipeline panel")

        self.header = QWidget()
        self.header.setObjectName("pipelineDrawerHeader")
        self.header.setMinimumHeight(44)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(14, 6, 14, 6)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.toggle_button)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 10, 0, 12)
        self.content_layout.setSpacing(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self.toggle_button.toggled.connect(self._set_expanded)
        self._set_expanded(self.toggle_button.isChecked())

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(Qt.ArrowType.LeftArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle_button.setToolTip("Hide pipeline panel" if expanded else "Show pipeline panel")
        self.title_label.setVisible(expanded)
        self.content.setVisible(expanded)

    def set_expanded(self, expanded: bool) -> None:
        if self.toggle_button.isChecked() != expanded:
            self.toggle_button.blockSignals(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.blockSignals(False)
        self._set_expanded(expanded)

    def header_height(self) -> int:
        return self.header.sizeHint().height()

    def header_width(self) -> int:
        return self.header.sizeHint().width()

    def collapsed_width(self) -> int:
        return max(42, self.toggle_button.sizeHint().width() + 10)


class GraphDrawer(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("graphDrawer")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("graphDrawerTitle")

        self.drag_handle = QLabel("||")
        self.drag_handle.setObjectName("graphDrawerHandle")
        self.drag_handle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drag_handle.setToolTip("Drag the drawer edge to resize the graph area")

        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("graphDrawerToggle")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.toggle_button.setArrowType(Qt.ArrowType.UpArrow)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setFixedSize(32, 32)
        self.toggle_button.setToolTip("Show graph panel")

        self.header = QWidget()
        self.header.setObjectName("graphDrawerHeader")
        self.header.setMinimumHeight(40)
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(14, 6, 14, 6)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.drag_handle)
        header_layout.addWidget(self.title_label, 1)
        header_layout.addWidget(self.toggle_button)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.content)

        self.toggle_button.toggled.connect(self._set_expanded)
        self._set_expanded(self.toggle_button.isChecked())

    def _set_expanded(self, expanded: bool) -> None:
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.UpArrow)
        self.toggle_button.setToolTip("Hide graph panel" if expanded else "Show graph panel")
        self.content.setVisible(expanded)

    def set_expanded(self, expanded: bool) -> None:
        if self.toggle_button.isChecked() != expanded:
            self.toggle_button.blockSignals(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.blockSignals(False)
        self._set_expanded(expanded)

    def header_height(self) -> int:
        return self.header.sizeHint().height()


class ContrastWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Contrast Residence Analyzer")
        self.resize(1500, 940)
        self._stream_processor: object | None = None
        self._stream_service: object | None = None
        self._stream_server: object | None = None
        self._stream_server_thread: Thread | None = None
        self._network_stream_display: VideoDisplay | None = None
        self._network_stream_frame_id = 0
        self._network_stream_geometry_revision: int | None = None
        self._live_measurements: deque[tuple[float, float, float, float, float]] = deque()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance_frame)
        self.is_playing = False
        self.current_frame_index = 0
        self.results: dict[str, AnalysisResult] = {}
        self.frame_brightness_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.needle_brightness_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.needle_brightness_baselines: dict[str, float] = {}
        self.parent_vessel_roi_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.background_roi_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.roi_needle_alignment_results: dict[str, ROIAlignmentPlotResult] = {}
        self.temporal_change_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.deep_denoisers: dict[str, FrameDenoiser] = {}
        self._enhancement_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="enhancement-coordinator")
        self._enhancement_future: Future[bool] | None = None
        self._enhancement_cancel: Event | None = None
        self._enhancement_active_request: EnhancementRequest | None = None
        self._enhancement_pending_request: EnhancementRequest | None = None
        self._analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-coordinator")
        self._analysis_future: Future[PipelineAnalysisResult] | None = None
        self._analysis_cancel: Event | None = None
        self._analysis_active_request: PipelineAnalysisRequest | None = None
        self._enhancement_generation = 0
        self._enhancement_frame_events: SimpleQueue[tuple[int, int, int, np.ndarray]] = SimpleQueue()
        self._roi_region_mask_events: SimpleQueue[tuple[int, int, int, np.ndarray]] = SimpleQueue()
        self._source_pipeline_events: SimpleQueue[tuple[int, list[SourcePipelineState], Event]] = SimpleQueue()
        self._enhancement_progress_lock = Lock()
        self._enhancement_progress_values = [0.0, 0.0]
        self._enhancement_progress_totals = [1.0, 1.0]
        self._enhancement_stage_messages = ["Waiting", "Waiting"]
        self._enhancement_message = "Preparing enhanced videos..."
        self._loading_config = False
        self._manual_roi_circles: list[tuple[int, int, int] | None] = [None, None]
        self._parent_vessel_rois: list[tuple[int, int, float] | None] = [None, None]
        self._background_rois: list[tuple[int, int] | None] = [None, None]
        self.enhancement_poll_timer = QTimer(self)
        self.enhancement_poll_timer.setInterval(30)
        self.enhancement_poll_timer.timeout.connect(self._poll_enhancement)
        self.network_stream_poll_timer = QTimer(self)
        self.network_stream_poll_timer.setInterval(30)
        self.network_stream_poll_timer.timeout.connect(self._poll_network_stream)

        self.pre_panel: VideoPanel | None = None
        self.post_panel: VideoPanel | None = None
        self.panels: list[VideoPanel] = []
        self.active_mode = MODE_SINGLE
        self.pending_mode: str | None = None
        self.pending_video_paths: list[Path | None] = []
        self.video_placeholders: list[VideoDropPlaceholder | None] = []
        self.pending_preview_panels: list[VideoPanel | None] = []
        self.pending_slot_labels: list[str] = []

        self.source_max_frame = 0
        self.max_frame = 0
        self.fps = 30.0
        self.playback_speed = 1.0
        self.play_interval_ms = self._play_interval_ms()
        self._sync_trimmed_video_window()

        self._build_actions()
        self._build_ui()
        self._load_default_pipeline_settings()
        self._update_temporal_alignment_controls()
        self._apply_source_pipeline_stages()
        self._apply_style()
        self.on_enhancement_settings_changed()
        self.set_display_enhancement(False)
        self.update_time_label()
        self._update_stage_statuses()
        self._set_video_controls_enabled(False)

    def _start_desktop_stream_service(self) -> bool:
        from stream_server import RawFrameRecorder, LiveStreamProcessor, StreamService, StreamSettings, create_http_server

        if self._stream_server is not None:
            return True
        settings = StreamSettings()
        processor = LiveStreamProcessor(
            EnhancementStages(),
            EnhancementParameters(),
            self.denoise_strength_spin.value(),
            settings.crop_sample_frames,
            settings.jpeg_quality,
            self._has_enabled_stage("auto_crop"),
            auto_crop_size_offset=self.auto_crop_size_offset_spin.value(),
            dsa_mask_delay_frames=round(settings.recording_fps),
        )
        service = StreamService(
            processor,
            settings.max_frame_bytes,
            RawFrameRecorder(settings.recording_directory, settings.recording_fps),
        )
        try:
            server = create_http_server(settings, service)
        except OSError as exc:
            LOGGER.exception("Could not start desktop stream service")
            QMessageBox.critical(self, "Could not start stream service", str(exc))
            return False
        self._stream_processor = processor
        self._stream_service = service
        self._stream_server = server
        self._stream_server_thread = Thread(
            target=server.serve_forever,
            name="desktop-stream-server",
            daemon=True,
        )
        self._stream_server_thread.start()
        self._refresh_desktop_stream_pipeline()
        LOGGER.info("Desktop stream service listening on http://%s:%s", settings.host, settings.port)
        self.statusBar().showMessage(
            f"Desktop stream service listening on http://{settings.host}:{settings.port}"
        )
        return True

    def _refresh_desktop_stream_pipeline(self) -> None:
        processor = self._stream_processor
        if processor is None:
            return
        parameters = self.enhancement_parameters()
        configured_stages = self.enhancement_stages()
        dsa_enabled = self.background_subtraction_stage_check.isChecked()
        live_stages = EnhancementStages(
            instances=tuple(
                (
                    (PipelineStage("background_subtraction", True, parameters),)
                    if dsa_enabled
                    else ()
                )
                + tuple(
                    stage
                    for stage in configured_stages.instances
                    if not stage.enabled or BUILTIN_STAGES.require(stage.key).supports_live(stage.parameters or parameters)
                )
            )
        )
        processor.configure(
            live_stages,
            parameters,
            self.denoise_strength_spin.value(),
            self._has_enabled_stage("auto_crop"),
            self._live_denoiser_for(live_stages),
            self.auto_crop_size_offset_spin.value(),
            self.background_video_one_level_spin.value(),
        )

    def _set_video_controls_enabled(self, enabled: bool) -> None:
        live_input = self.active_mode == MODE_LIVE
        for widget in self.playback_controls:
            widget.setVisible(not live_input)
            widget.setEnabled(enabled and not live_input)
        self.live_recording_controls.setVisible(live_input)
        self.live_recording_spacer.setVisible(live_input)
        self.live_device_name_edit.setEnabled(enabled and live_input)
        self.live_test_identifier_edit.setEnabled(enabled and live_input)
        self.live_phase_toggle.setEnabled(enabled and live_input)
        self.live_export_toggle.setEnabled(enabled and live_input)
        self.compare_view_check.setEnabled(enabled)
        self.overlay_mask_check.setEnabled(
            enabled
            and (
                self._has_enabled_stage("roi_extraction")
                or self._has_enabled_stage("needle_segmentation")
            )
        )
        self.open_pre_action.setEnabled(enabled and bool(self.pre_panel))
        self.open_post_action.setEnabled(enabled and bool(self.post_panel))
        if live_input:
            self._update_live_recording_name()

    def _update_live_recording_name(self) -> None:
        device_name = self.live_device_name_edit.text()
        test_identifier = self.live_test_identifier_edit.text()
        phase = "post" if self.live_phase_toggle.isChecked() else "pre"
        service = self._stream_service
        configure_recording = getattr(service, "configure_recording", None)
        path = configure_recording(device_name, test_identifier, phase) if configure_recording is not None else None
        if path is not None:
            self.live_recording_name_label.setText(path.name)
            return
        device = device_name.strip() or "device"
        test = test_identifier.strip() or "test"
        self.live_recording_name_label.setText(f"{device}_{test}_{phase}_0.avi")

    def _set_live_recording_enabled(self, enabled: bool) -> None:
        service = self._stream_service
        set_recording_enabled = getattr(service, "set_recording_enabled", None)
        if set_recording_enabled is not None:
            set_recording_enabled(enabled)

    def _set_live_incompatible_stages_enabled(self, enabled: bool) -> None:
        for stage_key in (
            "temporal_alignment",
            "brightness_stabilization",
            "camera_motion_stabilization",
            "roi_extraction",
            "temporal_filter",
            "quantum_mottle_filter",
            "needle_segmentation",
            "temporal_change_heatmap",
        ):
            for drawer in self._stage_drawers(stage_key):
                if not enabled:
                    drawer.enable_button.blockSignals(True)
                    drawer.enable_button.setChecked(False)
                    drawer.enable_button.blockSignals(False)
                    drawer.enable_button.setToolTip("Unavailable for live camera input")
                else:
                    drawer.enable_button.setToolTip("Enable stage")
                drawer.enable_button.setEnabled(enabled)

    def _open_video_file(self, title: str) -> Path | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(ROOT),
            "Video files (*.mov *.mp4 *.avi *.mkv);;DICOM cine files (*);;All files (*)",
        )
        return Path(path) if path else None

    def _clear_video_panels(self) -> None:
        self._clear_network_stream_display()
        for panel in self.panels:
            self.video_layout.removeWidget(panel)
            panel.hide()
            panel.deleteLater()
        self.panels = []
        self.pre_panel = None
        self.post_panel = None

    def _clear_network_stream_display(self) -> None:
        self.network_stream_poll_timer.stop()
        if self._network_stream_display is not None:
            self.video_layout.removeWidget(self._network_stream_display)
            self._network_stream_display.hide()
            self._network_stream_display.deleteLater()
            self._network_stream_display = None
        self._network_stream_frame_id = 0
        self._network_stream_geometry_revision = None
        self._live_measurements.clear()

    def _set_video_panels(self, videos: list[Path], live_input: bool = False) -> None:
        if not videos:
            self._clear_video_panels()
            self._clear_video_placeholders()
            self.active_mode = MODE_SINGLE
            self._update_temporal_alignment_controls()
            self.enhancement_progress.configure_panels(["Video 1"])
            self._set_video_controls_enabled(False)
            self.video_stack.setCurrentWidget(self.mode_selection_page)
            self._sync_trimmed_video_window()
            self._set_playback_limit(self.source_max_frame)
            self.current_frame_index = 0
            self.update_time_label()
            return

        self.pause()
        self.pending_mode = None
        self.pending_video_paths = []
        self._clear_video_placeholders()
        self._clear_video_panels()
        labels = ["Pre-deployment", "Post-deployment"] if len(videos) > 1 else ["Video"]
        for index, path in enumerate(videos):
            color = PANEL_COLORS[min(index, len(PANEL_COLORS) - 1)]
            panel = (
                VideoPanel(labels[index], color, path, live_input=True)
                if live_input
                else VideoPanel(labels[index], color, path)
            )
            panel.roiChanged.connect(self.on_roi_changed)
            roi_drawn = getattr(panel, "roiDrawn", None)
            if roi_drawn is not None:
                roi_drawn.connect(lambda roi, panel_index=index: self._on_manual_roi_drawn(panel_index, roi))
            parent_vessel_roi_drawn = getattr(panel, "parentVesselRoiDrawn", None)
            if parent_vessel_roi_drawn is not None:
                parent_vessel_roi_drawn.connect(
                    lambda center, panel_index=index: self._on_parent_vessel_roi_drawn(panel_index, center)
                )
            background_roi_drawn = getattr(panel, "backgroundRoiDrawn", None)
            if background_roi_drawn is not None:
                background_roi_drawn.connect(
                    lambda center, panel_index=index: self._on_background_roi_drawn(panel_index, center)
                )
            self.video_layout.addWidget(panel)
            self.panels.append(panel)
        self.pre_panel = self.panels[0] if self.panels else None
        self.post_panel = self.panels[1] if len(self.panels) > 1 else None
        self._manual_roi_circles = [None] * len(self.panels)
        self._parent_vessel_rois = [None] * len(self.panels)
        self._background_rois = [None] * len(self.panels)
        self.active_mode = MODE_LIVE if live_input else (MODE_COMPARISON if len(self.panels) > 1 else MODE_SINGLE)
        self._update_temporal_alignment_controls()
        self.open_pre_action.setText(f"Open {self.pre_panel.label.lower()} video..." if self.pre_panel else "Open video 1...")
        self.open_post_action.setText(f"Open {self.post_panel.label.lower()} video..." if self.post_panel else "Open video 2...")
        self.compare_view_check.setEnabled(True)
        self.enhancement_progress.configure_panels([panel.label for panel in self.panels])
        self.video_stack.setCurrentWidget(self.video_row)
        self._update_video_stack_geometry()
        self.results.clear()
        self._sync_trimmed_video_window()
        self._set_playback_limit(self.source_max_frame)
        self.current_frame_index = -1
        self.set_frame_index(0)
        self.clear_plots_and_metrics()
        self._update_stage_statuses()
        self._set_video_controls_enabled(True)
        self._set_live_incompatible_stages_enabled(not live_input)
        if live_input:
            self._apply_source_pipeline_stages()
        self.on_compare_view_toggled(self.compare_view_check.isChecked())
        if len(self.panels) == 1:
            self.pre_card.title.setText("Residence")
            self.post_card.title.setText("Comparison")
            self.delta_card.title.setText("Difference")
        else:
            self.pre_card.title.setText("Pre residence")
            self.post_card.title.setText("Post residence")
            self.delta_card.title.setText("Difference")
        if not self._loading_config and self._pipeline_has_active_stage():
            if live_input:
                self._render_live_frame()
            elif self.active_mode == MODE_COMPARISON and len(self.panels) == 2:
                QTimer.singleShot(0, self._rebuild_enhancement_pipeline_if_comparison_ready)
            else:
                self.rebuild_enhancement_pipeline()

    def _rebuild_enhancement_pipeline_if_comparison_ready(self) -> None:
        if self.active_mode != MODE_COMPARISON or len(self.panels) != 2:
            return
        if self.video_stack.currentWidget() is not self.video_row:
            return
        if any(panel.current_frame is None for panel in self.panels):
            QTimer.singleShot(0, self._rebuild_enhancement_pipeline_if_comparison_ready)
            return
        self.rebuild_enhancement_pipeline()

    def _select_mode_and_videos(self, mode: str) -> bool:
        if mode == MODE_LIVE:
            if not self._start_desktop_stream_service():
                return False
            self._activate_network_stream_mode()
            return True
        self._show_video_placeholders(mode)
        return True

    def _activate_network_stream_mode(self) -> None:
        self.pause()
        self._clear_video_panels()
        self._clear_video_placeholders()
        self.pending_mode = None
        self.pending_video_paths = []
        self.active_mode = MODE_LIVE
        self.compare_view_check.setChecked(True)
        self._network_stream_display = VideoDisplay("Network live camera", PANEL_COLORS[0], self.video_row)
        self._network_stream_display.set_comparison_enabled(self.compare_view_check.isChecked())
        self._network_stream_display.roiChanged.connect(lambda _roi: self.on_roi_changed())
        self.video_layout.addWidget(self._network_stream_display)
        self.video_layout.setStretch(0, 1)
        self._update_temporal_alignment_controls()
        self._set_live_incompatible_stages_enabled(False)
        self._set_video_controls_enabled(True)
        self.compare_view_check.setEnabled(True)
        self.video_stack.setCurrentWidget(self.video_row)
        self._update_video_stack_geometry()
        self._sync_trimmed_video_window()
        self._set_playback_limit(self.source_max_frame)
        self.current_frame_index = 0
        self.update_time_label()
        self._refresh_desktop_stream_pipeline()
        self.network_stream_poll_timer.start()
        self._poll_network_stream()
        self.statusBar().showMessage(
            "Network live stream active. Drag to select an ROI; live plots retain the latest 60 seconds."
        )

    def _poll_network_stream(self) -> None:
        if self.active_mode != MODE_LIVE or self._network_stream_display is None:
            return
        service = self._stream_service
        if service is None:
            return
        snapshot = service.latest_analysis_frames()
        if not isinstance(snapshot, tuple) or len(snapshot) != 6:
            return
        (
            frame_id,
            encoded_source,
            encoded_enhanced,
            crop_rect,
            camera_view_mask,
            geometry_revision,
        ) = snapshot
        if frame_id == self._network_stream_frame_id or encoded_source is None or encoded_enhanced is None:
            return
        source = cv2.imdecode(np.frombuffer(encoded_source, dtype=np.uint8), cv2.IMREAD_COLOR)
        enhanced = cv2.imdecode(np.frombuffer(encoded_enhanced, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if source is None or enhanced is None:
            LOGGER.warning("Could not decode network live frame %s for display", frame_id)
            self._network_stream_frame_id = frame_id
            return
        if (
            self._network_stream_geometry_revision is not None
            and geometry_revision != self._network_stream_geometry_revision
        ):
            self._network_stream_display.clear_roi()
            self._live_measurements.clear()
        self._network_stream_geometry_revision = geometry_revision
        displayed_source = crop_frame(source, crop_rect) if crop_rect is not None else source
        self._network_stream_display.set_frames(displayed_source, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        self._network_stream_frame_id = frame_id
        self._record_live_measurements(displayed_source, enhanced, camera_view_mask)

    def _record_live_measurements(
        self,
        source: np.ndarray,
        enhanced: np.ndarray,
        camera_view_mask: np.ndarray | None = None,
    ) -> None:
        timestamp = perf_counter()
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        if camera_view_mask is not None and (
            camera_view_mask.shape != source_gray.shape or camera_view_mask.shape != enhanced.shape
        ):
            raise ValueError("Live analysis geometry does not match the published frames.")
        roi = self._network_stream_display.roi() if self._network_stream_display is not None else None
        roi_mask = self._network_stream_display.roi_mask() if self._network_stream_display is not None else None
        roi_value = roi_mean(enhanced, roi, roi_mask, camera_view_mask) if roi is not None else math.nan
        reference_value = reference_mean(enhanced, roi, roi_mask, camera_view_mask) if roi is not None else math.nan
        self._live_measurements.append(
            (
                timestamp,
                average_frame_brightness([source_gray], camera_view_mask)[0],
                average_frame_brightness([enhanced], camera_view_mask)[0],
                roi_value,
                reference_value,
            )
        )
        while self._live_measurements and timestamp - self._live_measurements[0][0] > 60.0:
            self._live_measurements.popleft()
        self._refresh_live_analysis()

    def _refresh_live_analysis(self) -> None:
        if not self._live_measurements:
            return
        values = np.asarray(self._live_measurements, dtype=float)
        time = values[:, 0] - values[-1, 0]
        source_brightness = values[:, 1]
        enhanced_brightness = values[:, 2]
        roi_values = values[:, 3]
        reference_values = values[:, 4]
        label = "Live camera"

        if self._has_enabled_stage("frame_brightness_analysis"):
            self.frame_brightness_results = {label: (time, source_brightness, enhanced_brightness)}
            self.refresh_frame_brightness_plot()

        if self._has_enabled_stage("roi_residence_analysis") and np.any(np.isfinite(roi_values)):
            selected = np.isfinite(roi_values) & np.isfinite(reference_values)
            sample_time = time[selected]
            sample_roi = roi_values[selected]
            sample_reference = reference_values[selected]
            if len(sample_roi) >= 2:
                duration = max(0.001, sample_time[-1] - sample_time[0])
                fps = max(1.0, (len(sample_roi) - 1) / duration)
                result = build_analysis_result(
                    label,
                    Path("network-live"),
                    fps,
                    self._network_stream_display.roi() if self._network_stream_display is not None else QRect(),
                    sample_roi,
                    sample_reference,
                    self.threshold_spin.value(),
                    self._has_enabled_stage("gain_stabilization"),
                )
                result.time = sample_time
                self.results = {label: result}
                self._refresh_live_roi_plots(time, roi_values)
                self.pre_card.set_metric(format_seconds(result.residence_time), self._metric_detail(result))
                self.export_action.setEnabled(True)
                self.export_button.setEnabled(True)

    def _refresh_live_roi_plots(self, time: np.ndarray, roi_values: np.ndarray) -> None:
        selected = np.isfinite(roi_values)
        self.normalized_plot.clear()
        self.raw_plot.clear()
        self.derivative_plot.clear()
        self.baseline_to_apex_plot.clear()
        if not np.any(selected):
            return
        raw = roi_values[selected]
        baseline = float(np.median(raw[: max(1, min(len(raw), 30))]))
        contrast = np.clip(baseline - raw, 0, None)
        peak = float(np.max(contrast))
        normalized = contrast / peak if peak > 0 else np.zeros_like(contrast)
        live_time = time[selected]
        self.normalized_plot.plot(live_time, normalized, pen=pg.mkPen(PANEL_COLORS[0].name(), width=2.5), name="Live ROI")
        self.normalized_plot.addItem(
            pg.InfiniteLine(pos=self.threshold_spin.value(), angle=0, pen=pg.mkPen("#e2e8f0", width=1, style=Qt.PenStyle.DashLine))
        )
        self.raw_plot.plot(live_time, raw, pen=pg.mkPen(PANEL_COLORS[0].name(), width=2.5), name="Live ROI")
        self.raw_plot.addItem(
            pg.InfiniteLine(
                pos=baseline,
                angle=0,
                pen=pg.mkPen(PANEL_COLORS[0].name(), width=1.5, style=Qt.PenStyle.DashLine),
            )
        )
        self.derivative_plot.plot(
            live_time,
            temporal_derivative(live_time, normalized),
            pen=pg.mkPen(PANEL_COLORS[0].name(), width=2.5),
            name="Live ROI",
        )
        self.normalized_plot.setXRange(-60, 0, padding=0)
        self.raw_plot.setXRange(-60, 0, padding=0)
        self.derivative_plot.setXRange(-60, 0, padding=0)

    def _placeholder_labels(self, mode: str) -> list[str]:
        if mode == MODE_COMPARISON:
            return ["Pre-deployment", "Post-deployment"]
        if mode == MODE_LIVE:
            return ["Live camera source"]
        return ["Video"]

    def _clear_video_placeholders(self) -> None:
        for placeholder in self.video_placeholders:
            if placeholder is None:
                continue
            self.video_placeholder_layout.removeWidget(placeholder)
            placeholder.hide()
            placeholder.deleteLater()
        for panel in self.pending_preview_panels:
            if panel is None:
                continue
            self.video_placeholder_layout.removeWidget(panel)
            panel.hide()
            panel.deleteLater()
        self.video_placeholders = []
        self.pending_preview_panels = []
        self.pending_slot_labels = []

    def _show_pending_preview_panel(self, index: int, path: Path) -> None:
        if self.pending_mode is None or index < 0 or index >= len(self.pending_video_paths):
            return

        label = (
            self.pending_slot_labels[index]
            if index < len(self.pending_slot_labels)
            else self._placeholder_labels(self.pending_mode)[index]
        )
        color = PANEL_COLORS[min(index, len(PANEL_COLORS) - 1)]
        preview_panel = self.pending_preview_panels[index]
        if preview_panel is None:
            preview_panel = VideoPanel(
                label,
                color,
                path,
                parent=self.video_placeholder_row,
                live_input=self.pending_mode == MODE_LIVE,
            )
            preview_panel.set_comparison(False, 0)
            self.pending_preview_panels[index] = preview_panel
            placeholder = self.video_placeholders[index]
            if placeholder is not None:
                self.video_placeholder_layout.replaceWidget(placeholder, preview_panel)
                placeholder.close()
                placeholder.setParent(None)
                self.video_placeholders[index] = None
            else:
                self.video_placeholder_layout.addWidget(preview_panel)
            self.video_placeholder_layout.setStretch(index, 1)
            return

        preview_panel.set_video(path)
        preview_panel.set_comparison(False, 0)

    def _show_video_placeholders(self, mode: str) -> None:
        self.pause()
        self._clear_video_panels()
        self._clear_video_placeholders()

        labels = self._placeholder_labels(mode)
        self.pending_mode = mode
        self.pending_video_paths = [None] * len(labels)
        self.pending_slot_labels = list(labels)
        self.pending_preview_panels = [None] * len(labels)
        self.enhancement_progress.configure_panels(labels)
        self.active_mode = mode
        self._update_temporal_alignment_controls()

        for index, label in enumerate(labels):
            color = PANEL_COLORS[min(index, len(PANEL_COLORS) - 1)]
            placeholder = VideoDropPlaceholder(label, color, parent=self.video_placeholder_row)
            placeholder.fileDialogRequested.connect(lambda index=index: self._request_placeholder_video(index))
            placeholder.fileDropped.connect(lambda path, index=index: self._set_placeholder_video_path(index, cast(Path, path)))
            self.video_placeholder_layout.addWidget(placeholder)
            self.video_placeholder_layout.setStretch(index, 1)
            self.video_placeholders.append(placeholder)

        self._set_video_controls_enabled(False)
        self._set_live_incompatible_stages_enabled(mode != MODE_LIVE)
        self.video_stack.setCurrentWidget(self.video_placeholder_row)
        self._update_video_stack_geometry()
        self._sync_trimmed_video_window()
        self._set_playback_limit(self.source_max_frame)
        self.current_frame_index = 0
        self.update_time_label()
        self.statusBar().showMessage("Select a video file by dragging it into a placeholder or clicking to browse.")

    def _update_video_stack_geometry(self) -> None:
        self.video_stack.request_height_refresh()
        self.video_stack.updateGeometry()
        parent = self.video_stack.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().invalidate()
            parent.updateGeometry()
        QTimer.singleShot(0, self.video_stack.request_height_refresh)
        QTimer.singleShot(0, self.video_stack.updateGeometry)

    def _request_placeholder_video(self, index: int) -> None:
        if self.pending_mode is None or index < 0 or index >= len(self.pending_video_paths):
            return
        if index < len(self.pending_slot_labels):
            label = self.pending_slot_labels[index].lower()
        else:
            label = self._placeholder_labels(self.pending_mode)[index].lower()
        title = "Select video to loop as a live camera" if self.pending_mode == MODE_LIVE else f"Open {label} video"
        path = self._open_video_file(title)
        if path is None:
            return
        self._set_placeholder_video_path(index, path)

    def _set_placeholder_video_path(self, index: int, path: Path) -> None:
        if self.pending_mode is None or index < 0 or index >= len(self.pending_video_paths):
            return
        self.pending_video_paths[index] = path
        self._show_pending_preview_panel(index, path)

        remaining = sum(1 for selected in self.pending_video_paths if selected is None)
        if remaining > 0:
            noun = "video" if remaining == 1 else "videos"
            self.statusBar().showMessage(f"Select {remaining} more {noun} to continue.")
            return

        mode = self.pending_mode
        videos = [selected for selected in self.pending_video_paths if selected is not None]
        self.pending_mode = None
        self.pending_video_paths = []
        self._set_video_panels(cast(list[Path], videos), live_input=mode == MODE_LIVE)
        if mode == MODE_LIVE:
            self.statusBar().showMessage("Live camera simulation ready. The selected video loops continuously.")

    def _open_slot_video(self, slot_index: int) -> None:
        if slot_index == 0 and self.pre_panel is not None:
            self.open_video(self.pre_panel)
            return
        if slot_index == 1 and self.post_panel is not None:
            self.open_video(self.post_panel)
            return
        if self.pending_mode is not None:
            self._request_placeholder_video(slot_index)

    def _build_actions(self) -> None:
        self.open_pre_action = QAction("Open video 1...", self)
        self.open_pre_action.triggered.connect(lambda: self._open_slot_video(0))
        self.open_post_action = QAction("Open video 2...", self)
        self.open_post_action.triggered.connect(lambda: self._open_slot_video(1))
        self.open_single_mode_action = QAction("Switch to single video mode...", self)
        self.open_single_mode_action.triggered.connect(lambda: self._select_mode_and_videos(MODE_SINGLE))
        self.open_comparison_mode_action = QAction("Switch to comparison mode...", self)
        self.open_comparison_mode_action.triggered.connect(lambda: self._select_mode_and_videos(MODE_COMPARISON))
        self.open_live_mode_action = QAction("Switch to live camera mode...", self)
        self.open_live_mode_action.triggered.connect(lambda: self._select_mode_and_videos(MODE_LIVE))
        self.save_config_action = QAction("Save configuration...", self)
        self.save_config_action.triggered.connect(self.save_config)
        self.load_config_action = QAction("Load configuration...", self)
        self.load_config_action.triggered.connect(self.load_config)
        self.save_default_pipeline_settings_action = QAction("Save as default pipeline settings", self)
        self.save_default_pipeline_settings_action.triggered.connect(self.save_default_pipeline_settings)
        self.export_action = QAction("Export analysis CSV...", self)
        self.export_action.triggered.connect(self.export_csv)
        self.export_action.setEnabled(False)
        self.export_pdf_action = QAction("Export comparison report PDF...", self)
        self.export_pdf_action.triggered.connect(self.export_pdf_report)
        self.export_pdf_action.setEnabled(False)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_pre_action)
        file_menu.addAction(self.open_post_action)
        file_menu.addSeparator()
        file_menu.addAction(self.open_single_mode_action)
        file_menu.addAction(self.open_comparison_mode_action)
        file_menu.addAction(self.open_live_mode_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_config_action)
        file_menu.addAction(self.load_config_action)
        file_menu.addAction(self.save_default_pipeline_settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.export_pdf_action)

    def _sync_trimmed_video_window(self) -> None:
        if not self.panels:
            self.source_max_frame = 0
            self.max_frame = 0
            self.fps = 30.0
            self.play_interval_ms = self._play_interval_ms()
            return
        common_frame_count = min(panel.playback_frame_count for panel in self.panels)
        for panel in self.panels:
            if panel.trim_frame_count != common_frame_count:
                panel.set_trim_window(panel.trim_start_frame, common_frame_count)
        self.source_max_frame = max(0, common_frame_count - 1)
        self.max_frame = self.source_max_frame
        self.fps = min(panel.info.fps for panel in self.panels)
        self.play_interval_ms = self._play_interval_ms()

    def _apply_source_pipeline_stages(self) -> bool:
        if not self.panels:
            return False
        auto_crop_enabled = self._has_enabled_stage("auto_crop")
        startup_stabilization_enabled = self._has_enabled_stage("startup_stabilization_trim")
        temporal_alignment_enabled = self._has_enabled_stage("temporal_alignment")
        contrast_gain_alignment_enabled = self._has_enabled_stage("contrast_gain_alignment")
        histogram_matching_enabled = self.active_mode == MODE_COMPARISON and self._has_enabled_stage("histogram_matching")
        metal_needle_removal_enabled = self._has_enabled_stage("metal_needle_removal")
        auto_crop_size_offset = self.auto_crop_size_offset_spin.value()
        background_settings = self._background_subtraction_settings()
        states = [
            panel.calculate_source_pipeline(
                auto_crop_enabled,
                temporal_alignment_enabled,
                auto_crop_size_offset=auto_crop_size_offset,
                temporal_trim_offset_seconds=self.temporal_trim_offset_spin.value(),
                temporal_end_trim_seconds=self.temporal_end_trim_spin.value(),
                comparison_sync_offset_seconds=(
                    self.comparison_sync_offset_spin.value()
                    if self.active_mode == MODE_COMPARISON and panel_index == 1
                    else 0.0
                ),
                contrast_gain_alignment_enabled=contrast_gain_alignment_enabled,
                histogram_matching_enabled=histogram_matching_enabled,
                background_subtraction_enabled=background_settings[panel_index][0],
                background_level=background_settings[panel_index][1],
                metal_needle_removal_enabled=metal_needle_removal_enabled,
                startup_stabilization_enabled=startup_stabilization_enabled,
                temporal_gain_normalization_enabled=self.temporal_gain_normalization_check.isChecked(),
            )
            for panel_index, panel in enumerate(self.panels)
        ]
        if contrast_gain_alignment_enabled:
            states = align_source_gain_states(states)
        if histogram_matching_enabled and self.active_mode == MODE_COMPARISON:
            states = align_source_histogram_states(states)
        if temporal_alignment_enabled and self.active_mode == MODE_COMPARISON:
            states = align_source_brightness_onsets(states)
        return self._apply_source_pipeline_states(states)

    def _apply_source_pipeline_states(self, states: list[SourcePipelineState]) -> bool:
        changed = False
        for panel, state in zip(self.panels, states):
            changed = panel.apply_source_pipeline_state(state) or changed
        if not changed:
            return False

        self._sync_trimmed_video_window()
        self._set_playback_limit(self.source_max_frame)
        target_frame = max(0, min(self.current_frame_index, self.max_frame))
        self.current_frame_index = -1
        self.set_frame_index(target_frame)
        return True

    def _source_pipeline_is_current(
        self,
        auto_crop: bool,
        startup_stabilization_trim: bool,
        temporal_gain_normalization: bool,
        temporal_alignment: bool,
        contrast_gain_alignment: bool,
        histogram_matching: bool,
        auto_crop_size_offset: int,
        temporal_trim_offset_seconds: float,
        temporal_end_trim_seconds: float,
        comparison_sync_offset_seconds: float,
        background_settings: tuple[tuple[bool, int], ...],
        metal_needle_removal: bool,
    ) -> bool:
        if not self.panels:
            return True
        return all(
            panel.source_pipeline_configuration
            == (
                auto_crop,
                temporal_alignment,
                contrast_gain_alignment,
                background_settings[panel_index][0],
                background_settings[panel_index][1],
                auto_crop_size_offset,
                temporal_trim_offset_seconds,
                temporal_end_trim_seconds,
                comparison_sync_offset_seconds if self.active_mode == MODE_COMPARISON and panel_index == 1 else 0.0,
                metal_needle_removal,
                histogram_matching,
            ) + ((True,) if startup_stabilization_trim else ()) + ((False,) if not temporal_gain_normalization else ())
            for panel_index, panel in enumerate(self.panels)
        )

    def _background_subtraction_settings(self) -> tuple[tuple[bool, int], ...]:
        controls = (
            (self.background_video_one_check, self.background_video_one_level_spin),
            (self.background_video_two_check, self.background_video_two_level_spin),
        )
        stage_enabled = self.background_subtraction_stage_check.isChecked()
        return tuple((stage_enabled and check.isChecked(), level.value()) for check, level in controls[: len(self.panels)])

    def _build_ui(self) -> None:
        self.play_button = QPushButton()
        self.play_button.setText("")
        self.play_button.setToolTip("Play/Pause")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, self.max_frame)
        self.frame_slider.valueChanged.connect(self.set_frame_index)
        # Keep this flexible so the main splitter can allocate more width to the pipeline drawer.
        self.frame_slider.setMinimumWidth(180)

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

        self.loop_check = QCheckBox("Loop")
        self.loop_check.setToolTip("Restart playback from the first frame when the video ends")

        self.time_label = QLabel()
        self.time_label.setObjectName("timeLabel")

        self.live_device_name_edit = QLineEdit("device")
        self.live_device_name_edit.setObjectName("liveRecordingField")
        self.live_device_name_edit.setPlaceholderText("Device")
        self.live_device_name_edit.setMaximumWidth(120)
        self.live_device_name_edit.setToolTip("Device name for exported live recordings")
        self.live_test_identifier_edit = QLineEdit("test")
        self.live_test_identifier_edit.setObjectName("liveRecordingField")
        self.live_test_identifier_edit.setPlaceholderText("Test")
        self.live_test_identifier_edit.setMaximumWidth(120)
        self.live_test_identifier_edit.setToolTip("Test identifier for exported live recordings")
        self.live_phase_toggle = QCheckBox("Post")
        self.live_phase_toggle.setObjectName("liveRecordingPhaseToggle")
        self.live_phase_toggle.setToolTip("Unchecked: pre; checked: post")
        self.live_recording_name_label = QLabel()
        self.live_recording_name_label.setObjectName("timeLabel")
        self.live_recording_name_label.setToolTip("Name of the next live recording")
        self.live_export_toggle = QCheckBox("Export")
        self.live_export_toggle.setObjectName("liveRecordingPhaseToggle")
        self.live_export_toggle.setChecked(True)
        self.live_export_toggle.setToolTip("Save newly received live video clips")
        self.live_device_name_edit.textChanged.connect(self._update_live_recording_name)
        self.live_test_identifier_edit.textChanged.connect(self._update_live_recording_name)
        self.live_phase_toggle.toggled.connect(self._update_live_recording_name)
        self.live_export_toggle.toggled.connect(self._set_live_recording_enabled)
        self.live_recording_controls = QWidget()
        self.live_recording_controls.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        live_recording_layout = QHBoxLayout(self.live_recording_controls)
        live_recording_layout.setContentsMargins(0, 0, 0, 0)
        live_recording_layout.setSpacing(12)
        device_field = QWidget()
        device_field.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        device_field_layout = QHBoxLayout(device_field)
        device_field_layout.setContentsMargins(0, 0, 0, 0)
        device_field_layout.setSpacing(4)
        device_field_layout.addWidget(QLabel("Device"))
        device_field_layout.addWidget(self.live_device_name_edit)
        test_field = QWidget()
        test_field.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        test_field_layout = QHBoxLayout(test_field)
        test_field_layout.setContentsMargins(0, 0, 0, 0)
        test_field_layout.setSpacing(4)
        test_field_layout.addWidget(QLabel("Test"))
        test_field_layout.addWidget(self.live_test_identifier_edit)
        live_recording_layout.addWidget(device_field)
        live_recording_layout.addWidget(test_field)
        live_recording_layout.addWidget(self.live_phase_toggle)
        live_recording_layout.addWidget(self.live_recording_name_label)
        live_recording_layout.addWidget(self.live_export_toggle)

        self.compare_view_check = QCheckBox("Source")
        self.compare_view_check.setChecked(False)
        self.compare_view_check.setToolTip("Show source beside enhanced output")
        self.compare_view_check.toggled.connect(self.on_compare_view_toggled)

        self.temporal_change_view_check = QCheckBox("Heatmap")
        self.temporal_change_view_check.setChecked(False)
        self.temporal_change_view_check.setEnabled(False)
        self.temporal_change_view_check.setToolTip("Show the contrast residence heatmap beside enhanced output")
        self.temporal_change_view_check.toggled.connect(self.on_temporal_change_view_toggled)

        self.overlay_mask_check = QCheckBox("Masks")
        self.overlay_mask_check.setChecked(True)
        self.overlay_mask_check.setEnabled(False)
        self.overlay_mask_check.setToolTip("Show ROI and needle segmentation masks over enhanced video")
        self.overlay_mask_check.toggled.connect(self.on_roi_region_overlay_toggled)

        self.video_row = QWidget()
        self.video_row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.video_layout = QHBoxLayout(self.video_row)
        self.video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_layout.setSpacing(14)
        self.video_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.video_placeholder_row = QWidget()
        self.video_placeholder_row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.video_placeholder_layout = QHBoxLayout(self.video_placeholder_row)
        self.video_placeholder_layout.setContentsMargins(0, 0, 0, 0)
        self.video_placeholder_layout.setSpacing(14)
        self.video_placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.mode_selection_page = QWidget()
        self.mode_selection_page.setObjectName("modeSelectionPage")
        mode_layout = QVBoxLayout(self.mode_selection_page)
        mode_layout.setContentsMargins(24, 24, 24, 24)

        mode_panel = QFrame()
        mode_panel.setObjectName("modeSelectionPanel")
        mode_panel.setFixedSize(720, 560)
        mode_panel_layout = QVBoxLayout(mode_panel)
        mode_panel_layout.setContentsMargins(40, 30, 40, 30)
        mode_panel_layout.setSpacing(14)

        mode_setup_label = QLabel("Setup study")
        mode_setup_label.setObjectName("modeSelectionSectionTitle")
        mode_setup_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mode_buttons_row = QWidget()
        mode_buttons_row.setObjectName("modeSelectionButtons")
        mode_buttons_layout = QHBoxLayout(mode_buttons_row)
        mode_buttons_layout.setContentsMargins(0, 0, 0, 0)
        mode_buttons_layout.setSpacing(20)

        self.single_mode_button = ModeSelectionButton("Single video", MODE_SINGLE)
        self.single_mode_button.clicked.connect(lambda: self._select_mode_and_videos(MODE_SINGLE))

        self.comparison_mode_button = ModeSelectionButton("Comparison", MODE_COMPARISON)
        self.comparison_mode_button.clicked.connect(lambda: self._select_mode_and_videos(MODE_COMPARISON))
        self.live_mode_button = ModeSelectionButton("Live camera", MODE_SINGLE)
        self.live_mode_button.clicked.connect(lambda: self._select_mode_and_videos(MODE_LIVE))

        mode_buttons_layout.addStretch()
        mode_buttons_layout.addWidget(self.single_mode_button)
        mode_buttons_layout.addWidget(self.comparison_mode_button)
        mode_buttons_layout.addWidget(self.live_mode_button)
        mode_buttons_layout.addStretch()

        mode_separator = QFrame()
        mode_separator.setObjectName("modeSelectionSeparator")
        mode_separator.setFrameShape(QFrame.Shape.HLine)
        mode_separator.setFrameShadow(QFrame.Shadow.Plain)

        mode_load_label = QLabel("Load study")
        mode_load_label.setObjectName("modeSelectionSectionTitle")
        mode_load_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mode_config_row = QWidget()
        mode_config_row.setObjectName("modeSelectionConfigActions")
        mode_config_layout = QHBoxLayout(mode_config_row)
        mode_config_layout.setContentsMargins(0, 0, 0, 0)
        mode_config_layout.setSpacing(20)

        self.load_recent_mode_button = ModeSelectionButton("Recent", MODE_RECENT)
        self.load_recent_mode_button.clicked.connect(self.load_most_recent_config)

        self.select_config_mode_button = ModeSelectionButton("Select", MODE_SELECT)
        self.select_config_mode_button.clicked.connect(self.load_config)

        mode_config_layout.addStretch()
        mode_config_layout.addWidget(self.load_recent_mode_button)
        mode_config_layout.addWidget(self.select_config_mode_button)
        mode_config_layout.addStretch()

        mode_panel_layout.addWidget(mode_setup_label)
        mode_panel_layout.addWidget(mode_buttons_row)
        mode_panel_layout.addStretch(1)
        mode_panel_layout.addWidget(mode_separator)
        mode_panel_layout.addWidget(mode_load_label)
        mode_panel_layout.addWidget(mode_config_row)
        mode_panel_layout.addStretch(1)
        self.mode_selection_panel = mode_panel
        self.mode_selection_view = UniformScaleView(mode_panel)
        mode_layout.addWidget(self.mode_selection_view)

        self.video_stack = CurrentPageStack()
        self.video_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.video_stack.addWidget(self.mode_selection_page)
        self.video_stack.addWidget(self.video_placeholder_row)
        self.video_stack.addWidget(self.video_row)
        self.video_stack.setCurrentWidget(self.mode_selection_page)
        self._update_mode_selection_config_actions()

        playback_row = QWidget()
        self.playback_row = playback_row
        playback_layout = QHBoxLayout(playback_row)
        playback_layout.setContentsMargins(0, 0, 0, 0)
        playback_layout.setSpacing(8)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.frame_slider, 1)
        self.frame_label = QLabel("Frame")
        playback_layout.addWidget(self.frame_label)
        playback_layout.addWidget(self.frame_spin)
        self.speed_control_label = QLabel("Speed")
        playback_layout.addWidget(self.speed_control_label)
        playback_layout.addWidget(self.speed_slider)
        playback_layout.addWidget(self.speed_label)
        playback_layout.addWidget(self.loop_check)
        playback_layout.addWidget(self.time_label)
        playback_layout.addWidget(self.live_recording_controls)
        self.live_recording_spacer = QWidget()
        self.live_recording_spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        playback_layout.addWidget(self.live_recording_spacer, 1)
        playback_layout.addWidget(self.compare_view_check)
        playback_layout.addWidget(self.temporal_change_view_check)
        playback_layout.addWidget(self.overlay_mask_check)
        self.playback_controls = (
            self.play_button,
            self.frame_slider,
            self.frame_label,
            self.frame_spin,
            self.speed_control_label,
            self.speed_slider,
            self.speed_label,
            self.loop_check,
            self.time_label,
        )

        controls_panel = self._build_controls_panel()
        self.enhancement_progress = EnhancementProgressPanel()
        self.pipeline_drawer = CollapsibleDrawer("Pipeline")
        self.pipeline_left_column = self.pipeline_drawer
        self.pipeline_controls_scroll = controls_panel
        self.pipeline_drawer.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        self.pipeline_drawer.setMinimumWidth(self.pipeline_drawer.collapsed_width())
        self.pipeline_drawer.setMaximumWidth(1200)
        self.pipeline_drawer.setMinimumHeight(0)
        self.pipeline_drawer.content_layout.addWidget(controls_panel, 1)
        self.pipeline_drawer.content_layout.addWidget(self.enhancement_progress)
        self.pipeline_drawer.toggle_button.toggled.connect(self._set_pipeline_drawer_expanded)
        plot_panel = self._build_plot_panel()

        video_panel = QWidget()
        video_panel.setMinimumHeight(0)
        video_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 12)
        video_layout.setSpacing(14)
        video_layout.addWidget(self.video_stack, 1)
        video_layout.addWidget(playback_row, 0, Qt.AlignmentFlag.AlignTop)

        self.graph_drawer = GraphDrawer("Analysis")
        self.graph_drawer.setMinimumHeight(self.graph_drawer.header_height())
        self.graph_drawer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.graph_drawer.content_layout.addWidget(plot_panel)
        self.graph_drawer.toggle_button.toggled.connect(self._set_graph_drawer_expanded)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.graph_splitter = right_splitter
        right_splitter.addWidget(video_panel)
        right_splitter.addWidget(self.graph_drawer)
        right_splitter.setCollapsible(1, True)
        right_splitter.setCollapsible(0, False)
        right_splitter.setHandleWidth(8)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 0)
        right_splitter.splitterMoved.connect(self._remember_graph_drawer_height)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(12, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(right_splitter)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        self._pipeline_drawer_default_width = 420
        splitter.addWidget(self.pipeline_drawer)
        splitter.addWidget(right_column)
        splitter.setHandleWidth(8)
        splitter.setCollapsible(0, False)
        splitter.setSizes([self._pipeline_drawer_default_width, 1080])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        QTimer.singleShot(0, self._collapse_graph_panel_by_default)
        QTimer.singleShot(0, self._expand_pipeline_panel_by_default)
        QTimer.singleShot(0, self._update_pipeline_column_width)

    def _build_controls_panel(self) -> QWidget:
        controls = QWidget()
        controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(10)
        self._pipeline_controls_base_right_margin = controls_layout.contentsMargins().right()

        pipeline_label = QLabel("Source pipeline")
        pipeline_label.setObjectName("pipelineLabel")
        controls_layout.addWidget(pipeline_label)
        self.enhancement_layout = controls_layout
        self.source_pipeline_layout = QVBoxLayout()
        self.source_pipeline_layout.setContentsMargins(0, 0, 0, 0)
        self.source_pipeline_layout.setSpacing(controls_layout.spacing())
        controls_layout.addLayout(self.source_pipeline_layout)
        self.live_pipeline_label = QLabel("Live pipeline (applied in this order, top to bottom)")
        self.live_pipeline_label.setObjectName("pipelineLabel")
        controls_layout.addWidget(self.live_pipeline_label)
        self.live_pipeline_layout = QVBoxLayout()
        self.live_pipeline_layout.setContentsMargins(0, 0, 0, 0)
        self.live_pipeline_layout.setSpacing(controls_layout.spacing())
        controls_layout.addLayout(self.live_pipeline_layout)
        self.auto_crop_stage_drawer = StageDrawer(
            "auto_crop",
            "Auto-crop fluoroscope field",
            1,
        )
        self.startup_stabilization_trim_stage_drawer = StageDrawer(
            "startup_stabilization_trim",
            "Trim fluoroscope startup settling",
            2,
        )
        self.temporal_alignment_stage_drawer = StageDrawer(
            "temporal_alignment",
            "Temporal alignment (trim onset)",
            3,
        )
        self.contrast_gain_alignment_stage_drawer = StageDrawer(
            "contrast_gain_alignment",
            "Align fluoroscope contrast gain",
            3,
        )
        self.histogram_matching_stage_drawer = StageDrawer(
            "histogram_matching",
            "Match comparison histogram",
            4,
        )
        self.background_subtraction_stage_drawer = StageDrawer(
            "background_subtraction",
            "DSA mask subtraction",
            5,
        )
        self.metal_needle_removal_stage_drawer = StageDrawer(
            "metal_needle_removal",
            "Remove stationary metal needle",
            6,
        )
        self.brightness_stage_drawer = StageDrawer(
            "brightness_stabilization",
            "Gain / brightness stabilization",
            3,
        )
        self.camera_motion_stage_drawer = StageDrawer(
            "camera_motion_stabilization",
            "Stabilize camera motion",
            4,
        )
        self.roi_stage_drawer = StageDrawer("roi_extraction", "Aneurysm ROI extraction", 4)
        self.roi_needle_level_alignment_stage_drawer = StageDrawer(
            "roi_needle_level_alignment",
            "Align ROI / needle baseline levels",
            5,
        )
        self.gain_stage_drawer = StageDrawer("gain_stabilization", "Median gain normalization", 6)
        self.scanline_stage_drawer = StageDrawer("scanline_correction", "Scanline correction", 6)
        self.denoise_stage_drawer = StageDrawer("denoise", "Spatial denoising", 7)
        self.temporal_stage_drawer = StageDrawer("temporal_filter", "Motion-aware temporal filtering", 8)
        self.mottle_stage_drawer = StageDrawer("quantum_mottle_filter", "Quantum mottle reduction", 9)
        self.contrast_stage_drawer = StageDrawer("local_contrast", "Local contrast (CLAHE)", 10)
        self.adjustments_stage_drawer = StageDrawer("image_adjustments", "Image adjustments", 11)
        self.smoothing_stage_drawer = StageDrawer("final_smoothing", "Final Gaussian smoothing", 12)
        self.analysis_stage_drawer = StageDrawer(
            "roi_residence_analysis",
            "ROI residence analysis",
            14,
        )
        self.frame_brightness_analysis_stage_drawer = StageDrawer(
            "frame_brightness_analysis",
            "Frame brightness analysis",
            15,
        )
        self.needle_segmentation_stage_drawer = StageDrawer(
            "needle_segmentation",
            "Needle segmentation and brightness",
            16,
        )
        self.parent_vessel_roi_stage_drawer = StageDrawer(
            "parent_vessel_roi_analysis",
            "Parent vessel ROI darkness",
            17,
        )
        self.background_roi_stage_drawer = StageDrawer(
            "background_roi_analysis",
            "Background ROI brightness",
            18,
        )
        self.temporal_change_heatmap_stage_drawer = StageDrawer(
            "temporal_change_heatmap",
            "Contrast residence heatmap",
            19,
        )
        self.auto_crop_stage_check = self.auto_crop_stage_drawer.enable_button
        self.startup_stabilization_trim_stage_check = self.startup_stabilization_trim_stage_drawer.enable_button
        self.temporal_alignment_stage_check = self.temporal_alignment_stage_drawer.enable_button
        self.contrast_gain_alignment_stage_check = self.contrast_gain_alignment_stage_drawer.enable_button
        self.histogram_matching_stage_check = self.histogram_matching_stage_drawer.enable_button
        self.background_subtraction_stage_check = self.background_subtraction_stage_drawer.enable_button
        self.metal_needle_removal_stage_check = self.metal_needle_removal_stage_drawer.enable_button
        self.gain_stage_check = self.gain_stage_drawer.enable_button
        self.brightness_stage_check = self.brightness_stage_drawer.enable_button
        self.camera_motion_stage_check = self.camera_motion_stage_drawer.enable_button
        self.roi_stage_check = self.roi_stage_drawer.enable_button
        self.roi_needle_level_alignment_stage_check = self.roi_needle_level_alignment_stage_drawer.enable_button
        self.scanline_stage_check = self.scanline_stage_drawer.enable_button
        self.denoise_stage_check = self.denoise_stage_drawer.enable_button
        self.temporal_stage_check = self.temporal_stage_drawer.enable_button
        self.mottle_stage_check = self.mottle_stage_drawer.enable_button
        self.contrast_stage_check = self.contrast_stage_drawer.enable_button
        self.adjustments_stage_check = self.adjustments_stage_drawer.enable_button
        self.smoothing_stage_check = self.smoothing_stage_drawer.enable_button
        self.analysis_stage_check = self.analysis_stage_drawer.enable_button
        self.frame_brightness_analysis_stage_check = self.frame_brightness_analysis_stage_drawer.enable_button
        self.needle_segmentation_stage_check = self.needle_segmentation_stage_drawer.enable_button
        self.parent_vessel_roi_stage_check = self.parent_vessel_roi_stage_drawer.enable_button
        self.background_roi_stage_check = self.background_roi_stage_drawer.enable_button
        self.temporal_change_heatmap_stage_check = self.temporal_change_heatmap_stage_drawer.enable_button
        self.source_pipeline_stage_checks = [
            self.auto_crop_stage_check,
            self.startup_stabilization_trim_stage_check,
            self.temporal_alignment_stage_check,
            self.contrast_gain_alignment_stage_check,
            self.histogram_matching_stage_check,
            self.background_subtraction_stage_check,
            self.metal_needle_removal_stage_check,
        ]
        self.frame_pipeline_stage_checks = [
            self.brightness_stage_check,
            self.camera_motion_stage_check,
            self.roi_stage_check,
            self.roi_needle_level_alignment_stage_check,
            self.gain_stage_check,
            self.scanline_stage_check,
            self.denoise_stage_check,
            self.temporal_stage_check,
            self.mottle_stage_check,
            self.contrast_stage_check,
            self.adjustments_stage_check,
            self.smoothing_stage_check,
        ]
        self.pipeline_stage_checks = [
            *self.source_pipeline_stage_checks,
            *self.frame_pipeline_stage_checks,
            self.analysis_stage_check,
            self.frame_brightness_analysis_stage_check,
            self.needle_segmentation_stage_check,
            self.parent_vessel_roi_stage_check,
            self.background_roi_stage_check,
            self.temporal_change_heatmap_stage_check,
        ]
        self.source_pipeline_stage_drawers = [
            self.auto_crop_stage_drawer,
            self.startup_stabilization_trim_stage_drawer,
            self.temporal_alignment_stage_drawer,
            self.contrast_gain_alignment_stage_drawer,
            self.histogram_matching_stage_drawer,
            self.background_subtraction_stage_drawer,
            self.metal_needle_removal_stage_drawer,
        ]
        self.frame_pipeline_stage_drawers = [
            self.brightness_stage_drawer,
            self.camera_motion_stage_drawer,
            self.roi_stage_drawer,
            self.roi_needle_level_alignment_stage_drawer,
            self.gain_stage_drawer,
            self.scanline_stage_drawer,
            self.denoise_stage_drawer,
            self.temporal_stage_drawer,
            self.mottle_stage_drawer,
            self.contrast_stage_drawer,
            self.adjustments_stage_drawer,
            self.smoothing_stage_drawer,
        ]
        self.live_pipeline_stage_drawers = [
            *self.frame_pipeline_stage_drawers,
            self.analysis_stage_drawer,
            self.frame_brightness_analysis_stage_drawer,
            self.needle_segmentation_stage_drawer,
            self.parent_vessel_roi_stage_drawer,
            self.background_roi_stage_drawer,
            self.temporal_change_heatmap_stage_drawer,
        ]
        self.pipeline_stage_drawers = [
            *self.source_pipeline_stage_drawers,
            *self.live_pipeline_stage_drawers,
        ]
        self._dragged_stage_drawer: StageDrawer | None = None
        self._stage_drag_placeholder: QWidget | None = None
        self._dragged_pipeline_order: list[StageDrawer] | None = None
        self._dragged_pipeline_layout: QVBoxLayout | None = None
        self._stage_drag_offset_y = 0
        self._stage_drag_x = 0
        for check in self.pipeline_stage_checks:
            check.setChecked(
                check in {
                    self.auto_crop_stage_check,
                    self.startup_stabilization_trim_stage_check,
                    self.temporal_alignment_stage_check,
                    self.brightness_stage_check,
                }
            )
            check.toggled.connect(self.on_pipeline_stages_changed)
        self._build_stage_drawer_controls()
        self._name_stage_parameter_widgets()
        for drawer in self.pipeline_stage_drawers:
            drawer.reorderRequested.connect(self._reorder_pipeline_stage_by_key)
            drawer.dragStarted.connect(self._begin_pipeline_stage_drag)
            drawer.dragMoved.connect(self._move_pipeline_stage_drag)
            drawer.dragFinished.connect(self._finish_pipeline_stage_drag)
            drawer.optionsRequested.connect(self._show_stage_options)

        self.stage_drawer_templates = {drawer.stage_key: drawer for drawer in self.pipeline_stage_drawers}
        self.source_pipeline_stage_drawers = []
        self.live_pipeline_stage_drawers = []
        self.pipeline_stage_drawers = []
        self.source_add_stage_button = self._create_add_stage_button("Add source stage")
        self.source_add_stage_button.clicked.connect(lambda: self._show_add_stage_menu("source"))
        self.live_add_stage_button = self._create_add_stage_button("Add live stage")
        self.live_add_stage_button.clicked.connect(lambda: self._show_add_stage_menu("live"))
        self.add_stage_button = self.live_add_stage_button

        self._refresh_pipeline_stage_ui()
        controls_layout.addStretch()
        self.enhancement_mode_combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)
        self.denoise_strength_spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.inference_batch_spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.inference_precision_combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Clearance threshold"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.20)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setObjectName("analysisThreshold")
        self.threshold_spin.valueChanged.connect(self.on_analysis_threshold_changed)
        threshold_row.addWidget(self.threshold_spin)
        self._add_parameter_slider(threshold_row, self.threshold_spin)
        self.analysis_stage_drawer.content_layout.addLayout(threshold_row)

        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        self.analysis_stage_drawer.content_layout.addWidget(self.export_button)
        self.export_pdf_button = QPushButton("Export comparison PDF")
        self.export_pdf_button.clicked.connect(self.export_pdf_report)
        self.export_pdf_button.setEnabled(False)
        self.analysis_stage_drawer.content_layout.addWidget(self.export_pdf_button)

        self.pre_card = MetricCard("Pre residence")
        self.post_card = MetricCard("Post residence")
        self.delta_card = MetricCard("Difference")
        self.analysis_stage_drawer.content_layout.addWidget(self.pre_card)
        self.analysis_stage_drawer.content_layout.addWidget(self.post_card)
        self.analysis_stage_drawer.content_layout.addWidget(self.delta_card)

        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("controlsScroll")
        controls_scroll.setWidget(controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.verticalScrollBar().rangeChanged.connect(self._update_pipeline_column_width)
        return controls_scroll

    def _name_stage_parameter_widgets(self) -> None:
        widget_names = {
            "auto_crop_size_offset_spin": "autoCropSizeOffset",
            "temporal_trim_offset_spin": "temporalTrimOffset",
            "temporal_end_trim_spin": "temporalEndTrim",
            "comparison_sync_offset_spin": "comparisonSyncOffset",
            "temporal_gain_normalization_check": "temporalGainNormalization",
            "denoise_strength_label": "denoiseStrengthLabel",
            "enhancement_mode_combo": "denoiseMode",
            "denoise_strength_spin": "denoiseStrength",
            "inference_batch_spin": "denoiseBatchSize",
            "inference_precision_combo": "denoisePrecision",
            "gain_auto_target_check": "gainUseAutoTarget",
            "gain_target_spin": "gainTargetMedian",
            "gain_min_spin": "gainMinimum",
            "gain_max_spin": "gainMaximum",
            "roi_soften_check": "roiSoftenEnabled",
            "roi_radius_spin": "roiSofteningRadius",
            "roi_threshold_spin": "roiSofteningThreshold",
            "roi_convex_hull_check": "roiConvexHullEnabled",
            "roi_circle_fit_check": "roiCircleFitEnabled",
            "scanline_clip_spin": "scanlineBiasClip",
            "scanline_sigma_spin": "scanlineSigmaY",
            "temporal_sigma_spin": "temporalMotionSigma",
            "mottle_similarity_sigma_spin": "mottleSimilaritySigma",
            "mottle_window_combo": "mottleWindowRadius",
            "clahe_clip_spin": "claheClipLimit",
            "clahe_tile_spin": "claheTileSize",
            "adjustments_brightness_spin": "adjustmentsBrightness",
            "adjustments_contrast_spin": "adjustmentsContrast",
            "adjustments_sharpen_spin": "adjustmentsSharpen",
            "adjustments_gamma_spin": "adjustmentsGamma",
            "smoothing_sigma_spin": "smoothingSigma",
            "heatmap_colormap_combo": "heatmapColormap",
        }
        for attribute, object_name in widget_names.items():
            getattr(self, attribute).setObjectName(object_name)

    def _clone_stage_widget(self, widget: QWidget) -> QWidget | None:
        if isinstance(widget, QSlider):
            return None
        if isinstance(widget, QCheckBox):
            clone = QCheckBox(widget.text())
            clone.setChecked(widget.isChecked())
        elif isinstance(widget, QComboBox):
            clone = QComboBox()
            for index in range(widget.count()):
                clone.addItem(widget.itemText(index), widget.itemData(index))
            clone.setCurrentIndex(widget.currentIndex())
        elif isinstance(widget, QDoubleSpinBox):
            clone = QDoubleSpinBox()
            clone.setDecimals(widget.decimals())
            clone.setRange(widget.minimum(), widget.maximum())
            clone.setSingleStep(widget.singleStep())
            clone.setSuffix(widget.suffix())
            clone.setValue(widget.value())
            clone.setButtonSymbols(widget.buttonSymbols())
        elif isinstance(widget, QSpinBox):
            clone = QSpinBox()
            clone.setRange(widget.minimum(), widget.maximum())
            clone.setSingleStep(widget.singleStep())
            clone.setSuffix(widget.suffix())
            clone.setValue(widget.value())
            clone.setButtonSymbols(widget.buttonSymbols())
        elif isinstance(widget, QLabel):
            clone = QLabel(widget.text())
            clone.setWordWrap(widget.wordWrap())
        elif isinstance(widget, QPushButton):
            clone = QPushButton(widget.text())
        else:
            return None
        clone.setObjectName(widget.objectName())
        clone.setToolTip(widget.toolTip())
        clone.setEnabled(widget.isEnabled())
        return clone

    def _clone_stage_layout(self, source: QVBoxLayout | QHBoxLayout, target: QVBoxLayout | QHBoxLayout) -> None:
        target.setContentsMargins(source.contentsMargins())
        target.setSpacing(source.spacing())
        for index in range(source.count()):
            item = source.itemAt(index)
            child_layout = item.layout()
            if isinstance(child_layout, QHBoxLayout):
                clone_layout = QHBoxLayout()
                self._clone_stage_layout(child_layout, clone_layout)
                target.addLayout(clone_layout)
            elif isinstance(child_layout, QVBoxLayout):
                clone_layout = QVBoxLayout()
                self._clone_stage_layout(child_layout, clone_layout)
                target.addLayout(clone_layout)
            elif item.widget() is not None:
                clone = self._clone_stage_widget(item.widget())
                if clone is not None:
                    target.addWidget(clone)

    def _configure_dynamic_stage_drawer(self, drawer: StageDrawer) -> None:
        for check in drawer.findChildren(QCheckBox):
            check.toggled.connect(self.on_enhancement_settings_changed)
        for combo in drawer.findChildren(QComboBox):
            if combo.objectName() == "heatmapColormap":
                combo.currentIndexChanged.connect(self.on_heatmap_colormap_changed)
            else:
                combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)
        for spin in [*drawer.findChildren(QSpinBox), *drawer.findChildren(QDoubleSpinBox)]:
            spin.valueChanged.connect(self.on_enhancement_settings_changed)
        for button in drawer.findChildren(QPushButton):
            if button.text() == "Refresh ROI extraction":
                button.clicked.connect(self.redetect_aneurysms)
            elif button.text() == "Export CSV":
                button.clicked.connect(self.export_csv)
        gain_auto = drawer.findChild(QCheckBox, "gainUseAutoTarget")
        gain_target = drawer.findChild(QSpinBox, "gainTargetMedian")
        if gain_auto is not None and gain_target is not None:
            gain_auto.toggled.connect(lambda checked: gain_target.setEnabled(not checked))
            gain_target.setEnabled(not gain_auto.isChecked())
        roi_soften = drawer.findChild(QCheckBox, "roiSoftenEnabled")
        roi_radius = drawer.findChild(QDoubleSpinBox, "roiSofteningRadius")
        roi_threshold = drawer.findChild(QDoubleSpinBox, "roiSofteningThreshold")
        if roi_soften is not None and roi_radius is not None and roi_threshold is not None:
            roi_soften.toggled.connect(roi_radius.setEnabled)
            roi_soften.toggled.connect(roi_threshold.setEnabled)
            roi_radius.setEnabled(roi_soften.isChecked())
            roi_threshold.setEnabled(roi_soften.isChecked())
        self._add_sliders_to_parameter_layout(drawer.content_layout)
        drawer.enable_button.toggled.connect(self.on_pipeline_stages_changed)
        drawer.dragStarted.connect(self._begin_pipeline_stage_drag)
        drawer.dragMoved.connect(self._move_pipeline_stage_drag)
        drawer.dragFinished.connect(self._finish_pipeline_stage_drag)
        drawer.optionsRequested.connect(self._show_stage_options)

    def _copy_stage_drawer(self, source: StageDrawer) -> StageDrawer:
        drawer = StageDrawer(source.stage_key, source.stage_title, 0)
        self._clone_stage_layout(source.content_layout, drawer.content_layout)
        self._configure_dynamic_stage_drawer(drawer)
        return drawer

    def _create_add_stage_button(self, tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setObjectName("addStageButton")
        button.setText("+")
        button.setFixedSize(32, 32)
        button.setToolTip(tooltip)
        return button

    def _show_add_stage_menu(self, pipeline: str) -> None:
        menu = QMenu(self)
        source_stage = pipeline == "source"
        stage_keys = tuple(
            key
            for key in self.stage_drawer_templates
            if (BUILTIN_STAGES.require(key).execution_shape == ExecutionShape.SOURCE) == source_stage
        )
        if self.active_mode == MODE_LIVE:
            stage_keys = tuple(
                key for key in stage_keys if BUILTIN_STAGES.require(key).supports_live(EnhancementParameters())
            )
        for key in stage_keys:
            action = menu.addAction(BUILTIN_STAGES.require(key).display_name)
            action.triggered.connect(
                lambda _checked=False, stage_key=key, target=pipeline: self._add_pipeline_stage(stage_key, pipeline=target)
            )
        self._stage_menu = menu
        button = self.source_add_stage_button if pipeline == "source" else self.live_add_stage_button
        menu.popup(button.mapToGlobal(button.rect().bottomLeft()))

    def _show_stage_options(self, drawer: StageDrawer, position: QPoint) -> None:
        menu = QMenu(self)
        duplicate_action = menu.addAction("Duplicate")
        duplicate_action.triggered.connect(lambda: self._duplicate_pipeline_stage(drawer))
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(lambda: self._delete_pipeline_stage(drawer))
        self._stage_menu = menu
        menu.popup(position)

    def _add_pipeline_stage(
        self,
        stage_key: str,
        after: StageDrawer | None = None,
        pipeline: str | None = None,
    ) -> StageDrawer:
        template = self.stage_drawer_templates[stage_key]
        definition = BUILTIN_STAGES.require(stage_key)
        stage_pipeline = pipeline or (
            "source" if definition.execution_shape == ExecutionShape.SOURCE else "live"
        )
        drawers = self.source_pipeline_stage_drawers if stage_pipeline == "source" else self.live_pipeline_stage_drawers
        drawer = template if template not in drawers else self._copy_stage_drawer(template)
        if drawer not in drawers:
            drawer.setParent(self.enhancement_layout.parentWidget())
            insert_index = len(drawers)
            if after is not None:
                insert_index = drawers.index(after) + 1
            drawers.insert(insert_index, drawer)
        self._sync_pipeline_stage_lists()
        self._refresh_pipeline_stage_ui()
        if drawer.enable_button.isChecked() and not self._loading_config:
            self.on_pipeline_stages_changed()
        return drawer

    def _duplicate_pipeline_stage(self, drawer: StageDrawer) -> None:
        duplicate = self._copy_stage_drawer(drawer)
        duplicate.enable_button.setChecked(False)
        drawers = self._pipeline_drawers_for(drawer)
        drawers.insert(drawers.index(drawer) + 1, duplicate)
        self._sync_pipeline_stage_lists()
        self._refresh_pipeline_stage_ui()

    def _delete_pipeline_stage(self, drawer: StageDrawer) -> None:
        was_enabled = drawer.enable_button.isChecked()
        self._pipeline_drawers_for(drawer).remove(drawer)
        self._pipeline_layout_for(drawer).removeWidget(drawer)
        drawer.hide()
        drawer.setParent(None)
        self._sync_pipeline_stage_lists()
        self._refresh_pipeline_stage_ui()
        if was_enabled:
            self.on_pipeline_stages_changed()

    def _drawer_parameters(self, drawer: StageDrawer) -> EnhancementParameters:
        def value(name: str, default: int | float | bool | str) -> int | float | bool | str:
            widget = drawer.findChild(QWidget, name)
            if isinstance(widget, QComboBox):
                return widget.currentData()
            if isinstance(widget, QCheckBox):
                return widget.isChecked()
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                return widget.value()
            return default

        gain_min = float(value("gainMinimum", 0.70))
        gain_max = float(value("gainMaximum", 1.45))
        return EnhancementParameters(
            gain_use_auto_target=bool(value("gainUseAutoTarget", True)),
            gain_target_median=int(value("gainTargetMedian", 128)),
            gain_min=min(gain_min, gain_max),
            gain_max=max(gain_min, gain_max),
            roi_softening_enabled=bool(value("roiSoftenEnabled", False)),
            roi_softening_radius_ratio=float(value("roiSofteningRadius", 0.12)),
            roi_softening_threshold=float(value("roiSofteningThreshold", 0.10)),
            roi_convex_hull_enabled=bool(value("roiConvexHullEnabled", True)),
            roi_circle_fit_enabled=bool(value("roiCircleFitEnabled", False)),
            scanline_bias_clip=float(value("scanlineBiasClip", 6.0)),
            scanline_sigma_y=float(value("scanlineSigmaY", 2.0)),
            temporal_motion_sigma=float(value("temporalMotionSigma", 12.0)),
            mottle_similarity_sigma=float(value("mottleSimilaritySigma", 12.0)),
            mottle_window_radius=int(value("mottleWindowRadius", 2)),
            clahe_clip_limit=float(value("claheClipLimit", 1.0)),
            clahe_tile_size=int(value("claheTileSize", 6)),
            adjustments_brightness_offset=int(value("adjustmentsBrightness", 0)),
            adjustments_contrast_gain=float(value("adjustmentsContrast", 1.0)),
            adjustments_sharpen_amount=float(value("adjustmentsSharpen", 0.0)),
            adjustments_gamma=float(value("adjustmentsGamma", 1.0)),
            smoothing_sigma_x=float(value("smoothingSigma", 0.55)),
        )

    def _sync_active_denoise_controls(self) -> None:
        for drawer in self._stage_drawers("denoise"):
            mode = drawer.findChild(QComboBox, "denoiseMode")
            strength_label = drawer.findChild(QLabel, "denoiseStrengthLabel")
            strength = drawer.findChild(QSpinBox, "denoiseStrength")
            batch_size = drawer.findChild(QSpinBox, "denoiseBatchSize")
            precision = drawer.findChild(QComboBox, "denoisePrecision")
            active_mode = str(mode.currentData()) if mode is not None else "ffdnet-ngc"
            stage_enabled = drawer.enable_button.isChecked()
            uses_ffdnet = stage_enabled and active_mode.startswith("ffdnet")
            uses_nlm = active_mode in {"non-local-means", "tensor-nlm-ngc"}
            uses_accelerator = uses_ffdnet or active_mode == "tensor-nlm-ngc"
            if strength_label is not None:
                strength_label.setText("NLM filter strength" if uses_nlm else "FFDNet noise sigma")
                strength_label.setEnabled(stage_enabled)
            if strength is not None:
                strength.setToolTip(
                    "OpenCV non-local means luminance filter strength"
                    if uses_nlm
                    else "FFDNet's assumed input noise standard deviation"
                )
                strength.setEnabled(stage_enabled)
                self._sync_parameter_slider_enabled(strength)
            if batch_size is not None:
                batch_size.setEnabled(stage_enabled and uses_accelerator)
                self._sync_parameter_slider_enabled(batch_size)
            if precision is not None:
                precision.setEnabled(stage_enabled and uses_accelerator)

    def _pipeline_scrollbar_reserve_width(self) -> int:
        scrollbar = self.pipeline_controls_scroll.verticalScrollBar()
        style_width = self.style().pixelMetric(
            QStyle.PixelMetric.PM_ScrollBarExtent,
            None,
            self.pipeline_controls_scroll,
        )
        return max(style_width, scrollbar.sizeHint().width(), 0)

    def _sync_pipeline_scroll_gutter(self) -> None:
        margins = self.enhancement_layout.contentsMargins()
        base_right = getattr(self, "_pipeline_controls_base_right_margin", margins.right())
        scrollbar = self.pipeline_controls_scroll.verticalScrollBar()
        reserve_width = self._pipeline_scrollbar_reserve_width()
        needs_scrollbar = scrollbar.maximum() > scrollbar.minimum()
        target_right = base_right if needs_scrollbar else base_right + reserve_width
        if margins.right() != target_right:
            self.enhancement_layout.setContentsMargins(
                margins.left(),
                margins.top(),
                target_right,
                margins.bottom(),
            )

    def _update_pipeline_column_width(self) -> None:
        if not hasattr(self, "pipeline_controls_scroll"):
            return

        self._sync_pipeline_scroll_gutter()
        if hasattr(self, "pipeline_drawer"):
            self.pipeline_drawer.setMinimumWidth(self.pipeline_drawer.collapsed_width())
        if hasattr(self, "pipeline_controls_scroll"):
            self.pipeline_controls_scroll.setMinimumWidth(0)
        if hasattr(self, "pipeline_left_column"):
            self.pipeline_left_column.setMinimumWidth(self.pipeline_drawer.collapsed_width() if hasattr(self, "pipeline_drawer") else 0)

    def _build_plot_panel(self) -> QWidget:
        plot_group = QFrame()
        plot_group.setObjectName("plotPanel")
        plot_layout = QGridLayout(plot_group)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)

        self.analysis_tabs = QTabWidget()
        residence_panel = QWidget()
        residence_layout = QGridLayout(residence_panel)
        residence_layout.setContentsMargins(0, 0, 0, 0)
        residence_layout.setSpacing(10)
        roi_brightness_panel = QWidget()
        roi_brightness_layout = QGridLayout(roi_brightness_panel)
        roi_brightness_layout.setContentsMargins(0, 0, 0, 0)
        roi_brightness_layout.setSpacing(10)
        derivative_panel = QWidget()
        derivative_layout = QGridLayout(derivative_panel)
        derivative_layout.setContentsMargins(0, 0, 0, 0)
        derivative_layout.setSpacing(10)
        self.normalized_plot = pg.PlotWidget(title="Normalized Contrast Residence")
        self.parent_vessel_scaled_residence_plot = pg.PlotWidget(
            title="Aneurysm Contrast Residence Scaled by Parent Vessel Minimum"
        )
        self.raw_plot = pg.PlotWidget(title="Denoised ROI Brightness")
        self.roi_needle_alignment_plot = pg.PlotWidget(title="ROI Brightness Before and After Baseline Alignment")
        self.derivative_plot = pg.PlotWidget(title="Normalized Contrast Derivative")
        baseline_to_apex_panel = QWidget()
        baseline_to_apex_layout = QGridLayout(baseline_to_apex_panel)
        baseline_to_apex_layout.setContentsMargins(0, 0, 0, 0)
        baseline_to_apex_layout.setSpacing(10)
        self.baseline_to_apex_plot = pg.PlotWidget(title="Baseline-to-Apex Contrast")
        self.frame_brightness_panel = QWidget()
        self.frame_brightness_layout = QVBoxLayout(self.frame_brightness_panel)
        self.frame_brightness_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_brightness_layout.setSpacing(10)
        self.frame_brightness_plots: dict[str, pg.PlotWidget] = {}
        self.needle_brightness_plot = pg.PlotWidget(title="Needle Average Pixel Brightness")
        self.parent_vessel_roi_plot = pg.PlotWidget(title="Parent Vessel Darkest 10% Median")
        self.background_roi_plot = pg.PlotWidget(title="Background ROI Average Brightness")
        for plot in [
            self.normalized_plot,
            self.parent_vessel_scaled_residence_plot,
            self.raw_plot,
            self.roi_needle_alignment_plot,
            self.derivative_plot,
            self.baseline_to_apex_plot,
            self.needle_brightness_plot,
            self.parent_vessel_roi_plot,
            self.background_roi_plot,
        ]:
            plot.setBackground("#111827")
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.getAxis("bottom").setPen("#8aa0b8")
            plot.getAxis("left").setPen("#8aa0b8")
            plot.getAxis("bottom").setTextPen("#cbd5e1")
            plot.getAxis("left").setTextPen("#cbd5e1")
            plot.addLegend(offset=(12, 12))
        self.normalized_plot.setLabel("bottom", "Time", units="s")
        self.normalized_plot.setLabel("left", "Normalized signal")
        self.parent_vessel_scaled_residence_plot.setLabel("bottom", "Time", units="s")
        self.parent_vessel_scaled_residence_plot.setLabel("left", "Parent-vessel scaled signal")
        self.raw_plot.setLabel("bottom", "Time", units="s")
        self.raw_plot.setLabel("left", "Mean pixel value")
        self.roi_needle_alignment_plot.setLabel("bottom", "Time", units="s")
        self.roi_needle_alignment_plot.setLabel("left", "Mean pixel value")
        self.derivative_plot.setLabel("bottom", "Time", units="s")
        self.derivative_plot.setLabel("left", "Signal change", units="1/s")
        self.baseline_to_apex_plot.setLabel("bottom", "Time", units="s")
        self.baseline_to_apex_plot.setLabel("left", "Baseline to apex", units="normalized")
        self.needle_brightness_plot.setLabel("bottom", "Time", units="s")
        self.needle_brightness_plot.setLabel("left", "Mean pixel value")
        self.parent_vessel_roi_plot.setLabel("bottom", "Time", units="s")
        self.parent_vessel_roi_plot.setLabel("left", "Darkest 10% median")
        self.background_roi_plot.setLabel("bottom", "Time", units="s")
        self.background_roi_plot.setLabel("left", "Mean pixel value")

        residence_layout.addWidget(self.normalized_plot, 0, 0)
        parent_vessel_scaled_residence_panel = QWidget()
        parent_vessel_scaled_residence_layout = QGridLayout(parent_vessel_scaled_residence_panel)
        parent_vessel_scaled_residence_layout.setContentsMargins(0, 0, 0, 0)
        parent_vessel_scaled_residence_layout.addWidget(self.parent_vessel_scaled_residence_plot, 0, 0)
        roi_brightness_layout.addWidget(self.raw_plot, 0, 0)
        derivative_layout.addWidget(self.derivative_plot, 0, 0)
        baseline_to_apex_layout.addWidget(self.baseline_to_apex_plot, 0, 0)
        self.analysis_tabs.addTab(residence_panel, "ROI residence")
        self.parent_vessel_scaled_residence_tab_index = self.analysis_tabs.addTab(
            parent_vessel_scaled_residence_panel,
            "Parent vessel scaled residence",
        )
        self.analysis_tabs.setTabVisible(self.parent_vessel_scaled_residence_tab_index, False)
        self.analysis_tabs.addTab(roi_brightness_panel, "ROI brightness")
        self.analysis_tabs.addTab(derivative_panel, "Derivative")
        self.baseline_to_apex_tab_index = self.analysis_tabs.addTab(baseline_to_apex_panel, "Baseline to apex")
        self.analysis_tabs.setTabVisible(self.baseline_to_apex_tab_index, False)
        self.analysis_tabs.addTab(self.frame_brightness_panel, "Frame brightness")
        alignment_panel = QWidget()
        alignment_layout = QGridLayout(alignment_panel)
        alignment_layout.setContentsMargins(0, 0, 0, 0)
        alignment_layout.addWidget(self.roi_needle_alignment_plot, 0, 0)
        self.roi_needle_alignment_tab_index = self.analysis_tabs.addTab(alignment_panel, "ROI alignment")
        self.analysis_tabs.setTabVisible(self.roi_needle_alignment_tab_index, False)
        background_roi_panel = QWidget()
        background_roi_layout = QGridLayout(background_roi_panel)
        background_roi_layout.setContentsMargins(0, 0, 0, 0)
        background_roi_layout.addWidget(self.background_roi_plot, 0, 0)
        self.background_roi_tab_index = self.analysis_tabs.addTab(background_roi_panel, "Background ROI")
        self.analysis_tabs.setTabVisible(self.background_roi_tab_index, False)
        parent_vessel_panel = QWidget()
        parent_vessel_layout = QGridLayout(parent_vessel_panel)
        parent_vessel_layout.setContentsMargins(0, 0, 0, 0)
        parent_vessel_layout.addWidget(self.parent_vessel_roi_plot, 0, 0)
        self.parent_vessel_roi_tab_index = self.analysis_tabs.addTab(parent_vessel_panel, "Parent vessel")
        self.analysis_tabs.setTabVisible(self.parent_vessel_roi_tab_index, False)
        needle_brightness_panel = QWidget()
        needle_brightness_layout = QGridLayout(needle_brightness_panel)
        needle_brightness_layout.setContentsMargins(0, 0, 0, 0)
        needle_brightness_layout.addWidget(self.needle_brightness_plot, 0, 0)
        self.analysis_tabs.addTab(needle_brightness_panel, "Needle brightness")
        plot_layout.addWidget(self.analysis_tabs, 0, 0)
        return plot_group

    def _collapse_graph_panel_by_default(self) -> None:
        splitter = getattr(self, "graph_splitter", None)
        drawer = getattr(self, "graph_drawer", None)
        if splitter is None or drawer is None:
            return
        drawer.set_expanded(False)
        total_height = max(1, sum(splitter.sizes()) or splitter.height())
        header_height = max(1, drawer.header_height())
        splitter.setSizes([max(1, total_height - header_height), header_height])

    def _expand_pipeline_panel_by_default(self) -> None:
        drawer = getattr(self, "pipeline_drawer", None)
        if drawer is None:
            return
        drawer.set_expanded(True)
        self._set_pipeline_drawer_expanded(True)

    def _set_pipeline_drawer_expanded(self, expanded: bool) -> None:
        splitter = getattr(self, "main_splitter", None)
        drawer = getattr(self, "pipeline_drawer", None)
        if splitter is None or drawer is None:
            return

        sizes = splitter.sizes()
        if len(sizes) < 2:
            return
        total_width = max(1, sum(sizes) or splitter.width())
        collapsed_width = max(1, drawer.collapsed_width())
        last_width = getattr(self, "_pipeline_drawer_last_width", 0)
        default_expanded_width = getattr(self, "_pipeline_drawer_default_width", 600)
        expanded_width = max(
            default_expanded_width,
            last_width if last_width > collapsed_width else default_expanded_width,
        )
        if expanded:
            drawer.set_expanded(True)
            target_width = max(collapsed_width + 1, min(expanded_width, max(1, total_width - 1)))
            self._pipeline_drawer_last_width = target_width
            splitter.setSizes([target_width, max(1, total_width - target_width)])
        else:
            if sizes[0] > collapsed_width:
                self._pipeline_drawer_last_width = sizes[0]
            drawer.set_expanded(False)
            splitter.setSizes([collapsed_width, max(1, total_width - collapsed_width)])

    def _set_graph_drawer_expanded(self, expanded: bool) -> None:
        splitter = getattr(self, "graph_splitter", None)
        drawer = getattr(self, "graph_drawer", None)
        if splitter is None or drawer is None:
            return

        sizes = splitter.sizes()
        header_height = max(1, drawer.header_height())
        if expanded:
            drawer.set_expanded(True)
            total_height = max(1, sum(sizes) or splitter.height())
            target_height = max(getattr(self, "_graph_drawer_last_height", 260), header_height * 6)
            target_height = min(target_height, max(header_height, total_height - 1))
            splitter.setSizes([max(1, total_height - target_height), target_height])
        else:
            drawer.set_expanded(False)
            total_height = max(1, sum(sizes) or splitter.height())
            splitter.setSizes([max(1, total_height - header_height), header_height])

    def _remember_graph_drawer_height(self, _position: int, _index: int) -> None:
        if not self.graph_drawer.toggle_button.isChecked():
            return
        sizes = self.graph_splitter.sizes()
        header_height = max(1, self.graph_drawer.header_height())
        if len(sizes) > 1 and sizes[1] > header_height:
            self._graph_drawer_last_height = sizes[1]

    def _build_stage_drawer_controls(self) -> None:
        self._parameter_slider_pairs: list[tuple[QSpinBox | QDoubleSpinBox, QSlider]] = []
        enhancement_mode_row = QHBoxLayout()
        enhancement_mode_row.addWidget(QLabel("Enhancement model"))
        self.enhancement_mode_combo = QComboBox()
        self.enhancement_mode_combo.addItem("NGC FFDNet (Docker)", "ffdnet-ngc")
        self.enhancement_mode_combo.addItem("Native FFDNet (GPU)", "ffdnet-native")
        self.enhancement_mode_combo.addItem("NGC Tensor NLM (GPU)", "tensor-nlm-ngc")
        self.enhancement_mode_combo.addItem("Non-local means (CPU)", "non-local-means")
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

        inference_batch_row = QHBoxLayout()
        inference_batch_row.addWidget(QLabel("Batch frames"))
        self.inference_batch_spin = QSpinBox()
        self.inference_batch_spin.setRange(1, 16)
        self.inference_batch_spin.setValue(4)
        self.inference_batch_spin.setToolTip("More frames improve GPU throughput but use more GPU and shared memory")
        inference_batch_row.addWidget(self.inference_batch_spin)
        self.denoise_stage_drawer.content_layout.addLayout(inference_batch_row)

        inference_precision_row = QHBoxLayout()
        inference_precision_row.addWidget(QLabel("Precision"))
        self.inference_precision_combo = QComboBox()
        self.inference_precision_combo.addItem("FP16", "fp16")
        self.inference_precision_combo.addItem("FP32", "fp32")
        self.inference_precision_combo.setToolTip("FP16 is faster; FP32 is useful for numerical comparisons")
        inference_precision_row.addWidget(self.inference_precision_combo)
        self.denoise_stage_drawer.content_layout.addLayout(inference_precision_row)

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

        gain_min_row = QHBoxLayout()
        gain_min_row.addWidget(QLabel("Minimum gain clamp"))
        self.gain_min_spin = QDoubleSpinBox()
        self.gain_min_spin.setRange(0.10, 2.00)
        self.gain_min_spin.setSingleStep(0.05)
        self.gain_min_spin.setDecimals(2)
        self.gain_min_spin.setValue(0.70)
        gain_min_row.addWidget(self.gain_min_spin)
        self.gain_stage_drawer.content_layout.addLayout(gain_min_row)

        gain_max_row = QHBoxLayout()
        gain_max_row.addWidget(QLabel("Maximum gain clamp"))
        self.gain_max_spin = QDoubleSpinBox()
        self.gain_max_spin.setRange(0.10, 2.00)
        self.gain_max_spin.setSingleStep(0.05)
        self.gain_max_spin.setDecimals(2)
        self.gain_max_spin.setValue(1.45)
        gain_max_row.addWidget(self.gain_max_spin)
        self.gain_stage_drawer.content_layout.addLayout(gain_max_row)

        auto_crop_hint = QLabel(
            "Detects the active fluoroscope field and applies a centered aligned crop before downstream processing."
        )
        auto_crop_hint.setObjectName("subtleLabel")
        auto_crop_hint.setWordWrap(True)
        self.auto_crop_stage_drawer.content_layout.addWidget(auto_crop_hint)

        auto_crop_size_row = QHBoxLayout()
        auto_crop_size_row.addWidget(QLabel("Square size offset"))
        self.auto_crop_size_offset_spin = QSpinBox()
        self.auto_crop_size_offset_spin.setRange(-512, 512)
        self.auto_crop_size_offset_spin.setSingleStep(32)
        self.auto_crop_size_offset_spin.setValue(0)
        self.auto_crop_size_offset_spin.setSuffix(" px")
        self.auto_crop_size_offset_spin.setToolTip("Increase or decrease the detected square crop size in 32-pixel steps")
        auto_crop_size_row.addWidget(self.auto_crop_size_offset_spin)
        self._add_parameter_slider(auto_crop_size_row, self.auto_crop_size_offset_spin)
        self.auto_crop_stage_drawer.content_layout.addLayout(auto_crop_size_row)

        temporal_alignment_hint = QLabel(
            "Finds contrast onset and trims each video to start slightly before injection for timeline alignment."
        )
        temporal_alignment_hint.setObjectName("subtleLabel")
        temporal_alignment_hint.setWordWrap(True)
        self.temporal_alignment_stage_drawer.content_layout.addWidget(temporal_alignment_hint)

        self.temporal_gain_normalization_check = QCheckBox("Normalize gain before onset detection")
        self.temporal_gain_normalization_check.setChecked(True)
        self.temporal_gain_normalization_check.setToolTip(
            "Compensate frame-wide fluoroscope gain changes before detecting the whole-frame brightness drop"
        )
        self.temporal_gain_normalization_check.toggled.connect(self.on_enhancement_settings_changed)
        self.temporal_alignment_stage_drawer.content_layout.addWidget(self.temporal_gain_normalization_check)

        gain_alignment_hint = QLabel(
            "Matches the lower-gain control's pre-injection brightness and contrast excursion to the stronger comparison source."
        )
        gain_alignment_hint.setObjectName("subtleLabel")
        gain_alignment_hint.setWordWrap(True)
        self.contrast_gain_alignment_stage_drawer.content_layout.addWidget(gain_alignment_hint)

        histogram_matching_hint = QLabel(
            "Matches the post-deployment source histogram to the pre-deployment source before enhancement."
        )
        histogram_matching_hint.setObjectName("subtleLabel")
        histogram_matching_hint.setWordWrap(True)
        self.histogram_matching_stage_drawer.content_layout.addWidget(histogram_matching_hint)

        temporal_trim_row = QHBoxLayout()
        temporal_trim_row.addWidget(QLabel("Start trim offset"))
        self.temporal_trim_offset_spin = QDoubleSpinBox()
        self.temporal_trim_offset_spin.setRange(-2.0, 2.0)
        self.temporal_trim_offset_spin.setSingleStep(0.05)
        self.temporal_trim_offset_spin.setDecimals(2)
        self.temporal_trim_offset_spin.setValue(0.0)
        self.temporal_trim_offset_spin.setSuffix(" s")
        self.temporal_trim_offset_spin.setToolTip("Shift each video start earlier or later relative to its detected contrast injection")
        temporal_trim_row.addWidget(self.temporal_trim_offset_spin)
        self._add_parameter_slider(temporal_trim_row, self.temporal_trim_offset_spin)
        self.temporal_alignment_stage_drawer.content_layout.addLayout(temporal_trim_row)

        temporal_end_trim_row = QHBoxLayout()
        temporal_end_trim_row.addWidget(QLabel("End trim"))
        self.temporal_end_trim_spin = QDoubleSpinBox()
        self.temporal_end_trim_spin.setRange(0.0, 30.0)
        self.temporal_end_trim_spin.setSingleStep(0.05)
        self.temporal_end_trim_spin.setDecimals(2)
        self.temporal_end_trim_spin.setValue(0.0)
        self.temporal_end_trim_spin.setSuffix(" s")
        self.temporal_end_trim_spin.setToolTip("Remove this duration from the end of each video after temporal alignment")
        temporal_end_trim_row.addWidget(self.temporal_end_trim_spin)
        self._add_parameter_slider(temporal_end_trim_row, self.temporal_end_trim_spin)
        self.temporal_alignment_stage_drawer.content_layout.addLayout(temporal_end_trim_row)

        self.comparison_sync_offset_row = QWidget()
        comparison_sync_layout = QHBoxLayout(self.comparison_sync_offset_row)
        comparison_sync_layout.setContentsMargins(0, 0, 0, 0)
        comparison_sync_layout.addWidget(QLabel("Post-video sync offset"))
        self.comparison_sync_offset_spin = QDoubleSpinBox()
        self.comparison_sync_offset_spin.setRange(-2.0, 2.0)
        self.comparison_sync_offset_spin.setSingleStep(0.01)
        self.comparison_sync_offset_spin.setDecimals(2)
        self.comparison_sync_offset_spin.setValue(0.0)
        self.comparison_sync_offset_spin.setSuffix(" s")
        self.comparison_sync_offset_spin.setToolTip("Shift only the post-deployment video to fine-tune contrast injection synchronization")
        comparison_sync_layout.addWidget(self.comparison_sync_offset_spin)
        self._add_parameter_slider(comparison_sync_layout, self.comparison_sync_offset_spin)
        self.temporal_alignment_stage_drawer.content_layout.addWidget(self.comparison_sync_offset_row)

        background_hint = QLabel(
            "Acquires a median mask immediately before injection and subtracts subsequent frames from it."
        )
        background_hint.setObjectName("subtleLabel")
        background_hint.setWordWrap(True)
        self.background_subtraction_stage_drawer.content_layout.addWidget(background_hint)

        metal_needle_hint = QLabel(
            "Finds persistent dark metal in the opening frames and replaces it before downstream processing."
        )
        metal_needle_hint.setObjectName("subtleLabel")
        metal_needle_hint.setWordWrap(True)
        self.metal_needle_removal_stage_drawer.content_layout.addWidget(metal_needle_hint)
        self.background_video_rows: list[QWidget] = []
        background_controls = (
            ("Video 1", "backgroundVideo1Enabled", "backgroundVideo1Level"),
            ("Video 2", "backgroundVideo2Enabled", "backgroundVideo2Level"),
        )
        for label, enabled_name, level_name in background_controls:
            row_widget = QWidget()
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)
            enabled = QCheckBox(label)
            enabled.setObjectName(enabled_name)
            level = QSpinBox()
            level.setObjectName(level_name)
            level.setRange(0, 255)
            level.setValue(0)
            level.setPrefix("Threshold ")
            level.setSuffix(" levels")
            level.setToolTip("Minimum darkening below the reference retained as contrast")
            row.addWidget(enabled)
            row.addWidget(level)
            self._add_parameter_slider(row, level)
            self.background_subtraction_stage_drawer.content_layout.addWidget(row_widget)
            self.background_video_rows.append(row_widget)
            if enabled_name == "backgroundVideo1Enabled":
                self.background_video_one_check = enabled
                self.background_video_one_level_spin = level
            else:
                self.background_video_two_check = enabled
                self.background_video_two_level_spin = level

        brightness_hint = QLabel(
            "Corrects frame-wide gain and brightness jitter from robust, temporally stable image probes."
        )
        brightness_hint.setObjectName("subtleLabel")
        brightness_hint.setWordWrap(True)
        self.brightness_stage_drawer.content_layout.addWidget(brightness_hint)

        roi_hint = QLabel(
            "Extracts an aneurysm ROI mask from the current upstream video state. "
            "Enable softening to expand and round the detected mask before ROI analysis uses it."
        )
        roi_hint.setObjectName("subtleLabel")
        roi_hint.setWordWrap(True)
        self.roi_stage_drawer.content_layout.addWidget(roi_hint)

        self.roi_video_mode_rows: list[QWidget] = []
        self.roi_mode_combos: list[QComboBox] = []
        for index in range(2):
            mode_row = QWidget()
            mode_layout = QHBoxLayout(mode_row)
            mode_layout.setContentsMargins(0, 0, 0, 0)
            mode_label = QLabel("ROI mode" if index == 0 else "Post-deployment ROI mode")
            mode_combo = QComboBox()
            mode_combo.addItem("Automatic", "auto")
            mode_combo.addItem("Manual", "manual")
            mode_combo.setObjectName(f"roiVideo{index + 1}Mode")
            mode_combo.setToolTip("Drag from the circle center on the enhanced video to set a manual ROI radius")
            mode_combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)
            mode_layout.addWidget(mode_label)
            mode_layout.addWidget(mode_combo)
            self.roi_stage_drawer.content_layout.addWidget(mode_row)
            self.roi_video_mode_rows.append(mode_row)
            self.roi_mode_combos.append(mode_combo)

        roi_soften_row = QHBoxLayout()
        self.roi_soften_check = QCheckBox("Soften and expand mask")
        self.roi_soften_check.setChecked(False)
        roi_soften_row.addWidget(self.roi_soften_check)
        roi_soften_row.addStretch()
        self.roi_stage_drawer.content_layout.addLayout(roi_soften_row)

        roi_radius_row = QHBoxLayout()
        roi_radius_row.addWidget(QLabel("Softening radius"))
        self.roi_radius_spin = QDoubleSpinBox()
        self.roi_radius_spin.setRange(0.02, 0.40)
        self.roi_radius_spin.setSingleStep(0.01)
        self.roi_radius_spin.setDecimals(2)
        self.roi_radius_spin.setValue(0.12)
        self.roi_radius_spin.setSuffix(" x ROI")
        self.roi_radius_spin.setEnabled(False)
        roi_radius_row.addWidget(self.roi_radius_spin)
        self.roi_stage_drawer.content_layout.addLayout(roi_radius_row)

        roi_threshold_row = QHBoxLayout()
        roi_threshold_row.addWidget(QLabel("Soft mask threshold"))
        self.roi_threshold_spin = QDoubleSpinBox()
        self.roi_threshold_spin.setRange(0.01, 0.95)
        self.roi_threshold_spin.setSingleStep(0.01)
        self.roi_threshold_spin.setDecimals(2)
        self.roi_threshold_spin.setValue(0.10)
        self.roi_threshold_spin.setEnabled(False)
        roi_threshold_row.addWidget(self.roi_threshold_spin)
        self.roi_stage_drawer.content_layout.addLayout(roi_threshold_row)

        self.roi_convex_hull_check = QCheckBox("Fill aneurysm convex hull")
        self.roi_convex_hull_check.setChecked(True)
        self.roi_convex_hull_check.setToolTip("Smooth the automatic aneurysm region boundary by filling its convex hull")
        self.roi_stage_drawer.content_layout.addWidget(self.roi_convex_hull_check)

        self.roi_circle_fit_check = QCheckBox("Fit circle to aneurysm hull")
        self.roi_circle_fit_check.setChecked(False)
        self.roi_circle_fit_check.setToolTip("Refine the automatic aneurysm region to a circle fitted to its convex hull")
        self.roi_stage_drawer.content_layout.addWidget(self.roi_circle_fit_check)

        self.redetect_button = QPushButton("Refresh ROI extraction")
        self.redetect_button.clicked.connect(self.redetect_aneurysms)
        self.roi_stage_drawer.content_layout.addWidget(self.redetect_button)

        parent_vessel_hint = QLabel("Right-click the enhanced video to place the center of a fixed 50 x 50 pixel parent vessel box.")
        parent_vessel_hint.setObjectName("subtleLabel")
        parent_vessel_hint.setWordWrap(True)
        self.parent_vessel_roi_stage_drawer.content_layout.addWidget(parent_vessel_hint)
        self.parent_vessel_rotation_rows: list[QWidget] = []
        self.parent_vessel_rotation_spins: list[QDoubleSpinBox] = []
        for index in range(2):
            rotation_row = QWidget()
            rotation_layout = QHBoxLayout(rotation_row)
            rotation_layout.setContentsMargins(0, 0, 0, 0)
            rotation_layout.addWidget(QLabel("Rotation" if index == 0 else "Post-deployment rotation"))
            rotation = QDoubleSpinBox()
            rotation.setObjectName(f"parentVesselRotation{index + 1}")
            rotation.setRange(-180.0, 180.0)
            rotation.setSingleStep(1.0)
            rotation.setDecimals(1)
            rotation.setSuffix(" deg")
            rotation.setToolTip("Rotate this panel's parent vessel ROI box around its center")
            rotation.valueChanged.connect(lambda value, panel_index=index: self._on_parent_vessel_rotation_changed(panel_index, value))
            rotation_layout.addWidget(rotation)
            self._add_parameter_slider(rotation_layout, rotation)
            self.parent_vessel_roi_stage_drawer.content_layout.addWidget(rotation_row)
            self.parent_vessel_rotation_rows.append(rotation_row)
            self.parent_vessel_rotation_spins.append(rotation)

        background_roi_hint = QLabel("Shift-left-click the enhanced video to place the center of a fixed 150 x 150 pixel background box.")
        background_roi_hint.setObjectName("subtleLabel")
        background_roi_hint.setWordWrap(True)
        self.background_roi_stage_drawer.content_layout.addWidget(background_roi_hint)

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

        temporal_row = QHBoxLayout()
        temporal_row.addWidget(QLabel("Motion sensitivity sigma"))
        self.temporal_sigma_spin = QDoubleSpinBox()
        self.temporal_sigma_spin.setRange(1.0, 60.0)
        self.temporal_sigma_spin.setSingleStep(0.5)
        self.temporal_sigma_spin.setDecimals(1)
        self.temporal_sigma_spin.setValue(12.0)
        temporal_row.addWidget(self.temporal_sigma_spin)
        self.temporal_stage_drawer.content_layout.addLayout(temporal_row)

        mottle_similarity_row = QHBoxLayout()
        mottle_similarity_row.addWidget(QLabel("Similarity sigma"))
        self.mottle_similarity_sigma_spin = QDoubleSpinBox()
        self.mottle_similarity_sigma_spin.setRange(1.0, 60.0)
        self.mottle_similarity_sigma_spin.setSingleStep(0.5)
        self.mottle_similarity_sigma_spin.setDecimals(1)
        self.mottle_similarity_sigma_spin.setValue(12.0)
        self.mottle_similarity_sigma_spin.setToolTip(
            "Higher values remove more grain; lower values preserve faster intensity changes"
        )
        mottle_similarity_row.addWidget(self.mottle_similarity_sigma_spin)
        self.mottle_stage_drawer.content_layout.addLayout(mottle_similarity_row)

        mottle_window_row = QHBoxLayout()
        mottle_window_row.addWidget(QLabel("Temporal window"))
        self.mottle_window_combo = QComboBox()
        self.mottle_window_combo.addItem("3 frames", 1)
        self.mottle_window_combo.addItem("5 frames", 2)
        self.mottle_window_combo.addItem("7 frames", 3)
        self.mottle_window_combo.setCurrentIndex(1)
        self.mottle_window_combo.setToolTip("Longer windows remove more grain but span more time")
        mottle_window_row.addWidget(self.mottle_window_combo)
        self.mottle_stage_drawer.content_layout.addLayout(mottle_window_row)

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

        adjustments_brightness_row = QHBoxLayout()
        adjustments_brightness_row.addWidget(QLabel("Brightness offset"))
        self.adjustments_brightness_spin = QSpinBox()
        self.adjustments_brightness_spin.setRange(-128, 128)
        self.adjustments_brightness_spin.setValue(0)
        self.adjustments_brightness_spin.setSuffix(" levels")
        adjustments_brightness_row.addWidget(self.adjustments_brightness_spin)
        self.adjustments_stage_drawer.content_layout.addLayout(adjustments_brightness_row)

        adjustments_contrast_row = QHBoxLayout()
        adjustments_contrast_row.addWidget(QLabel("Contrast gain"))
        self.adjustments_contrast_spin = QDoubleSpinBox()
        self.adjustments_contrast_spin.setRange(0.30, 3.00)
        self.adjustments_contrast_spin.setSingleStep(0.05)
        self.adjustments_contrast_spin.setDecimals(2)
        self.adjustments_contrast_spin.setValue(1.00)
        adjustments_contrast_row.addWidget(self.adjustments_contrast_spin)
        self.adjustments_stage_drawer.content_layout.addLayout(adjustments_contrast_row)

        adjustments_sharpen_row = QHBoxLayout()
        adjustments_sharpen_row.addWidget(QLabel("Sharpen amount"))
        self.adjustments_sharpen_spin = QDoubleSpinBox()
        self.adjustments_sharpen_spin.setRange(0.00, 3.00)
        self.adjustments_sharpen_spin.setSingleStep(0.05)
        self.adjustments_sharpen_spin.setDecimals(2)
        self.adjustments_sharpen_spin.setValue(0.00)
        adjustments_sharpen_row.addWidget(self.adjustments_sharpen_spin)
        self.adjustments_stage_drawer.content_layout.addLayout(adjustments_sharpen_row)

        adjustments_gamma_row = QHBoxLayout()
        adjustments_gamma_row.addWidget(QLabel("Gamma"))
        self.adjustments_gamma_spin = QDoubleSpinBox()
        self.adjustments_gamma_spin.setRange(0.30, 3.00)
        self.adjustments_gamma_spin.setSingleStep(0.05)
        self.adjustments_gamma_spin.setDecimals(2)
        self.adjustments_gamma_spin.setValue(1.00)
        adjustments_gamma_row.addWidget(self.adjustments_gamma_spin)
        self.adjustments_stage_drawer.content_layout.addLayout(adjustments_gamma_row)

        smoothing_row = QHBoxLayout()
        smoothing_row.addWidget(QLabel("Gaussian sigma"))
        self.smoothing_sigma_spin = QDoubleSpinBox()
        self.smoothing_sigma_spin.setRange(0.1, 4.0)
        self.smoothing_sigma_spin.setSingleStep(0.05)
        self.smoothing_sigma_spin.setDecimals(2)
        self.smoothing_sigma_spin.setValue(0.55)
        smoothing_row.addWidget(self.smoothing_sigma_spin)
        self.smoothing_stage_drawer.content_layout.addLayout(smoothing_row)

        heatmap_colormap_row = QHBoxLayout()
        heatmap_colormap_row.addWidget(QLabel("Color map"))
        self.heatmap_colormap_combo = QComboBox()
        self.heatmap_colormap_combo.addItem("Hot", "hot")
        self.heatmap_colormap_combo.addItem("Inferno", "inferno")
        self.heatmap_colormap_combo.addItem("Turbo", "turbo")
        self.heatmap_colormap_combo.addItem("Viridis", "viridis")
        self.heatmap_colormap_combo.addItem("Cividis", "cividis")
        self.heatmap_colormap_combo.setToolTip("Choose the color scale without recomputing contrast residence")
        heatmap_colormap_row.addWidget(self.heatmap_colormap_combo)
        self.temporal_change_heatmap_stage_drawer.content_layout.addLayout(heatmap_colormap_row)

        for drawer in self.frame_pipeline_stage_drawers:
            self._add_sliders_to_parameter_layout(drawer.content_layout)

        self.gain_auto_target_check.toggled.connect(self._on_gain_auto_target_toggled)
        self.roi_soften_check.toggled.connect(self._on_roi_soften_toggled)
        self.roi_convex_hull_check.toggled.connect(self.on_enhancement_settings_changed)
        self.roi_circle_fit_check.toggled.connect(self.on_enhancement_settings_changed)
        self.mottle_window_combo.currentIndexChanged.connect(self.on_enhancement_settings_changed)
        self.heatmap_colormap_combo.currentIndexChanged.connect(self.on_heatmap_colormap_changed)
        for spin in [
            self.auto_crop_size_offset_spin,
            self.temporal_trim_offset_spin,
            self.comparison_sync_offset_spin,
            self.gain_target_spin,
            self.gain_min_spin,
            self.gain_max_spin,
            self.roi_radius_spin,
            self.roi_threshold_spin,
            self.scanline_clip_spin,
            self.scanline_sigma_spin,
            self.temporal_sigma_spin,
            self.mottle_similarity_sigma_spin,
            self.clahe_clip_spin,
            self.clahe_tile_spin,
            self.adjustments_brightness_spin,
            self.adjustments_contrast_spin,
            self.adjustments_sharpen_spin,
            self.adjustments_gamma_spin,
            self.smoothing_sigma_spin,
        ]:
            spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.background_video_one_check.toggled.connect(self._on_background_subtraction_video_toggled)
        self.background_video_two_check.toggled.connect(self._on_background_subtraction_video_toggled)
        self.background_video_one_level_spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.background_video_two_level_spin.valueChanged.connect(self.on_enhancement_settings_changed)
        self.background_subtraction_stage_check.toggled.connect(self._on_background_subtraction_stage_toggled)

    def _add_sliders_to_parameter_layout(self, layout: QHBoxLayout | QVBoxLayout) -> None:
        parameter_spins: list[tuple[QHBoxLayout | QVBoxLayout, QSpinBox | QDoubleSpinBox]] = []
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child_layout = item.layout()
            if isinstance(child_layout, (QHBoxLayout, QVBoxLayout)):
                self._add_sliders_to_parameter_layout(child_layout)
            widget = item.widget()
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                parameter_spins.append((layout, widget))

        for parameter_layout, spin in parameter_spins:
            self._add_parameter_slider(parameter_layout, spin)

    def _add_parameter_slider(
        self,
        layout: QHBoxLayout | QVBoxLayout,
        spin: QSpinBox | QDoubleSpinBox,
    ) -> None:
        scale = 10**spin.decimals() if isinstance(spin, QDoubleSpinBox) else 1
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(round(spin.minimum() * scale), round(spin.maximum() * scale))
        slider.setSingleStep(max(1, round(spin.singleStep() * scale)))
        slider.setPageStep(max(slider.singleStep(), (slider.maximum() - slider.minimum()) // 10))
        slider.setValue(round(spin.value() * scale))
        slider.setMinimumWidth(80)
        slider.setEnabled(spin.isEnabled())
        slider.setToolTip("Adjust value")

        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setMinimumWidth(86)
        spin.setToolTip(f"{spin.toolTip()} Enter an exact value.".strip())

        slider.valueChanged.connect(lambda value: spin.setValue(value / scale))
        spin.valueChanged.connect(lambda value: slider.setValue(round(value * scale)))
        layout.insertWidget(layout.indexOf(spin), slider, 1)
        self._parameter_slider_pairs.append((spin, slider))

    def _sync_parameter_slider_enabled(self, spin: QSpinBox | QDoubleSpinBox) -> None:
        for parameter_spin, slider in self._parameter_slider_pairs:
            if parameter_spin is spin:
                slider.setEnabled(spin.isEnabled())
                return

    def _on_gain_auto_target_toggled(self, checked: bool) -> None:
        self.gain_target_spin.setEnabled(not checked)
        self._sync_parameter_slider_enabled(self.gain_target_spin)
        self.on_enhancement_settings_changed()

    def _on_roi_soften_toggled(self, checked: bool) -> None:
        self.roi_radius_spin.setEnabled(checked)
        self.roi_threshold_spin.setEnabled(checked)
        self._sync_parameter_slider_enabled(self.roi_radius_spin)
        self._sync_parameter_slider_enabled(self.roi_threshold_spin)
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
            QWidget#modeSelectionPage { background: #0d131d; }
            QFrame#modeSelectionPanel { background: #111827; border: 1px solid #334155; border-radius: 8px; }
            QFrame#modeSelectionPanel QLabel, QWidget#modeSelectionButtons, QWidget#modeSelectionConfigActions { background: transparent; border: none; }
            QLabel#modeSelectionSectionTitle { color: #f8fafc; font-size: 18px; font-weight: 700; letter-spacing: 0.2px; }
            QFrame#modeSelectionSeparator { color: #334155; min-height: 1px; max-height: 1px; }
            QPushButton#modeSelectionButton { background: #182233; border: 1px solid #334155; border-radius: 8px; color: #e5edf6; padding: 0; }
            QPushButton#modeSelectionButton:hover { background: #202d40; border-color: #5eead4; }
            QPushButton#modeSelectionButton:pressed { background: #0f172a; border-color: #14b8a6; }
            QFrame#modePreview { background: #0b1018; border: 1px solid #334155; border-radius: 7px; }
            QFrame#modePreview[actionPreview="true"] { background: #159bb0; }
            QFrame#modePreviewPane { background: #159bb0; border: 1px solid #67e8f9; border-radius: 5px; }
            QWidget#modeActionGlyph { background: transparent; border: none; }
            QLabel#modeSelectionLabel { color: #f8fafc; font-size: 16px; font-weight: 700; }
            QFrame#videoDropPlaceholder { background: #111827; border: 1px solid #334155; border-radius: 8px; }
            QFrame#videoDropPlaceholder[dragActive="true"] { background: #072226; border: 1px solid #2dd4bf; }
            QFrame#videoDropSurface { background: #000000; border: 1px solid #253044; border-radius: 8px; }
            QFrame#videoDropSurface[dragActive="true"] { background: #0b1018; border: 1px solid #2dd4bf; }
            QPushButton#videoDropHintButton { background: #173244; border: 1px solid #253044; border-radius: 10px; padding: 0; }
            QPushButton#videoDropHintButton:hover { background: #20435b; border-color: #334155; }
            QPushButton#videoDropHintButton[dragActive="true"] { background: #1a4b5f; border-color: #2dd4bf; }
            QWidget#videoDropGlyph { background: transparent; }
            QSlider::groove:horizontal { height: 6px; background: #273449; border-radius: 3px; }
            QSlider::handle:horizontal { background: #67e8f9; width: 16px; margin: -5px 0; border-radius: 8px; }
            QSplitter::handle:vertical { background: #1c2637; border-top: 1px solid #334155; border-bottom: 1px solid #253044; margin: 2px 0; }
            QSplitter::handle:horizontal { background: #1c2637; border-left: 1px solid #334155; border-right: 1px solid #253044; margin: 0; }
            QSpinBox, QDoubleSpinBox, QComboBox { background: #111827; border: 1px solid #334155; border-radius: 6px; padding: 5px; color: #e5edf6; }
            QLineEdit#liveRecordingField { background: #111827; border: 1px solid #5b718c; border-radius: 5px; padding: 5px 7px; color: #f8fafc; }
            QLineEdit#liveRecordingField:focus { border-color: #5eead4; }
            QCheckBox#liveRecordingPhaseToggle { background: #111827; border: 1px solid #5b718c; border-radius: 5px; padding: 5px 8px; color: #e5edf6; }
            QCheckBox#liveRecordingPhaseToggle:hover { border-color: #7dd3fc; }
            QCheckBox#liveRecordingPhaseToggle:checked { background: #134e4a; border-color: #2dd4bf; }
            QGroupBox { background: #111827; border: 1px solid #253044; border-radius: 8px; margin-top: 12px; padding: 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #f8fafc; }
            QFrame#videoPanel, QFrame#plotPanel { background: #111827; border: 1px solid #253044; border-radius: 8px; }
            QFrame#graphDrawer { background: #0f172a; border: 1px solid #253044; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QWidget#graphDrawerHeader { background: #111827; border-top: 1px solid #253044; border-bottom: 1px solid #334155; }
            QLabel#graphDrawerTitle { color: #f8fafc; font-size: 14px; font-weight: 700; padding-left: 6px; }
            QLabel#graphDrawerHandle { color: #9fb0c6; font-weight: 700; letter-spacing: 1px; }
            QToolButton#graphDrawerToggle { background: #1c2637; border: 1px solid #334155; border-radius: 6px; padding: 0; }
            QToolButton#graphDrawerToggle:hover { background: #263449; }
            QFrame#pipelineDrawer { background: #0f172a; border: 1px solid #253044; border-radius: 8px; }
            QWidget#pipelineDrawerHeader { background: #111827; border-top: 1px solid #253044; border-bottom: 1px solid #334155; }
            QLabel#pipelineDrawerTitle { color: #f8fafc; font-size: 14px; font-weight: 700; padding-left: 6px; }
            QToolButton#pipelineDrawerToggle { background: #1c2637; border: 1px solid #334155; border-radius: 6px; padding: 0; }
            QToolButton#pipelineDrawerToggle:hover { background: #263449; }
            QFrame#stageDrawer { background: #0f172a; border: 1px solid #273449; border-radius: 8px; }
            QToolButton#stageEnableButton, QToolButton#stageExpandButton, QToolButton#stageOptionsButton { border: 1px solid transparent; border-radius: 6px; icon-size: 16px; padding: 0; }
            QToolButton#stageEnableButton:hover, QToolButton#stageExpandButton:hover, QToolButton#stageOptionsButton:hover { background: #1c2637; border-color: #334155; }
            QToolButton#stageEnableButton:checked { background: #134e4a; border-color: #14b8a6; }
            QLabel#stageGrabHandle { color: #9fb0c6; font-weight: 700; }
            QLabel#stageGrabHandle:disabled { color: #475569; }
            QLabel#stageLabel { color: #f8fafc; font-weight: 700; }
            QLabel#panelTitle { font-size: 16px; font-weight: 700; color: #f8fafc; }
            QLabel#pipelineLabel { color: #e2e8f0; font-weight: 700; padding-top: 4px; }
            QLabel#subtleLabel, QLabel#hintLabel { color: #9fb0c6; }
            QLabel#timeLabel { color: #cbd5e1; padding-left: 12px; }
            QFrame#metricCard { background: #0f172a; border: 1px solid #273449; border-radius: 8px; }
            QLabel#metricTitle { color: #9fb0c6; font-size: 12px; }
            QLabel#metricValue { color: #f8fafc; font-size: 22px; font-weight: 800; }
            QLabel#metricDetail { color: #94a3b8; font-size: 12px; }
            QStatusBar { background: #0b1018; color: #9fb0c6; }
            QFrame#enhancementProgressPanel { background: #0f172a; border: 1px solid #334155; border-radius: 8px; }
            QLabel#enhancementProgressLabel { color: #e2e8f0; font-size: 13px; font-weight: 700; }
            QProgressBar { border: 1px solid #334155; border-radius: 6px; background: #111827; color: #e5edf6; text-align: center; }
            QProgressBar::chunk { background: #14b8a6; border-radius: 5px; }
            """
        )
        self.mode_selection_panel.setStyleSheet(self.styleSheet())

    def open_video(self, panel: VideoPanel | None) -> None:
        if panel is None:
            return
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
            LOGGER.exception("Could not open %s video from %s", panel.label, path)
            QMessageBox.critical(self, "Could not open video", str(exc))
            return
        LOGGER.info("Loaded %s video from %s", panel.label, path)

        self.results.clear()
        self._sync_trimmed_video_window()
        self.frame_slider.setRange(0, self.max_frame)
        self.frame_spin.setRange(0, self.max_frame)
        self.set_frame_index(0)
        self.clear_plots_and_metrics()
        self._update_stage_statuses()

        if self._pipeline_has_active_stage():
            self.rebuild_enhancement_pipeline()
        else:
            self.set_display_enhancement(False)

    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.active_mode != MODE_LIVE and self.panels and self.current_frame_index >= self.max_frame:
            self.set_frame_index(0)
        self.is_playing = True
        self.play_button.setText("")
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
        self.play_button.setText("")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        if self.active_mode != MODE_LIVE and self.panels and any(panel.enhance_display for panel in self.panels):
            for panel in self.panels:
                panel.seek(self.current_frame_index)

    def advance_frame(self) -> None:
        if not self.panels:
            self.pause()
            return
        live_input = self.active_mode == MODE_LIVE
        if not live_input and self.current_frame_index >= self.max_frame:
            if self._background_pipeline_work_active():
                return
            if self.loop_check.isChecked():
                self.set_frame_index(0)
                return
            self.pause()
            return
        next_frame_index = self.current_frame_index + 1
        if not all(panel.read_next(playback=True) for panel in self.panels):
            self.pause()
            return
        self.current_frame_index = next_frame_index
        if live_input:
            self._render_live_frame()
        for widget in [self.frame_slider, self.frame_spin]:
            if widget.value() != next_frame_index:
                widget.blockSignals(True)
                widget.setValue(next_frame_index)
                widget.blockSignals(False)
        self.update_time_label()

    def set_frame_index(self, frame_index: int) -> None:
        if not self.panels:
            self.current_frame_index = 0
            for widget in [self.frame_slider, self.frame_spin]:
                if widget.value() != 0:
                    widget.blockSignals(True)
                    widget.setValue(0)
                    widget.blockSignals(False)
            self.update_time_label()
            return
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

    def _set_playback_limit(self, frame_index: int) -> None:
        self.max_frame = max(0, min(frame_index, self.source_max_frame))
        for widget in [self.frame_slider, self.frame_spin]:
            widget.blockSignals(True)
            widget.setRange(0, self.max_frame)
            widget.blockSignals(False)
        if self.current_frame_index > self.max_frame:
            self.current_frame_index = self.max_frame
            for panel in self.panels:
                panel.seek(self.current_frame_index)
            for widget in [self.frame_slider, self.frame_spin]:
                widget.blockSignals(True)
                widget.setValue(self.current_frame_index)
                widget.blockSignals(False)
        self.update_time_label()

    def update_time_label(self) -> None:
        if self.active_mode == MODE_LIVE:
            self.time_label.setText(f"LIVE | {self.current_frame_index / self.fps:05.2f} s")
            return
        current_time = self.current_frame_index / self.fps if self.fps else 0.0
        total_time = self.max_frame / self.fps if self.fps else 0.0
        self.time_label.setText(f"{current_time:05.2f} s / {total_time:05.2f} s")

    def _missing_stage_roi_labels(self) -> list[str]:
        return [panel.label for panel in self.panels if not panel.has_stage_roi_mask()]

    def _stage_drawers(self, stage_key: str) -> list[StageDrawer]:
        return [drawer for drawer in self.pipeline_stage_drawers if drawer.stage_key == stage_key]

    def _has_enabled_stage(self, stage_key: str) -> bool:
        return any(drawer.enable_button.isChecked() for drawer in self._stage_drawers(stage_key))

    def _background_pipeline_work_active(self) -> bool:
        return self._enhancement_future is not None or self._analysis_future is not None

    def _analysis_requirement_failure(self) -> str | None:
        roi_extraction_enabled = (
            self._has_enabled_stage("roi_extraction")
            if hasattr(self, "pipeline_stage_drawers")
            else self.roi_stage_check.isChecked()
        )
        if not roi_extraction_enabled:
            return "ROI residence analysis failed: enable upstream aneurysm ROI extraction."
        missing_masks = self._missing_stage_roi_labels()
        if missing_masks:
            labels = ", ".join(missing_masks)
            return f"ROI residence analysis failed: ROI extraction did not produce masks for {labels}."
        return None

    def _update_stage_statuses(self) -> None:
        roi_drawers = self._stage_drawers("roi_extraction")
        alignment_drawers = self._stage_drawers("roi_needle_level_alignment")
        analysis_drawers = self._stage_drawers("roi_residence_analysis")
        frame_brightness_drawers = self._stage_drawers("frame_brightness_analysis")
        needle_drawers = self._stage_drawers("needle_segmentation")
        temporal_change_drawers = self._stage_drawers("temporal_change_heatmap")
        analysis_enabled = self._has_enabled_stage("roi_residence_analysis")
        frame_brightness_enabled = self._has_enabled_stage("frame_brightness_analysis")
        needle_enabled = self._has_enabled_stage("needle_segmentation")
        alignment_enabled = self._has_enabled_stage("roi_needle_level_alignment")
        temporal_change_enabled = self._has_enabled_stage("temporal_change_heatmap")
        if self.active_mode == MODE_LIVE and self._network_stream_display is not None:
            for drawer in roi_drawers:
                drawer.set_status("Unavailable for live input. Drag on the source image to select an ROI.", False)
            for drawer in alignment_drawers:
                drawer.set_status("Unavailable for live input.", alignment_enabled)
            for drawer in analysis_drawers:
                drawer.set_status(
                    "Collecting the latest 60 seconds from the manually selected ROI."
                    if analysis_enabled
                    else None,
                    False,
                )
            for drawer in frame_brightness_drawers:
                drawer.set_status(
                    "Collecting the latest 60 seconds of source and enhanced brightness."
                    if frame_brightness_enabled
                    else None,
                    False,
                )
            for drawer in needle_drawers:
                drawer.set_status("Unavailable for live input.", needle_enabled)
            for drawer in temporal_change_drawers:
                drawer.set_status("Unavailable for live input.", temporal_change_enabled)
            return
        if not self.panels:
            for drawer in roi_drawers:
                drawer.set_status("Load video files to run ROI extraction.", analysis_enabled)
            for drawer in alignment_drawers:
                drawer.set_status("Load a two-video comparison to align baseline levels.", alignment_enabled)
            for drawer in analysis_drawers:
                drawer.set_status("Load video files to run ROI residence analysis.", analysis_enabled)
            for drawer in frame_brightness_drawers:
                drawer.set_status("Load video files to run frame brightness analysis.", frame_brightness_enabled)
            for drawer in needle_drawers:
                drawer.set_status("Load video files to segment the stationary needle.", needle_enabled)
            for drawer in temporal_change_drawers:
                drawer.set_status("Load video files to build a contrast residence heatmap.", temporal_change_enabled)
            return
        if not self._has_enabled_stage("roi_extraction"):
            extraction_message = "Disabled. Enable this stage to extract aneurysm ROI masks from the current enhanced video."
            for drawer in roi_drawers:
                drawer.set_status(extraction_message, analysis_enabled)
        elif self._background_pipeline_work_active():
            for drawer in roi_drawers:
                drawer.set_status("Running ROI extraction on the current enhanced video...", False)
        else:
            missing_masks = self._missing_stage_roi_labels()
            if missing_masks:
                labels = ", ".join(missing_masks)
                for drawer in roi_drawers:
                    drawer.set_status(f"Failed for {labels}. Adjust upstream stages or ROI extraction parameters.", True)
            else:
                for drawer in roi_drawers:
                    drawer.set_status("ROI masks ready for downstream analysis.", False)

        if not alignment_enabled:
            for drawer in alignment_drawers:
                drawer.set_status(None)
        elif self.active_mode != MODE_COMPARISON or len(self.panels) != 2:
            for drawer in alignment_drawers:
                drawer.set_status("Requires a two-video comparison.", True)
        else:
            enabled_live_drawers = [
                drawer for drawer in self.live_pipeline_stage_drawers if drawer.enable_button.isChecked()
            ]
            alignment_index = next(
                (index for index, drawer in enumerate(enabled_live_drawers) if drawer.stage_key == "roi_needle_level_alignment"),
                -1,
            )
            has_upstream_roi = any(
                drawer.stage_key == "roi_extraction" for drawer in enabled_live_drawers[:alignment_index]
            )
            for drawer in alignment_drawers:
                if not has_upstream_roi:
                    drawer.set_status("Move this stage after enabled aneurysm ROI extraction.", True)
                elif self._background_pipeline_work_active():
                    drawer.set_status("Measuring pre-injection ROI and needle baseline levels...", False)
                else:
                    drawer.set_status("The narrower baseline range is aligned to the wider-range video.", False)

        if not analysis_enabled:
            for drawer in analysis_drawers:
                drawer.set_status(None)
        else:
            failure = self._analysis_requirement_failure()
            if failure is not None:
                for drawer in analysis_drawers:
                    drawer.set_status(failure, True)
            elif self._background_pipeline_work_active():
                for drawer in analysis_drawers:
                    drawer.set_status("Waiting for upstream ROI extraction to finish.", False)
            else:
                for drawer in analysis_drawers:
                    drawer.set_status("Ready to analyze the current ROI masks.", False)

        if not frame_brightness_enabled:
            for drawer in frame_brightness_drawers:
                drawer.set_status(None)
        elif self._background_pipeline_work_active():
            for drawer in frame_brightness_drawers:
                drawer.set_status("Waiting for enhanced video frames.", False)
        else:
            for drawer in frame_brightness_drawers:
                drawer.set_status("Ready to compare original and enhanced frame brightness.", False)

        if not needle_enabled:
            for drawer in needle_drawers:
                drawer.set_status(None)
        elif self._background_pipeline_work_active():
            for drawer in needle_drawers:
                drawer.set_status("Waiting for enhanced video frames.", False)
        else:
            missing_masks = [panel.label for panel in self.panels if panel.needle_segmentation_mask is None]
            for drawer in needle_drawers:
                drawer.set_status(
                    f"No single darkest needle region found for {', '.join(missing_masks)}."
                    if missing_masks
                    else "Needle masks and full-duration brightness traces are ready.",
                    bool(missing_masks),
                )

        if not temporal_change_enabled:
            for drawer in temporal_change_drawers:
                drawer.set_status(None)
        elif self._background_pipeline_work_active():
            for drawer in temporal_change_drawers:
                drawer.set_status("Waiting for enhanced video frames.", False)
        else:
            for drawer in temporal_change_drawers:
                drawer.set_status("Ready to map time-integrated and peak dark contrast.", False)

    def on_roi_changed(self) -> None:
        if self.active_mode == MODE_LIVE and self._network_stream_display is not None:
            self._live_measurements.clear()
            self.results.clear()
            self.clear_plots_and_metrics()
            self._update_stage_statuses()
            self.statusBar().showMessage("Live ROI selected. Collecting measurements for the latest 60 seconds.")
            return
        if not self.panels:
            self.statusBar().showMessage("Load a video to begin processing.")
            return
        self.results.clear()
        self.clear_roi_analysis()
        self._update_stage_statuses()
        ready = all(panel.roi() and panel.roi_mask() is not None for panel in self.panels)
        if ready:
            if self._has_enabled_stage("roi_residence_analysis") and not self._background_pipeline_work_active():
                if self.run_analysis():
                    return
            self.statusBar().showMessage(
                "ROI masks are ready. Enable ROI residence analysis to compare contrast residence."
            )
        else:
            self.statusBar().showMessage("ROI extraction needs review; refresh the stage or draw a correction.")

    def redetect_aneurysms(self) -> None:
        self.pause()
        if not self._has_enabled_stage("roi_extraction"):
            self._update_stage_statuses()
            self.statusBar().showMessage("Enable the Aneurysm ROI extraction stage before refreshing ROI masks.")
            return
        self.rebuild_enhancement_pipeline()

    def set_display_enhancement(self, enabled: bool) -> None:
        if not self.panels:
            return
        if self.active_mode == MODE_LIVE:
            self._render_live_frame()
            self.statusBar().showMessage("Live enhancement enabled.")
            return
        for panel in self.panels:
            panel.set_enhancement(enabled, self.current_frame_index)
        self.statusBar().showMessage("Video enhancement enabled." if enabled else "Video enhancement disabled.")

    def _render_live_frame(self) -> None:
        if self.active_mode != MODE_LIVE:
            return
        stages = self.enhancement_stages()
        parameters = self.enhancement_parameters()
        denoiser = self._live_denoiser_for(stages)
        for panel in self.panels:
            panel.apply_live_enhancement(stages, parameters, denoiser, self.denoise_strength_spin.value())

    def _live_denoiser_for(self, stages: EnhancementStages) -> FrameDenoiser | None:
        mode = str(self.enhancement_mode_combo.currentData())
        if "denoise" not in stages.enabled_stage_order:
            return None
        key = self._denoiser_key(mode)
        denoiser = self.deep_denoisers.get(key)
        if denoiser is not None:
            return denoiser
        if mode == "non-local-means":
            denoiser = NonLocalMeansDenoiser()
        elif mode == "tensor-nlm-ngc":
            from container_denoiser import ContainerDenoiser

            denoiser = ContainerDenoiser(
                "tensor-nlm",
                None,
                self.inference_batch_spin.value(),
                str(self.inference_precision_combo.currentData()),
            )
        elif mode.endswith("-ngc"):
            from container_denoiser import ContainerDenoiser

            denoiser = ContainerDenoiser(
                "ffdnet",
                ROOT / "models" / "ffdnet_gray.pth",
                self.inference_batch_spin.value(),
                str(self.inference_precision_combo.currentData()),
            )
        elif mode == "ffdnet-native":
            from deep_denoiser import FFDNetDenoiser

            denoiser = FFDNetDenoiser(
                ROOT / "models" / "ffdnet_gray.pth",
                str(self.inference_precision_combo.currentData()),
            )
        else:
            raise ValueError(f"Unsupported denoiser mode: {mode}")
        self.deep_denoisers[key] = denoiser
        return denoiser

    def on_compare_view_toggled(self, enabled: bool) -> None:
        if enabled and self.temporal_change_view_check.isChecked():
            self.temporal_change_view_check.blockSignals(True)
            self.temporal_change_view_check.setChecked(False)
            self.temporal_change_view_check.blockSignals(False)
        for panel in self.panels:
            panel.set_comparison(enabled, self.current_frame_index)
        if self._network_stream_display is not None:
            self._network_stream_display.set_comparison_enabled(enabled)
        if enabled:
            self.statusBar().showMessage("Showing source beside enhanced output.")
        else:
            self.statusBar().showMessage("Showing enhanced output only.")

    def on_temporal_change_view_toggled(self, enabled: bool) -> None:
        if enabled and not self.temporal_change_results:
            self.temporal_change_view_check.blockSignals(True)
            self.temporal_change_view_check.setChecked(False)
            self.temporal_change_view_check.blockSignals(False)
            return
        if enabled:
            self.compare_view_check.blockSignals(True)
            self.compare_view_check.setChecked(False)
            self.compare_view_check.blockSignals(False)
        for panel in self.panels:
            panel.set_temporal_change_display(enabled, self.current_frame_index)
        self.statusBar().showMessage(
            "Showing contrast residence beside enhanced output." if enabled else "Showing enhanced output only."
        )

    def on_heatmap_colormap_changed(self) -> None:
        if not self.temporal_change_results:
            return
        self.refresh_temporal_change_views()
        self.statusBar().showMessage("Contrast residence heatmap color map updated.")

    def on_roi_region_overlay_toggled(self, enabled: bool) -> None:
        if not self.panels:
            return
        for panel in self.panels:
            panel.set_roi_region_overlay(enabled, self.current_frame_index)
        self.statusBar().showMessage(
            "ROI region overlay visible." if enabled else "ROI region overlay hidden."
        )

    def on_enhancement_settings_changed(self) -> None:
        if self._loading_config:
            return
        active_mode = str(self.enhancement_mode_combo.currentData())
        stages = self.enhancement_stages()
        uses_ffdnet = stages.denoise and active_mode.startswith("ffdnet")
        uses_accelerator = uses_ffdnet or (stages.denoise and active_mode == "tensor-nlm-ngc")
        uses_spatial_denoiser = stages.denoise
        self.overlay_mask_check.setEnabled(
            bool(self.panels)
            and (stages.roi_extraction or self._has_enabled_stage("needle_segmentation"))
        )
        self.denoise_strength_label.setEnabled(uses_spatial_denoiser)
        self.denoise_strength_spin.setEnabled(uses_spatial_denoiser)
        self.inference_batch_spin.setEnabled(uses_accelerator)
        self._sync_parameter_slider_enabled(self.denoise_strength_spin)
        self._sync_parameter_slider_enabled(self.inference_batch_spin)
        self.inference_precision_combo.setEnabled(uses_accelerator)
        self._sync_active_denoise_controls()
        self._refresh_desktop_stream_pipeline()
        if self.active_mode == MODE_LIVE:
            self._apply_source_pipeline_stages()
            self._render_live_frame()
        elif self._pipeline_has_active_stage():
            self.rebuild_enhancement_pipeline()
        else:
            self._stop_enhancement_preview()
            self._update_stage_statuses()
            self.statusBar().showMessage("Enhancement settings updated. Enable one or more stages to preview.")

    def enhancement_stages(self) -> EnhancementStages:
        instances = tuple(
            PipelineStage(
                key=drawer.stage_key,
                enabled=(
                    drawer.enable_button.isChecked()
                    and (drawer.stage_key != "roi_needle_level_alignment" or self.active_mode == MODE_COMPARISON)
                ),
                parameters=self._drawer_parameters(drawer),
                noise_sigma=(
                    drawer.findChild(QSpinBox, "denoiseStrength").value()
                    if drawer.findChild(QSpinBox, "denoiseStrength") is not None
                    else None
                ),
            )
            for drawer in self.frame_pipeline_stage_drawers
        )
        return EnhancementStages(
            gain_stabilization=self._has_enabled_stage("gain_stabilization"),
            brightness_stabilization=self._has_enabled_stage("brightness_stabilization"),
            camera_motion_stabilization=self._has_enabled_stage("camera_motion_stabilization"),
            roi_extraction=self._has_enabled_stage("roi_extraction"),
            roi_needle_level_alignment=(
                self.active_mode == MODE_COMPARISON and self._has_enabled_stage("roi_needle_level_alignment")
            ),
            scanline_correction=self._has_enabled_stage("scanline_correction"),
            denoise=self._has_enabled_stage("denoise"),
            temporal_filter=self._has_enabled_stage("temporal_filter"),
            quantum_mottle_filter=self._has_enabled_stage("quantum_mottle_filter"),
            local_contrast=self._has_enabled_stage("local_contrast"),
            image_adjustments=self._has_enabled_stage("image_adjustments"),
            final_smoothing=self._has_enabled_stage("final_smoothing"),
            stage_order=tuple(stage.key for stage in instances),
            instances=instances,
        )

    def _pipeline_has_active_stage(self) -> bool:
        return (
            self.enhancement_stages().any_enabled
            or self._has_enabled_stage("roi_residence_analysis")
            or self._has_enabled_stage("frame_brightness_analysis")
            or self._has_enabled_stage("needle_segmentation")
            or self._has_enabled_stage("parent_vessel_roi_analysis")
            or self._has_enabled_stage("background_roi_analysis")
            or self._has_enabled_stage("temporal_change_heatmap")
            or self._has_enabled_stage("auto_crop")
            or self._has_enabled_stage("startup_stabilization_trim")
            or self._has_enabled_stage("temporal_alignment")
            or self._has_enabled_stage("contrast_gain_alignment")
                or self._has_enabled_stage("histogram_matching")
            or self._has_enabled_stage("metal_needle_removal")
            or any(enabled for enabled, _level in self._background_subtraction_settings())
            or any(
                panel.crop_rect != panel._full_frame_rect() or panel.trim_start_frame != 0
                for panel in self.panels
            )
        )

    def _current_backend_id(self, stages: EnhancementStages) -> str:
        mode = str(self.enhancement_mode_combo.currentData())
        if not stages.denoise:
            return "none"
        if mode == "non-local-means":
            return DEFAULT_NLM_BACKEND_ID
        precision = str(self.inference_precision_combo.currentData())
        if mode.endswith("-ngc"):
            return f"{mode.removesuffix('-ngc')}-ngc-26.06-{precision}-batch{self.inference_batch_spin.value()}"
        return f"ffdnet-native-{precision}"

    def _is_fixed_pipeline_stage(self, drawer: StageDrawer) -> bool:
        return False

    def _sync_pipeline_stage_lists(self) -> None:
        self.pipeline_stage_drawers = [
            *self.source_pipeline_stage_drawers,
            *self.live_pipeline_stage_drawers,
        ]
        self.frame_pipeline_stage_drawers = [
            drawer
            for drawer in self.live_pipeline_stage_drawers
            if drawer.stage_key
            not in {
                "roi_residence_analysis",
                "frame_brightness_analysis",
                "needle_segmentation",
                "parent_vessel_roi_analysis",
                "background_roi_analysis",
                "temporal_change_heatmap",
            }
        ]
        self.source_pipeline_stage_checks = [item.enable_button for item in self.source_pipeline_stage_drawers]
        self.frame_pipeline_stage_checks = [item.enable_button for item in self.frame_pipeline_stage_drawers]
        self.pipeline_stage_checks = [drawer.enable_button for drawer in self.pipeline_stage_drawers]

    def _pipeline_drawers_for(self, drawer: StageDrawer) -> list[StageDrawer]:
        if drawer in self.source_pipeline_stage_drawers:
            return self.source_pipeline_stage_drawers
        return self.live_pipeline_stage_drawers

    def _pipeline_layout_for(self, drawer: StageDrawer) -> QVBoxLayout:
        if drawer in self.source_pipeline_stage_drawers:
            return self.source_pipeline_layout
        return self.live_pipeline_layout

    def _reorder_pipeline_stage_by_key(self, source_key: str, target_key: str) -> None:
        if source_key == target_key:
            return
        source_drawer = next((drawer for drawer in self.pipeline_stage_drawers if drawer.stage_key == source_key), None)
        target_drawer = next((drawer for drawer in self.pipeline_stage_drawers if drawer.stage_key == target_key), None)
        if source_drawer is None or target_drawer is None:
            return
        if self._is_fixed_pipeline_stage(source_drawer):
            return
        if self._is_fixed_pipeline_stage(target_drawer):
            return
        drawers = self._pipeline_drawers_for(source_drawer)
        if drawers is not self._pipeline_drawers_for(target_drawer):
            return
        source_index = drawers.index(source_drawer)
        target_index = drawers.index(target_drawer)
        moving_drawer = drawers.pop(source_index)
        if source_index < target_index:
            target_index -= 1
        drawers.insert(target_index, moving_drawer)
        self._sync_pipeline_stage_lists()
        self._refresh_pipeline_stage_ui()
        self.on_enhancement_settings_changed()

    def _begin_pipeline_stage_drag(self, drawer: StageDrawer, global_position: QPoint) -> None:
        if self._is_fixed_pipeline_stage(drawer):
            return
        self._dragged_stage_drawer = drawer
        self._dragged_pipeline_order = list(self._pipeline_drawers_for(drawer))
        self._dragged_pipeline_layout = self._pipeline_layout_for(drawer)
        layout = self._dragged_pipeline_layout
        self._stage_drag_placeholder = QWidget(layout.parentWidget())
        self._stage_drag_placeholder.setFixedHeight(drawer.height())
        self._stage_drag_placeholder.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        drawer_geometry = drawer.geometry()
        self._stage_drag_x = drawer_geometry.x()
        self._stage_drag_offset_y = layout.parentWidget().mapFromGlobal(global_position).y() - drawer_geometry.y()
        layout.replaceWidget(drawer, self._stage_drag_placeholder)
        drawer.setGeometry(drawer_geometry)
        drawer.raise_()
        self._move_pipeline_stage_drag(global_position)

    def _move_pipeline_stage_drag(self, global_position: QPoint) -> None:
        drawer = self._dragged_stage_drawer
        placeholder = self._stage_drag_placeholder
        order = self._dragged_pipeline_order
        if drawer is None or placeholder is None or order is None:
            return

        layout = self._dragged_pipeline_layout
        if layout is None:
            return
        parent = layout.parentWidget()
        cursor_y = parent.mapFromGlobal(global_position).y()
        other_drawers = [item for item in order if item is not drawer]
        if not other_drawers:
            return

        clamp_widgets: list[QWidget] = [*other_drawers, placeholder]
        top = min(item.geometry().top() for item in clamp_widgets)
        bottom = max(item.geometry().bottom() for item in clamp_widgets) - drawer.height()
        drawer.move(self._stage_drag_x, max(top, min(cursor_y - self._stage_drag_offset_y, bottom)))

        source_index = order.index(drawer)
        insertion_index = len(other_drawers)
        for index, item in enumerate(other_drawers):
            if cursor_y < item.geometry().center().y():
                insertion_index = index
                break

        order.pop(source_index)
        order.insert(insertion_index, drawer)

        add_button = self.source_add_stage_button if drawer in self.source_pipeline_stage_drawers else self.live_add_stage_button
        next_drawer = order[insertion_index + 1] if insertion_index < len(order) - 1 else add_button
        layout.removeWidget(placeholder)
        layout.insertWidget(layout.indexOf(next_drawer), placeholder)

    def _finish_pipeline_stage_drag(self, global_position: QPoint) -> None:
        self._move_pipeline_stage_drag(global_position)
        drawer = self._dragged_stage_drawer
        placeholder = self._stage_drag_placeholder
        order = self._dragged_pipeline_order
        if drawer is None or placeholder is None or order is None:
            return
        layout = self._dragged_pipeline_layout
        if layout is None:
            return
        placeholder_index = layout.indexOf(placeholder)
        layout.removeWidget(placeholder)
        placeholder.deleteLater()
        layout.insertWidget(placeholder_index, drawer)
        self._pipeline_drawers_for(drawer)[:] = order
        self._sync_pipeline_stage_lists()
        self._dragged_stage_drawer = None
        self._stage_drag_placeholder = None
        self._dragged_pipeline_order = None
        self._dragged_pipeline_layout = None
        self._refresh_pipeline_stage_ui()
        self.on_enhancement_settings_changed()

    def _refresh_pipeline_stage_ui(self) -> None:
        for drawer in self.pipeline_stage_drawers:
            self.source_pipeline_layout.removeWidget(drawer)
            self.live_pipeline_layout.removeWidget(drawer)
        self.source_pipeline_layout.removeWidget(self.source_add_stage_button)
        self.live_pipeline_layout.removeWidget(self.live_add_stage_button)
        for offset, drawer in enumerate(self.source_pipeline_stage_drawers):
            drawer.set_stage_index(offset + 1)
            drawer.set_reorder_enabled(not self._is_fixed_pipeline_stage(drawer))
            self.source_pipeline_layout.insertWidget(offset, drawer)
            drawer.show()
        self.source_pipeline_layout.insertWidget(len(self.source_pipeline_stage_drawers), self.source_add_stage_button)
        for offset, drawer in enumerate(self.live_pipeline_stage_drawers):
            drawer.set_stage_index(offset + 1)
            drawer.set_reorder_enabled(not self._is_fixed_pipeline_stage(drawer))
            self.live_pipeline_layout.insertWidget(offset, drawer)
            drawer.show()
        self.live_pipeline_layout.insertWidget(len(self.live_pipeline_stage_drawers), self.live_add_stage_button)
        self._update_pipeline_column_width()

    def _update_temporal_alignment_controls(self) -> None:
        self.comparison_sync_offset_row.setVisible(self.active_mode == MODE_COMPARISON)
        histogram_matching_available = self.active_mode == MODE_COMPARISON
        self.histogram_matching_stage_check.setEnabled(histogram_matching_available)
        self.histogram_matching_stage_check.setToolTip(
            "Enable stage" if histogram_matching_available else "Available only for two-video comparison"
        )
        labels = ("Pre-deployment", "Post-deployment") if self.active_mode == MODE_COMPARISON else ("Video",)
        checks = (self.background_video_one_check, self.background_video_two_check)
        for index, row in enumerate(self.background_video_rows):
            row.setVisible(index < len(labels))
            if index < len(labels):
                checks[index].setText(labels[index])
        for index, row in enumerate(self.roi_video_mode_rows):
            row.setVisible(index == 0 or self.active_mode == MODE_COMPARISON)
            if index == 0:
                row.findChild(QLabel).setText("Pre-deployment ROI mode" if self.active_mode == MODE_COMPARISON else "ROI mode")
        for index, row in enumerate(self.parent_vessel_rotation_rows):
            row.setVisible(index == 0 or self.active_mode == MODE_COMPARISON)

    def _roi_parameters_for_panel(self, panel_index: int) -> EnhancementParameters:
        parameters = self.enhancement_parameters()
        mode_combo = self.roi_mode_combos[min(panel_index, len(self.roi_mode_combos) - 1)]
        manual_circle = self._manual_roi_circles[panel_index] if panel_index < len(self._manual_roi_circles) else None
        return replace(parameters, roi_mode=str(mode_combo.currentData()), roi_manual_circle=manual_circle)

    def _stages_for_roi_parameters(
        self,
        stages: EnhancementStages,
        roi_parameters: EnhancementParameters,
    ) -> EnhancementStages:
        return replace(
            stages,
            instances=tuple(
                replace(stage, parameters=roi_parameters) if stage.key == "roi_extraction" else stage
                for stage in stages.instances
            ),
        )

    def _prepare_roi_needle_level_alignment(
        self,
        request: EnhancementRequest,
        panel_denoisers: list[FrameDenoiser | None],
        backend_id: str,
        cancel_event: Event,
    ) -> None:
        self.roi_needle_alignment_results = {}
        for panel in self.panels:
            panel.roi_needle_alignment_gain = 1.0
            panel.roi_needle_alignment_offset = 0.0

        enabled_stages = request.stages.enabled_stage_instances(request.parameters)
        alignment_indexes = [
            index for index, stage in enumerate(enabled_stages) if stage.key == "roi_needle_level_alignment"
        ]
        if not alignment_indexes:
            return
        if len(alignment_indexes) != 1:
            raise ValueError("ROI / needle level alignment can appear only once in the live pipeline.")
        if self.active_mode != MODE_COMPARISON or len(self.panels) != 2:
            raise ValueError("ROI / needle level alignment requires a two-video comparison.")

        alignment_index = alignment_indexes[0]
        prefix_instances = enabled_stages[:alignment_index]
        if not any(stage.key == "roi_extraction" for stage in prefix_instances):
            raise ValueError("ROI / needle level alignment requires upstream aneurysm ROI extraction.")
        prefix_stages = EnhancementStages(
            stage_order=tuple(stage.key for stage in prefix_instances),
            instances=prefix_instances,
        )
        with self._enhancement_progress_lock:
            self._enhancement_message = "Measuring ROI and needle baseline levels..."

        def prepare_prefix(panel_index: int, frame_executor: AdaptiveFrameExecutor) -> bool:
            roi_parameters = (
                request.roi_parameters_by_panel[panel_index]
                if panel_index < len(request.roi_parameters_by_panel)
                else request.parameters
            )
            return self.panels[panel_index].prepare_enhanced_frames(
                panel_denoisers[panel_index],
                request.noise_sigma,
                request.batch_size,
                self._stages_for_roi_parameters(prefix_stages, roi_parameters),
                request.parameters,
                activate_result=False,
                cancel_callback=cancel_event.is_set,
                frame_executor=frame_executor,
            )

        prepared = [False] * len(self.panels)
        with frame_parallel_opencv(), AdaptiveFrameExecutor() as frame_executor, ThreadPoolExecutor(
            max_workers=len(self.panels),
            thread_name_prefix="level-alignment",
        ) as executor:
            futures = {
                executor.submit(prepare_prefix, panel_index, frame_executor): panel_index
                for panel_index in range(len(self.panels))
            }
            for future in as_completed(futures):
                prepared[futures[future]] = future.result()
        if cancel_event.is_set():
            return
        if not all(prepared):
            raise RuntimeError("Could not prepare the ROI / needle alignment prefix.")

        anchors: list[tuple[float, float]] = []
        roi_curves: list[np.ndarray] = []
        needle_masks: list[np.ndarray] = []
        alignment_frames: list[list[np.ndarray]] = []
        for panel_index, panel in enumerate(self.panels):
            roi_parameters = (
                request.roi_parameters_by_panel[panel_index]
                if panel_index < len(request.roi_parameters_by_panel)
                else request.parameters
            )
            panel_stages = self._stages_for_roi_parameters(prefix_stages, roi_parameters)
            sequence_key = panel._sequence_key(panel_stages, backend_id, request.noise_sigma, request.parameters)
            frame_key = panel._frame_sequence_key(sequence_key)
            frames = panel.stage_frame_cache.get(frame_key) if frame_key else panel.source_gray_frames
            roi = panel._roi_selection_for_sequence(sequence_key)
            if not frames or roi is None:
                raise ValueError(f"ROI / needle level alignment could not obtain an ROI for {panel.label}.")
            roi_mask = roi_selection_mask(roi, frames[0].shape)
            roi_level, needle_level, needle_mask = measure_roi_needle_baselines(
                frames,
                panel.info.fps,
                roi_mask,
                getattr(panel, "camera_view_mask", None),
            )
            anchors.append((roi_level, needle_level))
            roi_curves.append(masked_average_brightness(frames, roi_mask))
            needle_masks.append(needle_mask)
            alignment_frames.append(frames)

        baseline_ranges = [abs(roi_level - needle_level) for roi_level, needle_level in anchors]
        reference_index = int(np.argmax(baseline_ranges))
        adjusted_index = 1 - reference_index
        target_roi_level, target_needle_level = anchors[reference_index]
        source_roi_level, source_needle_level = anchors[adjusted_index]
        gain, offset = intensity_alignment_parameters(
            source_roi_level,
            source_needle_level,
            target_roi_level,
            target_needle_level,
        )
        if gain < 1.0:
            raise RuntimeError("ROI / needle level alignment selected the wrong baseline range reference.")
        self.panels[adjusted_index].roi_needle_alignment_gain = gain
        self.panels[adjusted_index].roi_needle_alignment_offset = offset

        results: dict[str, ROIAlignmentPlotResult] = {}
        for panel_index, panel in enumerate(self.panels):
            panel_gain = float(panel.roi_needle_alignment_gain)
            panel_offset = float(panel.roi_needle_alignment_offset)
            frames = alignment_frames[panel_index]
            roi_before = roi_curves[panel_index]
            roi_parameters = (
                request.roi_parameters_by_panel[panel_index]
                if panel_index < len(request.roi_parameters_by_panel)
                else request.parameters
            )
            panel_stages = self._stages_for_roi_parameters(prefix_stages, roi_parameters)
            sequence_key = panel._sequence_key(panel_stages, backend_id, request.noise_sigma, request.parameters)
            roi = panel._roi_selection_for_sequence(sequence_key)
            assert roi is not None
            roi_mask = roi_selection_mask(roi, frames[0].shape)
            roi_after = masked_affine_average_brightness(
                frames,
                roi_mask,
                panel_gain,
                panel_offset,
            )
            needle_after = masked_affine_average_brightness(
                frames,
                needle_core_mask(needle_masks[panel_index]),
                panel_gain,
                panel_offset,
            )
            baseline_count = min(
                len(frames),
                max(
                    1,
                    detect_pre_injection_trim_start(
                        frames,
                        panel.info.fps,
                        camera_view_mask=getattr(panel, "camera_view_mask", None),
                    )
                    + max(1, round(float(panel.info.fps) * 0.5)),
                ),
            )
            roi_before_level, needle_before_level = anchors[panel_index]
            results[panel.label] = ROIAlignmentPlotResult(
                time=np.arange(len(frames), dtype=float) / panel.info.fps,
                roi_brightness_before=roi_before,
                roi_brightness_after=roi_after,
                roi_baseline_before=roi_before_level,
                needle_baseline_before=needle_before_level,
                roi_baseline_after=float(np.mean(roi_after[:baseline_count])),
                needle_baseline_after=float(np.mean(needle_after[:baseline_count])),
            )
        self.roi_needle_alignment_results = results

    def _on_manual_roi_drawn(self, panel_index: int, circle: object) -> None:
        if (
            panel_index >= len(self._manual_roi_circles)
            or not isinstance(circle, tuple)
            or len(circle) != 3
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in circle)
        ):
            return
        self._manual_roi_circles[panel_index] = circle
        mode_combo = self.roi_mode_combos[min(panel_index, len(self.roi_mode_combos) - 1)]
        mode_combo.setCurrentIndex(mode_combo.findData("manual"))
        self.statusBar().showMessage(f"{self.panels[panel_index].label} ROI set to manual selection.")

    def _parent_vessel_roi_for_panel(self, panel_index: int) -> tuple[int, int, float] | None:
        if panel_index >= len(self._parent_vessel_rois):
            return None
        roi = self._parent_vessel_rois[panel_index]
        if roi is None:
            return None
        rotation = self.parent_vessel_rotation_spins[min(panel_index, len(self.parent_vessel_rotation_spins) - 1)].value()
        return roi[0], roi[1], rotation

    def _update_parent_vessel_roi_display(self, panel_index: int) -> None:
        if panel_index < len(self.panels):
            self.panels[panel_index].display.set_parent_vessel_roi(self._parent_vessel_roi_for_panel(panel_index))

    def _on_parent_vessel_roi_drawn(self, panel_index: int, center: object) -> None:
        if (
            not self._has_enabled_stage("parent_vessel_roi_analysis")
            or panel_index >= len(self._parent_vessel_rois)
            or not isinstance(center, tuple)
            or len(center) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in center)
        ):
            return
        center_x, center_y = center
        rotation = self.parent_vessel_rotation_spins[min(panel_index, len(self.parent_vessel_rotation_spins) - 1)].value()
        self._parent_vessel_rois[panel_index] = (center_x, center_y, rotation)
        self._update_parent_vessel_roi_display(panel_index)
        self.statusBar().showMessage(f"{self.panels[panel_index].label} parent vessel ROI set.")
        if not self._background_pipeline_work_active():
            self.run_parent_vessel_roi_analysis()

    def _on_parent_vessel_rotation_changed(self, panel_index: int, rotation: float) -> None:
        if panel_index < len(self._parent_vessel_rois):
            roi = self._parent_vessel_rois[panel_index]
            if roi is not None:
                self._parent_vessel_rois[panel_index] = (roi[0], roi[1], rotation)
        self._update_parent_vessel_roi_display(panel_index)
        if (
            self._has_enabled_stage("parent_vessel_roi_analysis")
            and self._parent_vessel_roi_for_panel(panel_index) is not None
            and not self._background_pipeline_work_active()
        ):
            self.run_parent_vessel_roi_analysis()

    def _background_roi_for_panel(self, panel_index: int) -> tuple[int, int] | None:
        return self._background_rois[panel_index] if panel_index < len(self._background_rois) else None

    def _update_background_roi_display(self, panel_index: int) -> None:
        if panel_index < len(self.panels):
            self.panels[panel_index].display.set_background_roi(self._background_roi_for_panel(panel_index))

    def _on_background_roi_drawn(self, panel_index: int, center: object) -> None:
        if (
            not self._has_enabled_stage("background_roi_analysis")
            or panel_index >= len(self._background_rois)
            or not isinstance(center, tuple)
            or len(center) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in center)
        ):
            return
        self._background_rois[panel_index] = center
        self._update_background_roi_display(panel_index)
        self.statusBar().showMessage(f"{self.panels[panel_index].label} background ROI set.")
        if not self._background_pipeline_work_active():
            self.run_background_roi_analysis()

    def _roi_settings_data(self) -> list[dict[str, object]]:
        settings: list[dict[str, object]] = []
        for index, mode_combo in enumerate(self.roi_mode_combos):
            circle = self._manual_roi_circles[index] if index < len(self._manual_roi_circles) else None
            settings.append(
                {
                    "mode": str(mode_combo.currentData()),
                    "manual_circle": None if circle is None else list(circle),
                }
            )
        return settings

    def _parent_vessel_roi_settings_data(self) -> list[dict[str, object]]:
        settings: list[dict[str, object]] = []
        for index in range(len(self.parent_vessel_rotation_spins)):
            roi = self._parent_vessel_roi_for_panel(index)
            settings.append(
                {
                    "center": None if roi is None else [roi[0], roi[1]],
                    "rotation": self.parent_vessel_rotation_spins[index].value(),
                }
            )
        return settings

    def _background_roi_settings_data(self) -> list[dict[str, object]]:
        return [
            {"center": None if roi is None else list(roi)}
            for roi in (self._background_roi_for_panel(index) for index in range(len(self.panels)))
        ]

    def _apply_roi_settings_data(self, settings: object) -> None:
        if not isinstance(settings, list):
            return
        for index, value in enumerate(settings[: len(self.roi_mode_combos)]):
            if not isinstance(value, dict):
                continue
            mode = value.get("mode")
            if mode in {"auto", "manual"}:
                combo = self.roi_mode_combos[index]
                combo.blockSignals(True)
                try:
                    combo.setCurrentIndex(combo.findData(mode))
                finally:
                    combo.blockSignals(False)
            circle_values = value.get("manual_circle")
            if (
                isinstance(circle_values, list)
                and len(circle_values) == 3
                and all(isinstance(item, int) and not isinstance(item, bool) for item in circle_values)
            ):
                center_x, center_y, radius = circle_values
                self._manual_roi_circles[index] = (center_x, center_y, radius) if radius >= 2 else None
                continue
            rect_values = value.get("manual_rect")
            if (
                isinstance(rect_values, list)
                and len(rect_values) == 4
                and all(isinstance(item, int) and not isinstance(item, bool) for item in rect_values)
            ):
                rect = QRect(*rect_values).normalized()
                radius = min(rect.width(), rect.height()) // 2
                self._manual_roi_circles[index] = (rect.center().x(), rect.center().y(), radius) if radius >= 2 else None

    def _apply_parent_vessel_roi_settings_data(self, settings: object) -> None:
        if not isinstance(settings, list):
            return
        for index, value in enumerate(settings[: len(self.parent_vessel_rotation_spins)]):
            if not isinstance(value, dict):
                continue
            rotation = value.get("rotation")
            if isinstance(rotation, (int, float)) and not isinstance(rotation, bool):
                spin = self.parent_vessel_rotation_spins[index]
                spin.blockSignals(True)
                try:
                    spin.setValue(float(rotation))
                finally:
                    spin.blockSignals(False)
            center = value.get("center")
            if (
                isinstance(center, list)
                and len(center) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) for item in center)
                and index < len(self._parent_vessel_rois)
            ):
                self._parent_vessel_rois[index] = (center[0], center[1], self.parent_vessel_rotation_spins[index].value())
                self._update_parent_vessel_roi_display(index)

    def _apply_background_roi_settings_data(self, settings: object) -> None:
        if not isinstance(settings, list):
            return
        for index, value in enumerate(settings[: len(self._background_rois)]):
            center = value.get("center") if isinstance(value, dict) else None
            if (
                isinstance(center, list)
                and len(center) == 2
                and all(isinstance(item, int) and not isinstance(item, bool) for item in center)
            ):
                self._background_rois[index] = (center[0], center[1])
                self._update_background_roi_display(index)

    def _on_background_subtraction_stage_toggled(self, enabled: bool) -> None:
        if getattr(self, "_syncing_background_subtraction", False):
            return
        self._syncing_background_subtraction = True
        try:
            for check in (self.background_video_one_check, self.background_video_two_check):
                check.setChecked(enabled)
        finally:
            self._syncing_background_subtraction = False
        self.on_enhancement_settings_changed()

    def _on_background_subtraction_video_toggled(self, _enabled: bool) -> None:
        if getattr(self, "_syncing_background_subtraction", False):
            return
        self._syncing_background_subtraction = True
        try:
            self.background_subtraction_stage_check.setChecked(
                self.background_video_one_check.isChecked() or self.background_video_two_check.isChecked()
            )
        finally:
            self._syncing_background_subtraction = False
        self.on_enhancement_settings_changed()

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
            roi_softening_enabled=self.roi_soften_check.isChecked(),
            roi_softening_radius_ratio=self.roi_radius_spin.value(),
            roi_softening_threshold=self.roi_threshold_spin.value(),
            roi_convex_hull_enabled=self.roi_convex_hull_check.isChecked(),
            roi_circle_fit_enabled=self.roi_circle_fit_check.isChecked(),
            scanline_bias_clip=self.scanline_clip_spin.value(),
            scanline_sigma_y=self.scanline_sigma_spin.value(),
            temporal_motion_sigma=self.temporal_sigma_spin.value(),
            mottle_similarity_sigma=self.mottle_similarity_sigma_spin.value(),
            mottle_window_radius=int(self.mottle_window_combo.currentData()),
            clahe_clip_limit=self.clahe_clip_spin.value(),
            clahe_tile_size=self.clahe_tile_spin.value(),
            adjustments_brightness_offset=self.adjustments_brightness_spin.value(),
            adjustments_contrast_gain=self.adjustments_contrast_spin.value(),
            adjustments_sharpen_amount=self.adjustments_sharpen_spin.value(),
            adjustments_gamma=self.adjustments_gamma_spin.value(),
            smoothing_sigma_x=self.smoothing_sigma_spin.value(),
        )

    def on_pipeline_stages_changed(self) -> None:
        if self._loading_config:
            return
        stages = self.enhancement_stages()
        active_mode = str(self.enhancement_mode_combo.currentData())
        uses_ffdnet = stages.denoise and active_mode.startswith("ffdnet")
        uses_accelerator = uses_ffdnet or (stages.denoise and active_mode == "tensor-nlm-ngc")
        uses_spatial_denoiser = stages.denoise
        needle_enabled = self._has_enabled_stage("needle_segmentation")
        parent_vessel_enabled = self._has_enabled_stage("parent_vessel_roi_analysis")
        background_roi_enabled = self._has_enabled_stage("background_roi_analysis")
        self.overlay_mask_check.setEnabled(bool(self.panels) and (stages.roi_extraction or needle_enabled))
        if not needle_enabled:
            self.needle_brightness_results.clear()
            self.needle_brightness_baselines.clear()
            self.needle_brightness_plot.clear()
            for panel in self.panels:
                panel.set_needle_segmentation_mask(None, self.current_frame_index)
        if not parent_vessel_enabled:
            self.parent_vessel_roi_results.clear()
            self.parent_vessel_roi_plot.clear()
            self.analysis_tabs.setTabVisible(self.parent_vessel_roi_tab_index, False)
            for panel in self.panels:
                panel.display.set_parent_vessel_roi(None)
        else:
            for panel_index in range(len(self.panels)):
                self._update_parent_vessel_roi_display(panel_index)
        if not background_roi_enabled:
            self.background_roi_results.clear()
            self.background_roi_plot.clear()
            self.analysis_tabs.setTabVisible(self.background_roi_tab_index, False)
            for panel in self.panels:
                panel.display.set_background_roi(None)
        else:
            for panel_index in range(len(self.panels)):
                self._update_background_roi_display(panel_index)
        self.denoise_strength_label.setEnabled(uses_spatial_denoiser)
        self.denoise_strength_spin.setEnabled(uses_spatial_denoiser)
        self.inference_batch_spin.setEnabled(uses_accelerator)
        self.inference_precision_combo.setEnabled(uses_accelerator)
        self._sync_active_denoise_controls()
        self._refresh_desktop_stream_pipeline()
        if self.active_mode == MODE_LIVE:
            self._apply_source_pipeline_stages()
            self._render_live_frame()
        elif self._pipeline_has_active_stage():
            self.rebuild_enhancement_pipeline()
        else:
            self._stop_enhancement_preview()
            self.set_display_enhancement(False)
            self.results.clear()
            self.clear_plots_and_metrics()
            self._update_stage_statuses()
            self.statusBar().showMessage("Showing original videos.")

    def reset_enhancement_pipeline(self) -> None:
        for check in self.pipeline_stage_checks:
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
        self.on_pipeline_stages_changed()

    def _denoiser_key(self, mode: str) -> str:
        if mode == "non-local-means":
            return mode
        precision = str(self.inference_precision_combo.currentData())
        batch_size = self.inference_batch_spin.value()
        return f"{mode}:{precision}:batch{batch_size}"

    def rebuild_enhancement_pipeline(self) -> None:
        if self.active_mode == MODE_LIVE:
            self._apply_source_pipeline_stages()
            self._render_live_frame()
            return
        mode = str(self.enhancement_mode_combo.currentData())
        auto_crop = self._has_enabled_stage("auto_crop")
        startup_stabilization_trim = self._has_enabled_stage("startup_stabilization_trim")
        temporal_gain_normalization = self.temporal_gain_normalization_check.isChecked()
        temporal_alignment = self._has_enabled_stage("temporal_alignment")
        contrast_gain_alignment = self._has_enabled_stage("contrast_gain_alignment")
        histogram_matching = self.active_mode == MODE_COMPARISON and self._has_enabled_stage("histogram_matching")
        metal_needle_removal = self._has_enabled_stage("metal_needle_removal")
        auto_crop_size_offset = self.auto_crop_size_offset_spin.value()
        temporal_trim_offset_seconds = self.temporal_trim_offset_spin.value()
        temporal_end_trim_seconds = self.temporal_end_trim_spin.value()
        comparison_sync_offset_seconds = self.comparison_sync_offset_spin.value()
        background_subtraction_settings = self._background_subtraction_settings()
        self._enhancement_generation += 1
        request = EnhancementRequest(
            generation=self._enhancement_generation,
            mode=mode,
            model_label=self.enhancement_mode_combo.currentText().split(" (")[0],
            stages=self.enhancement_stages(),
            parameters=self.enhancement_parameters(),
            noise_sigma=self.denoise_strength_spin.value(),
            batch_size=self.inference_batch_spin.value(),
            precision=str(self.inference_precision_combo.currentData()),
            auto_crop=auto_crop,
            startup_stabilization_trim=startup_stabilization_trim,
            temporal_gain_normalization=temporal_gain_normalization,
            temporal_alignment=temporal_alignment,
            contrast_gain_alignment=contrast_gain_alignment,
            histogram_matching=histogram_matching,
            source_pipeline_current=self._source_pipeline_is_current(
                auto_crop,
                startup_stabilization_trim,
                temporal_gain_normalization,
                temporal_alignment,
                contrast_gain_alignment,
                histogram_matching,
                auto_crop_size_offset,
                temporal_trim_offset_seconds,
                temporal_end_trim_seconds,
                comparison_sync_offset_seconds,
                background_subtraction_settings,
                metal_needle_removal,
            ),
            auto_crop_size_offset=auto_crop_size_offset,
            temporal_trim_offset_seconds=temporal_trim_offset_seconds,
            temporal_end_trim_seconds=temporal_end_trim_seconds,
            comparison_sync_offset_seconds=comparison_sync_offset_seconds,
            background_subtraction_settings=background_subtraction_settings,
            metal_needle_removal=metal_needle_removal,
            prepare_source_frames=self._has_enabled_stage("frame_brightness_analysis"),
            roi_parameters_by_panel=tuple(self._roi_parameters_for_panel(index) for index in range(len(self.panels))),
        )
        self._enhancement_pending_request = request
        if self._analysis_cancel is not None:
            self._analysis_cancel.set()
            self.enhancement_progress.begin("Updating enhancement settings...")
        elif self._enhancement_cancel is not None:
            self._enhancement_cancel.set()
            self.enhancement_progress.begin("Updating enhancement settings...")
        else:
            self._start_enhancement_request(request)

    def _start_enhancement_request(self, request: EnhancementRequest) -> None:
        if not self.panels:
            self._update_stage_statuses()
            self.statusBar().showMessage("Load a video before running pipeline stages.")
            return
        self.roi_needle_alignment_results.clear()
        self.roi_needle_alignment_plot.clear()
        self.analysis_tabs.setTabVisible(self.roi_needle_alignment_tab_index, False)
        self._enhancement_pending_request = None
        self._enhancement_active_request = request
        self._enhancement_cancel = Event()
        with self._enhancement_progress_lock:
            self._enhancement_progress_values = [0.0, 0.0]
            self._enhancement_progress_totals = [1.0, 1.0]
            self._enhancement_stage_messages = ["Waiting", "Waiting"]
            self._enhancement_message = "Preparing enhanced videos..."
        for panel in self.panels:
            panel.enhanced_frames = []
            panel.roi_region_masks = [] if request.stages.roi_extraction else None
            panel.enhance_display = True
        self._set_playback_limit(0)
        for panel in self.panels:
            panel.seek(self.current_frame_index)
        self.open_pre_action.setEnabled(False)
        self.open_post_action.setEnabled(False)
        self.open_single_mode_action.setEnabled(False)
        self.open_comparison_mode_action.setEnabled(False)
        self.load_config_action.setEnabled(False)
        self.enhancement_progress.begin("Preparing enhanced videos...")
        self.statusBar().showMessage("Enhancement is running; playback follows the frames ready in both videos.")
        self._update_stage_statuses()
        self._enhancement_future = self._enhancement_executor.submit(
            self._run_enhancement_request,
            request,
            self._enhancement_cancel,
        )
        self.enhancement_poll_timer.start()

    def _run_enhancement_request(self, request: EnhancementRequest, cancel_event: Event) -> bool:
        if not self.panels:
            return request.source_pipeline_current and not request.stages.any_enabled
        source_stage_count = 0
        def calculate_source_panel(panel_index: int) -> SourcePipelineState:
            panel = self.panels[panel_index]

            def source_progress(stage_message: str, done: float, total: float) -> bool:
                with self._enhancement_progress_lock:
                    self._enhancement_stage_messages[panel_index] = stage_message
                    self._enhancement_progress_values[panel_index] = done
                    self._enhancement_progress_totals[panel_index] = max(total, 0.001)
                    self._enhancement_message = "Preparing source pipeline..."
                return not cancel_event.is_set()

            return panel.calculate_source_pipeline(
                request.auto_crop,
                request.temporal_alignment,
                source_progress,
                request.auto_crop_size_offset,
                request.temporal_trim_offset_seconds,
                request.temporal_end_trim_seconds,
                request.comparison_sync_offset_seconds
                if self.active_mode == MODE_COMPARISON and panel_index == 1
                else 0.0,
                request.contrast_gain_alignment,
                request.histogram_matching,
                request.background_subtraction_settings[panel_index][0],
                request.background_subtraction_settings[panel_index][1],
                request.metal_needle_removal,
                request.startup_stabilization_trim,
                request.temporal_gain_normalization,
            )

        if not request.source_pipeline_current:
            source_stage_count = (
                int(request.auto_crop)
                + int(request.startup_stabilization_trim)
                + int(request.temporal_alignment)
                + int(request.metal_needle_removal)
            )
            source_states: list[SourcePipelineState | None] = [None] * len(self.panels)
            with frame_parallel_opencv(), ThreadPoolExecutor(
                max_workers=len(self.panels),
                thread_name_prefix="source-pipeline",
            ) as executor:
                futures = {
                    executor.submit(calculate_source_panel, panel_index): panel_index
                    for panel_index in range(len(self.panels))
                }
                for future in as_completed(futures):
                    source_states[futures[future]] = future.result()
                    if cancel_event.is_set():
                        return False

            resolved_source_states = [state for state in source_states if state is not None]
            if request.contrast_gain_alignment:
                resolved_source_states = align_source_gain_states(resolved_source_states)
            if request.histogram_matching and self.active_mode == MODE_COMPARISON:
                resolved_source_states = align_source_histogram_states(resolved_source_states)
            if request.temporal_alignment and self.active_mode == MODE_COMPARISON:
                resolved_source_states = align_source_brightness_onsets(resolved_source_states)

            source_states_applied = Event()
            self._source_pipeline_events.put(
                (request.generation, resolved_source_states, source_states_applied)
            )
            while not source_states_applied.wait(0.05):
                if cancel_event.is_set():
                    return False
            if cancel_event.is_set():
                return False
        if not request.stages.any_enabled and not request.prepare_source_frames:
            return True

        use_denoiser = request.stages.denoise
        denoiser_base_key = ""
        if use_denoiser:
            denoiser_base_key = (
                request.mode
                if request.mode == "non-local-means"
                else f"{request.mode}:{request.precision}:batch{request.batch_size}"
            )
        denoiser_count = len(self.panels) if request.mode.endswith("-ngc") and use_denoiser else int(use_denoiser)
        active_denoiser_keys = {
            f"{denoiser_base_key}:worker{worker_index}"
            for worker_index in range(denoiser_count)
        }
        for key, inactive_denoiser in list(self.deep_denoisers.items()):
            if key in active_denoiser_keys:
                continue
            close = getattr(inactive_denoiser, "close", None)
            if close is not None:
                close()
            del self.deep_denoisers[key]

        denoisers: list[FrameDenoiser] = []
        if use_denoiser:
            for worker_index in range(denoiser_count):
                denoiser_key = f"{denoiser_base_key}:worker{worker_index}"
                if denoiser_key not in self.deep_denoisers:
                    with self._enhancement_progress_lock:
                        worker_label = f" worker {worker_index + 1}/{denoiser_count}" if denoiser_count > 1 else ""
                        self._enhancement_message = f"Loading {request.model_label}{worker_label}..."
                    if request.mode == "non-local-means":
                        self.deep_denoisers[denoiser_key] = NonLocalMeansDenoiser()
                    elif request.mode == "tensor-nlm-ngc":
                        from container_denoiser import ContainerDenoiser

                        self.deep_denoisers[denoiser_key] = ContainerDenoiser(
                            "tensor-nlm",
                            None,
                            request.batch_size,
                            request.precision,
                        )
                    elif request.mode.endswith("-ngc"):
                        from container_denoiser import ContainerDenoiser

                        self.deep_denoisers[denoiser_key] = ContainerDenoiser(
                            "ffdnet",
                            ROOT / "models" / "ffdnet_gray.pth",
                            request.batch_size,
                            request.precision,
                        )
                    elif request.mode == "ffdnet-native":
                        from deep_denoiser import FFDNetDenoiser

                        self.deep_denoisers[denoiser_key] = FFDNetDenoiser(
                            ROOT / "models" / "ffdnet_gray.pth",
                            request.precision,
                        )
                    else:
                        raise ValueError(f"Unsupported denoiser mode: {request.mode}")
                denoisers.append(self.deep_denoisers[denoiser_key])

        if cancel_event.is_set():
            return False
        backend_id = denoisers[0].backend_id if denoisers else "none"
        if len(denoisers) == 1:
            panel_denoisers: list[FrameDenoiser | None] = [SynchronizedFrameDenoiser(denoisers[0])] * len(self.panels)
        elif denoisers:
            panel_denoisers = list(denoisers)
        else:
            panel_denoisers = [None] * len(self.panels)

        self._prepare_roi_needle_level_alignment(
            request,
            panel_denoisers,
            backend_id,
            cancel_event,
        )
        if cancel_event.is_set():
            return False

        panel_work = [
            panel.estimate_prepare_work(
                backend_id,
                request.noise_sigma,
                self._stages_for_roi_parameters(request.stages, request.roi_parameters_by_panel[index]),
                request.parameters,
            )
            for index, panel in enumerate(self.panels)
        ]
        with self._enhancement_progress_lock:
            self._enhancement_progress_totals = [source_stage_count + max(work, 0.001) for work in panel_work]
            if denoisers:
                worker_label = f" with {len(denoisers)} accelerator workers" if len(denoisers) > 1 else ""
                self._enhancement_message = (
                    f"Running {request.model_label} on {denoisers[0].device_name}{worker_label}..."
                )
            else:
                self._enhancement_message = "Running video enhancement..."

        def prepare_panel(panel_index: int, frame_executor: AdaptiveFrameExecutor) -> bool:
            panel = self.panels[panel_index]

            def progress_callback(done: float, panel_total: float) -> bool:
                with self._enhancement_progress_lock:
                    self._enhancement_progress_values[panel_index] = source_stage_count + done
                    self._enhancement_progress_totals[panel_index] = source_stage_count + panel_total
                return not cancel_event.is_set()

            def stage_progress_callback(stage_message: str, done: int, total: int) -> bool:
                with self._enhancement_progress_lock:
                    self._enhancement_stage_messages[panel_index] = stage_message
                return not cancel_event.is_set()

            def encoded_frame_callback(frame_index: int, encoded: np.ndarray) -> None:
                self._enhancement_frame_events.put((request.generation, panel_index, frame_index, encoded))

            def roi_region_mask_callback(frame_index: int, encoded: np.ndarray) -> None:
                self._roi_region_mask_events.put((request.generation, panel_index, frame_index, encoded))

            prepared = panel.prepare_enhanced_frames(
                panel_denoisers[panel_index],
                request.noise_sigma,
                request.batch_size,
                self._stages_for_roi_parameters(request.stages, request.roi_parameters_by_panel[panel_index]),
                request.parameters,
                progress_callback,
                stage_progress_callback,
                encoded_frame_callback,
                roi_region_mask_callback,
                False,
                cancel_event.is_set,
                frame_executor,
            )
            if prepared:
                with self._enhancement_progress_lock:
                    self._enhancement_progress_values[panel_index] = self._enhancement_progress_totals[panel_index]
            return prepared

        prepared = [False] * len(self.panels)
        with frame_parallel_opencv(), AdaptiveFrameExecutor() as frame_executor, ThreadPoolExecutor(
            max_workers=len(self.panels),
            thread_name_prefix="enhancement",
        ) as executor:
            future_indices = {
                executor.submit(prepare_panel, panel_index, frame_executor): panel_index
                for panel_index in range(len(self.panels))
            }
            for future in as_completed(future_indices):
                panel_index = future_indices[future]
                try:
                    prepared[panel_index] = future.result()
                except Exception:
                    cancel_event.set()
                    raise
                if not prepared[panel_index]:
                    cancel_event.set()
        return all(prepared)

    def _analysis_heatmap_colormap(self) -> str:
        active_drawer = next(
            (
                drawer
                for drawer in self._stage_drawers("temporal_change_heatmap")
                if drawer.enable_button.isChecked()
            ),
            None,
        )
        combo = active_drawer.findChild(QComboBox, "heatmapColormap") if active_drawer is not None else None
        return str(combo.currentData()) if combo is not None else "hot"

    def _build_pipeline_analysis_request(
        self,
        *,
        roi_residence: bool,
        frame_brightness: bool,
        needle_segmentation: bool,
        parent_vessel_roi: bool,
        background_roi: bool,
        temporal_change: bool,
    ) -> PipelineAnalysisRequest | None:
        stages = self.enhancement_stages()
        parameters = self.enhancement_parameters()
        backend_id = self._current_backend_id(stages)
        snapshots: list[AnalysisPanelSnapshot] = []
        for panel_index, panel in enumerate(self.panels):
            panel_stages = self._stages_for_roi_parameters(
                stages,
                self._roi_parameters_for_panel(panel_index),
            )
            encoded_frames = panel.encoded_analysis_frames(
                backend_id,
                self.denoise_strength_spin.value(),
                panel_stages,
                parameters,
            )
            sequence_key = panel._sequence_key(
                panel_stages,
                backend_id,
                self.denoise_strength_spin.value(),
                parameters,
            )
            if encoded_frames is None and panel._frame_sequence_key(sequence_key):
                return None
            source_frames = panel.source_gray_frames
            if encoded_frames is None and source_frames is None:
                return None
            if frame_brightness and source_frames is None:
                return None
            roi = panel.roi()
            roi_mask = panel.roi_mask()
            snapshots.append(
                AnalysisPanelSnapshot(
                    label=panel.label,
                    path=panel.path,
                    fps=panel.info.fps,
                    roi=None if roi is None else QRect(roi),
                    roi_mask=None if roi_mask is None else roi_mask.copy(),
                    parent_vessel_roi=self._parent_vessel_roi_for_panel(panel_index),
                    background_roi=self._background_roi_for_panel(panel_index),
                    camera_view_mask=(
                        None
                        if getattr(panel, "camera_view_mask", None) is None
                        else panel.camera_view_mask.copy()
                    ),
                    encoded_frames=None if encoded_frames is None else tuple(encoded_frames),
                    source_frames=None if source_frames is None else tuple(source_frames),
                )
            )

        return PipelineAnalysisRequest(
            generation=self._enhancement_generation,
            panels=tuple(snapshots),
            threshold=self.threshold_spin.value(),
            gain_corrected=(
                stages.brightness_stabilization
                or stages.roi_needle_level_alignment
                or stages.gain_stabilization
                or self._has_enabled_stage("contrast_gain_alignment")
            ),
            roi_residence=roi_residence,
            frame_brightness=frame_brightness,
            needle_segmentation=needle_segmentation,
            parent_vessel_roi=parent_vessel_roi,
            background_roi=background_roi,
            temporal_change=temporal_change,
            heatmap_colormap=self._analysis_heatmap_colormap(),
        )

    def _start_pipeline_analysis(
        self,
        *,
        roi_residence: bool,
        frame_brightness: bool,
        needle_segmentation: bool,
        parent_vessel_roi: bool,
        background_roi: bool,
        temporal_change: bool,
    ) -> bool:
        if not self.panels or self._enhancement_future is not None or self._analysis_future is not None:
            return False
        if roi_residence and self._analysis_requirement_failure() is not None:
            roi_residence = False
        if parent_vessel_roi and any(
            self._parent_vessel_roi_for_panel(index) is None
            for index in range(len(self.panels))
        ):
            parent_vessel_roi = False
        if background_roi and any(
            self._background_roi_for_panel(index) is None
            for index in range(len(self.panels))
        ):
            background_roi = False
        if not any((roi_residence, frame_brightness, needle_segmentation, parent_vessel_roi, background_roi, temporal_change)):
            return False
        request = self._build_pipeline_analysis_request(
            roi_residence=roi_residence,
            frame_brightness=frame_brightness,
            needle_segmentation=needle_segmentation,
            parent_vessel_roi=parent_vessel_roi,
            background_roi=background_roi,
            temporal_change=temporal_change,
        )
        if request is None:
            return False

        self.pause()
        self._enhancement_pending_request = None
        self._analysis_active_request = request
        self._analysis_cancel = Event()
        values: list[float] = []
        totals: list[float] = []
        for panel in request.panels:
            frame_count = len(panel.encoded_frames or panel.source_frames or ())
            decode_passes = int(panel.encoded_frames is not None)
            analysis_passes = (
                int(request.roi_residence)
                + int(request.frame_brightness)
                + int(request.needle_segmentation)
                + int(request.parent_vessel_roi)
                + int(request.background_roi)
                + int(request.temporal_change) * 2
            )
            values.append(0.0)
            totals.append(float(max(1, frame_count * (decode_passes + analysis_passes))))
        with self._enhancement_progress_lock:
            self._enhancement_progress_values = values
            self._enhancement_progress_totals = totals
            self._enhancement_stage_messages = ["Preparing analysis"] * len(request.panels)
            self._enhancement_message = "Running video analysis..."
        self.open_pre_action.setEnabled(False)
        self.open_post_action.setEnabled(False)
        self.open_single_mode_action.setEnabled(False)
        self.open_comparison_mode_action.setEnabled(False)
        self.load_config_action.setEnabled(False)
        self.enhancement_progress.begin("Running video analysis...")
        self._analysis_future = self._analysis_executor.submit(
            self._run_pipeline_analysis,
            request,
            self._analysis_cancel,
        )
        self.enhancement_poll_timer.start()
        self._update_stage_statuses()
        return True

    def _run_pipeline_analysis(
        self,
        request: PipelineAnalysisRequest,
        cancel_event: Event,
    ) -> PipelineAnalysisResult:
        panel_outputs: dict[
            str,
            tuple[
                AnalysisResult | None,
                tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
                float | None,
                np.ndarray | None,
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
            ],
        ] = {}

        def process_panel(
            panel_index: int,
            panel: AnalysisPanelSnapshot,
        ) -> tuple[
            AnalysisResult | None,
            tuple[np.ndarray, np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
            float | None,
            np.ndarray | None,
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]:
            done = 0.0
            frame_count = len(panel.encoded_frames or panel.source_frames or ())

            def publish(stage: str, completed: float) -> None:
                with self._enhancement_progress_lock:
                    self._enhancement_stage_messages[panel_index] = stage
                    self._enhancement_progress_values[panel_index] = completed
                    self._enhancement_message = "Running video analysis..."

            if panel.encoded_frames is not None:
                gray_frames: list[np.ndarray] = []
                publish("Decoding enhanced frames", done)
                for encoded in panel.encoded_frames:
                    if cancel_event.is_set():
                        raise RuntimeError("Analysis cancelled")
                    frame = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                    if frame is None:
                        raise ValueError(f"Could not decode an enhanced frame for {panel.label}.")
                    gray_frames.append(frame)
                    done += 1.0
                    publish("Decoding enhanced frames", done)
            else:
                gray_frames = list(panel.source_frames or ())

            roi_result: AnalysisResult | None = None
            brightness_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            needle_result: tuple[np.ndarray, np.ndarray] | None = None
            needle_baseline: float | None = None
            needle_mask: np.ndarray | None = None
            parent_vessel_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            background_roi_result: tuple[np.ndarray, np.ndarray] | None = None
            temporal_result: tuple[np.ndarray, np.ndarray] | None = None

            if request.roi_residence:
                if panel.roi is None or panel.roi_mask is None:
                    raise ValueError(f"ROI masks need review for {panel.label}.")
                publish("Analyzing ROI residence", done)
                roi_result = analyze_gray_frames(
                    panel.label,
                    panel.path,
                    panel.fps,
                    panel.roi,
                    panel.roi_mask,
                    gray_frames,
                    request.threshold,
                    request.gain_corrected,
                    panel.camera_view_mask,
                )
                done += frame_count
                publish("Analyzing ROI residence", done)

            if request.frame_brightness:
                publish("Measuring frame brightness", done)
                source_frames = list(panel.source_frames or ())
                brightness_count = min(len(source_frames), len(gray_frames))
                if brightness_count == 0:
                    raise ValueError(f"No frames are available for brightness analysis of {panel.label}.")
                brightness_result = (
                    np.arange(brightness_count, dtype=float) / panel.fps,
                    average_frame_brightness(source_frames[:brightness_count], panel.camera_view_mask),
                    average_frame_brightness(gray_frames[:brightness_count], panel.camera_view_mask),
                )
                done += frame_count
                publish("Measuring frame brightness", done)

            if request.needle_segmentation:
                publish("Segmenting needle", done)
                needle_mask = segment_pre_injection_needle(
                    gray_frames,
                    panel.fps,
                    panel.camera_view_mask,
                )
                if needle_mask is not None:
                    brightness = needle_average_brightness(gray_frames, needle_mask)
                    pre_injection_count = min(
                        len(brightness),
                        max(
                            1,
                            detect_pre_injection_trim_start(
                                gray_frames,
                                panel.fps,
                                camera_view_mask=panel.camera_view_mask,
                            )
                            + max(1, round(float(panel.fps) * 0.5)),
                        ),
                    )
                    needle_result = (
                        np.arange(len(gray_frames), dtype=float) / panel.fps,
                        brightness,
                    )
                    needle_baseline = float(np.mean(brightness[:pre_injection_count]))
                done += frame_count
                publish("Segmenting needle", done)

            if request.parent_vessel_roi:
                if panel.parent_vessel_roi is None:
                    raise ValueError(f"Place a parent vessel ROI box for {panel.label}.")
                publish("Measuring parent vessel darkness", done)
                center_x, center_y, rotation = panel.parent_vessel_roi
                darkest_median, average_brightness = parent_vessel_brightness_measurements(
                    gray_frames,
                    center_x,
                    center_y,
                    rotation,
                    panel.camera_view_mask,
                )
                parent_vessel_result = (
                    np.arange(len(gray_frames), dtype=float) / panel.fps,
                    darkest_median,
                    average_brightness,
                )
                done += frame_count
                publish("Measuring parent vessel darkness", done)

            if request.background_roi:
                if panel.background_roi is None:
                    raise ValueError(f"Place a background ROI box for {panel.label}.")
                publish("Measuring background brightness", done)
                center_x, center_y = panel.background_roi
                background_roi_result = (
                    np.arange(len(gray_frames), dtype=float) / panel.fps,
                    background_roi_average_brightness(
                        gray_frames, center_x, center_y, panel.camera_view_mask
                    ),
                )
                done += frame_count
                publish("Measuring background brightness", done)

            if request.temporal_change:
                publish("Computing contrast residence heatmap", done)
                temporal_result = compute_temporal_change_summary(
                    gray_frames,
                    panel.fps,
                    panel.camera_view_mask,
                )
                done += frame_count
                publish("Computing contrast residence heatmap", done)

            return (
                roi_result,
                brightness_result,
                needle_result,
                needle_baseline,
                needle_mask,
                parent_vessel_result,
                background_roi_result,
                temporal_result,
            )

        with ThreadPoolExecutor(
            max_workers=max(1, len(request.panels)),
            thread_name_prefix="analysis-panel",
        ) as executor:
            futures = {
                executor.submit(process_panel, panel_index, panel): panel
                for panel_index, panel in enumerate(request.panels)
            }
            for future in as_completed(futures):
                panel = futures[future]
                panel_outputs[panel.label] = future.result()

        roi_results = {
            label: output[0]
            for label, output in panel_outputs.items()
            if output[0] is not None
        }
        normalized_roi_results = normalize_analysis_results(
            cast(dict[str, AnalysisResult], roi_results),
            request.threshold,
        )
        frame_brightness_results = {
            label: output[1]
            for label, output in panel_outputs.items()
            if output[1] is not None
        }
        needle_brightness_results = {
            label: output[2]
            for label, output in panel_outputs.items()
            if output[2] is not None
        }
        needle_brightness_baselines = {
            label: output[3]
            for label, output in panel_outputs.items()
            if output[3] is not None
        }
        needle_masks = {
            label: output[4]
            for label, output in panel_outputs.items()
        } if request.needle_segmentation else {}
        parent_vessel_roi_results = {
            label: output[5]
            for label, output in panel_outputs.items()
            if output[5] is not None
        }
        background_roi_results = {
            label: output[6]
            for label, output in panel_outputs.items()
            if output[6] is not None
        }
        temporal_change_results = {
            label: output[7]
            for label, output in panel_outputs.items()
            if output[7] is not None
        }
        typed_temporal_results = cast(dict[str, tuple[np.ndarray, np.ndarray]], temporal_change_results)
        temporal_change_heatmaps: dict[str, np.ndarray] = {}
        if typed_temporal_results:
            shared_scale = len(typed_temporal_results) > 1
            burden_peak, contrast_peak = (
                temporal_change_heatmap_peaks(typed_temporal_results)
                if shared_scale
                else (None, None)
            )
            for panel_index, panel in enumerate(request.panels):
                summary = typed_temporal_results.get(panel.label)
                if summary is None:
                    continue
                with self._enhancement_progress_lock:
                    self._enhancement_stage_messages[panel_index] = "Rendering contrast residence heatmap"
                temporal_change_heatmaps[panel.label] = render_temporal_change_heatmap(
                    summary[0],
                    summary[1],
                    burden_peak,
                    contrast_peak,
                    colormap=request.heatmap_colormap,
                )
                with self._enhancement_progress_lock:
                    self._enhancement_progress_values[panel_index] += len(
                        panel.encoded_frames or panel.source_frames or ()
                    )

        return PipelineAnalysisResult(
            generation=request.generation,
            roi_results=normalized_roi_results,
            frame_brightness_results=cast(
                dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                frame_brightness_results,
            ),
            needle_brightness_results=cast(
                dict[str, tuple[np.ndarray, np.ndarray]],
                needle_brightness_results,
            ),
            needle_brightness_baselines=cast(dict[str, float], needle_brightness_baselines),
            needle_masks=needle_masks,
            parent_vessel_roi_results=cast(
                dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                parent_vessel_roi_results,
            ),
            background_roi_results=cast(dict[str, tuple[np.ndarray, np.ndarray]], background_roi_results),
            temporal_change_results=typed_temporal_results,
            temporal_change_heatmaps=temporal_change_heatmaps,
        )

    def _finish_background_pipeline_work(self) -> None:
        self._set_video_controls_enabled(bool(self.panels))
        self.open_single_mode_action.setEnabled(True)
        self.open_comparison_mode_action.setEnabled(True)
        self.load_config_action.setEnabled(True)
        if self._enhancement_future is None and self._analysis_future is None:
            self.enhancement_progress.finish()
            self.enhancement_poll_timer.stop()

    def _poll_analysis(self) -> None:
        request = self._analysis_active_request
        if request is not None and request.generation == self._enhancement_generation:
            with self._enhancement_progress_lock:
                values = list(self._enhancement_progress_values)
                totals = list(self._enhancement_progress_totals)
                stages = list(self._enhancement_stage_messages)
                message = self._enhancement_message
            self.enhancement_progress.message_label.setText(message)
            self.enhancement_progress.set_progress(sum(values), sum(totals))
            for panel_index in range(min(len(self.panels), len(values))):
                self.enhancement_progress.set_panel_progress(
                    panel_index,
                    stages[panel_index],
                    values[panel_index],
                    totals[panel_index],
                )

        future = self._analysis_future
        if future is None or not future.done():
            return
        completed_request = self._analysis_active_request
        self._analysis_future = None
        self._analysis_active_request = None
        self._analysis_cancel = None
        try:
            analysis_result = future.result()
            error: Exception | None = None
        except Exception as exc:
            analysis_result = None
            error = exc
            if completed_request is not None and completed_request.generation == self._enhancement_generation:
                LOGGER.exception("Analysis pipeline failed")

        if self._enhancement_pending_request is not None:
            self._start_enhancement_request(self._enhancement_pending_request)
            return
        self._finish_background_pipeline_work()
        if (
            completed_request is None
            or completed_request.generation != self._enhancement_generation
            or analysis_result is None
            or analysis_result.generation != self._enhancement_generation
        ):
            if error is not None and completed_request is not None and completed_request.generation == self._enhancement_generation:
                self._update_stage_statuses()
                QMessageBox.critical(self, "Analysis failed", str(error))
            return

        if completed_request.roi_residence:
            self.results = analysis_result.roi_results
            for panel, snapshot in zip(self.panels, completed_request.panels, strict=True):
                panel.report_encoded_frames = (
                    None if snapshot.encoded_frames is None else list(snapshot.encoded_frames)
                )
            self.refresh_plots_and_metrics()
            self.export_action.setEnabled(bool(self.results))
            self.export_button.setEnabled(bool(self.results))
            comparison_ready = self.active_mode == MODE_COMPARISON and len(self.panels) == 2 and bool(self.results)
            self.export_pdf_action.setEnabled(comparison_ready)
            self.export_pdf_button.setEnabled(comparison_ready)
        if completed_request.frame_brightness:
            self.frame_brightness_results = analysis_result.frame_brightness_results
            self.refresh_frame_brightness_plot()
        if completed_request.needle_segmentation:
            self.needle_brightness_results = analysis_result.needle_brightness_results
            self.needle_brightness_baselines = analysis_result.needle_brightness_baselines
            for panel in self.panels:
                panel.set_needle_segmentation_mask(
                    analysis_result.needle_masks.get(panel.label),
                    self.current_frame_index,
                )
            self.refresh_needle_brightness_plot()
        if completed_request.parent_vessel_roi:
            self.parent_vessel_roi_results = analysis_result.parent_vessel_roi_results
            self.refresh_parent_vessel_roi_plot()
            self.refresh_parent_vessel_scaled_residence_plot()
        if completed_request.background_roi:
            self.background_roi_results = analysis_result.background_roi_results
            self.refresh_background_roi_plot()
        if completed_request.temporal_change:
            self.temporal_change_results = analysis_result.temporal_change_results
            for panel in self.panels:
                panel.set_temporal_change_heatmap(
                    analysis_result.temporal_change_heatmaps.get(panel.label)
                )
            self.temporal_change_view_check.setEnabled(
                bool(self.temporal_change_results) and self.active_mode != MODE_LIVE
            )
            if self.temporal_change_view_check.isChecked():
                for panel in self.panels:
                    panel.set_temporal_change_display(True, self.current_frame_index)
        self._update_stage_statuses()
        self.statusBar().showMessage("Video enhancement and analysis complete.")

    def _poll_enhancement(self) -> None:
        if self._analysis_future is not None:
            self._poll_analysis()
            if self._enhancement_future is None:
                return
        changed_panels: set[int] = set()
        while True:
            try:
                generation, states, states_applied = self._source_pipeline_events.get_nowait()
            except Empty:
                break
            try:
                if generation == self._enhancement_generation:
                    source_changed = self._apply_source_pipeline_states(states)
                    if source_changed:
                        self.results.clear()
                        self.clear_plots_and_metrics()
                        active_request = self._enhancement_active_request
                        if active_request is not None and active_request.generation == generation:
                            for panel in self.panels:
                                panel.enhanced_frames = []
                                panel.roi_region_masks = [] if active_request.stages.roi_extraction else None
                                panel.enhance_display = True
            finally:
                states_applied.set()

        while True:
            try:
                generation, panel_index, frame_index, encoded = self._roi_region_mask_events.get_nowait()
            except Empty:
                break
            if generation != self._enhancement_generation:
                continue
            masks = self.panels[panel_index].roi_region_masks
            if masks is None:
                continue
            if frame_index == len(masks):
                masks.append(encoded)
                changed_panels.add(panel_index)

        while True:
            try:
                generation, panel_index, frame_index, encoded = self._enhancement_frame_events.get_nowait()
            except Empty:
                break
            if generation != self._enhancement_generation:
                continue
            frames = self.panels[panel_index].enhanced_frames
            if frames is None:
                continue
            if frame_index == len(frames):
                frames.append(encoded)
                changed_panels.add(panel_index)

        if changed_panels and self.panels:
            ready_frame = min(len(panel.enhanced_frames or []) for panel in self.panels) - 1
            self._set_playback_limit(max(0, ready_frame))
            for panel_index in changed_panels:
                panel = self.panels[panel_index]
                if panel.enhanced_frames is not None and self.current_frame_index < len(panel.enhanced_frames):
                    panel.seek(self.current_frame_index)

        request = self._enhancement_active_request
        if request is not None and request.generation == self._enhancement_generation:
            with self._enhancement_progress_lock:
                values = list(self._enhancement_progress_values)
                totals = list(self._enhancement_progress_totals)
                stages = list(self._enhancement_stage_messages)
                message = self._enhancement_message
            self.enhancement_progress.message_label.setText(message)
            self.enhancement_progress.set_progress(sum(values), sum(totals))
            for panel_index in range(len(self.panels)):
                self.enhancement_progress.set_panel_progress(panel_index, stages[panel_index], values[panel_index], totals[panel_index])

        future = self._enhancement_future
        if future is None or not future.done():
            return
        completed_request = self._enhancement_active_request
        self._enhancement_future = None
        self._enhancement_active_request = None
        self._enhancement_cancel = None
        try:
            prepared = future.result()
            error: Exception | None = None
        except Exception as exc:
            LOGGER.exception("Enhancement pipeline failed")
            prepared = False
            error = exc

        if self._enhancement_pending_request is not None:
            self._start_enhancement_request(self._enhancement_pending_request)
            return

        if completed_request is None or completed_request.generation != self._enhancement_generation:
            self._finish_background_pipeline_work()
            return
        if error is not None:
            self.set_display_enhancement(False)
            self._set_playback_limit(self.source_max_frame)
            self._update_stage_statuses()
            self._finish_background_pipeline_work()
            QMessageBox.critical(self, "Enhancement failed", str(error))
            return
        if prepared:
            stages = self.enhancement_stages()
            parameters = self.enhancement_parameters()
            backend_id = self._current_backend_id(stages)
            for panel_index, panel in enumerate(self.panels):
                panel.activate_prepared_sequence(
                    backend_id,
                    self.denoise_strength_spin.value(),
                    self._stages_for_roi_parameters(stages, self._roi_parameters_for_panel(panel_index)),
                    parameters,
                )
            self._set_playback_limit(self.source_max_frame)
            for panel in self.panels:
                panel.seek(self.current_frame_index)
            self.refresh_roi_needle_alignment_plot()
            self._update_stage_statuses()
            analysis_started = self._start_pipeline_analysis(
                roi_residence=self._has_enabled_stage("roi_residence_analysis"),
                frame_brightness=self._has_enabled_stage("frame_brightness_analysis"),
                needle_segmentation=self._has_enabled_stage("needle_segmentation"),
                parent_vessel_roi=self._has_enabled_stage("parent_vessel_roi_analysis"),
                background_roi=self._has_enabled_stage("background_roi_analysis"),
                temporal_change=self._has_enabled_stage("temporal_change_heatmap"),
            )
            if analysis_started:
                return
            failure = self._analysis_requirement_failure()
            self._finish_background_pipeline_work()
            self.statusBar().showMessage(failure if failure is not None else "Video enhancement complete.")
        else:
            self.set_display_enhancement(False)
            self._set_playback_limit(self.source_max_frame)
            self._update_stage_statuses()
            self._finish_background_pipeline_work()

    def _stop_enhancement_preview(self) -> None:
        self._enhancement_generation += 1
        self._enhancement_pending_request = None
        if self._enhancement_cancel is not None:
            self._enhancement_cancel.set()
        if self._analysis_cancel is not None:
            self._analysis_cancel.set()
        if self._enhancement_cancel is None and self._analysis_cancel is None:
            self._set_video_controls_enabled(bool(self.panels))
            self.open_single_mode_action.setEnabled(True)
            self.open_comparison_mode_action.setEnabled(True)
            self.load_config_action.setEnabled(True)
            self.enhancement_poll_timer.stop()
        self.enhancement_progress.finish()
        self._set_playback_limit(self.source_max_frame)
        self._update_stage_statuses()

    def on_analysis_threshold_changed(self) -> None:
        if self.active_mode == MODE_LIVE and self._network_stream_display is not None:
            self._refresh_live_analysis()
            return
        if self.results:
            self.refresh_analysis_from_existing()
        elif self._has_enabled_stage("roi_residence_analysis") and not self._background_pipeline_work_active():
            self.run_analysis()

    def run_analysis(self) -> bool:
        if not self.panels:
            return False
        failure = self._analysis_requirement_failure()
        if failure is not None:
            self.results.clear()
            self.clear_plots_and_metrics()
            self._update_stage_statuses()
            self.statusBar().showMessage(failure)
            return False

        missing = [panel.label for panel in self.panels if panel.roi() is None or panel.roi_mask() is None]
        if missing:
            self.results.clear()
            self.clear_plots_and_metrics()
            self._update_stage_statuses()
            self.statusBar().showMessage("ROI masks need review before ROI residence analysis can run.")
            return False

        if self._background_pipeline_work_active():
            return False
        return self._start_pipeline_analysis(
            roi_residence=True,
            frame_brightness=False,
            needle_segmentation=False,
            parent_vessel_roi=False,
            background_roi=False,
            temporal_change=False,
        )

    def run_frame_brightness_analysis(self) -> bool:
        return self._start_pipeline_analysis(
            roi_residence=False,
            frame_brightness=True,
            needle_segmentation=False,
            parent_vessel_roi=False,
            background_roi=False,
            temporal_change=False,
        )

    def run_needle_segmentation(self) -> bool:
        return self._start_pipeline_analysis(
            roi_residence=False,
            frame_brightness=False,
            needle_segmentation=True,
            parent_vessel_roi=False,
            background_roi=False,
            temporal_change=False,
        )

    def run_parent_vessel_roi_analysis(self) -> bool:
        if not self.panels or self._background_pipeline_work_active():
            return False
        if any(self._parent_vessel_roi_for_panel(index) is None for index in range(len(self.panels))):
            self.statusBar().showMessage("Right-click each enhanced video to place its parent vessel ROI box.")
            return False
        return self._start_pipeline_analysis(
            roi_residence=False,
            frame_brightness=False,
            needle_segmentation=False,
            parent_vessel_roi=True,
            background_roi=False,
            temporal_change=False,
        )

    def run_background_roi_analysis(self) -> bool:
        if not self.panels or self._background_pipeline_work_active():
            return False
        if any(self._background_roi_for_panel(index) is None for index in range(len(self.panels))):
            self.statusBar().showMessage("Shift-left-click each enhanced video to place its background ROI box.")
            return False
        return self._start_pipeline_analysis(
            roi_residence=False,
            frame_brightness=False,
            needle_segmentation=False,
            parent_vessel_roi=False,
            background_roi=True,
            temporal_change=False,
        )

    def run_temporal_change_heatmap(self) -> bool:
        return self._start_pipeline_analysis(
            roi_residence=False,
            frame_brightness=False,
            needle_segmentation=False,
            parent_vessel_roi=False,
            background_roi=False,
            temporal_change=True,
        )

    def refresh_analysis_from_existing(self) -> None:
        if not self.results:
            return
        threshold = self.threshold_spin.value()
        self.results = normalize_analysis_results(self.results, threshold)
        self.refresh_plots_and_metrics()

    def clear_roi_analysis(self) -> None:
        self.normalized_plot.clear()
        self.raw_plot.clear()
        self.derivative_plot.clear()
        self.baseline_to_apex_plot.clear()
        self.analysis_tabs.setTabVisible(self.baseline_to_apex_tab_index, False)
        self.roi_needle_alignment_plot.clear()
        self.roi_needle_alignment_results.clear()
        self.analysis_tabs.setTabVisible(self.roi_needle_alignment_tab_index, False)
        self.parent_vessel_scaled_residence_plot.clear()
        self.analysis_tabs.setTabVisible(self.parent_vessel_scaled_residence_tab_index, False)
        self.pre_card.set_metric("--")
        self.post_card.set_metric("--")
        self.delta_card.set_metric("--")
        self.export_action.setEnabled(False)
        self.export_button.setEnabled(False)
        self.export_pdf_action.setEnabled(False)
        self.export_pdf_button.setEnabled(False)

    def clear_plots_and_metrics(self) -> None:
        self.clear_roi_analysis()
        self._clear_frame_brightness_plots()
        self.frame_brightness_results.clear()
        self.needle_brightness_plot.clear()
        self.needle_brightness_results.clear()
        self.needle_brightness_baselines.clear()
        self.parent_vessel_roi_plot.clear()
        self.parent_vessel_roi_results.clear()
        self.analysis_tabs.setTabVisible(self.parent_vessel_roi_tab_index, False)
        self.background_roi_plot.clear()
        self.background_roi_results.clear()
        self.analysis_tabs.setTabVisible(self.background_roi_tab_index, False)
        for panel in self.panels:
            panel.needle_segmentation_mask = None
        self.temporal_change_results.clear()
        self._clear_temporal_change_views()

    def refresh_plots_and_metrics(self) -> None:
        self.normalized_plot.clear()
        self.raw_plot.clear()
        self.derivative_plot.clear()
        self.baseline_to_apex_plot.clear()
        comparison_active = self.active_mode == MODE_COMPARISON and len(self.results) > 1
        self.analysis_tabs.setTabVisible(self.baseline_to_apex_tab_index, comparison_active)
        pens = {
            panel.label: pg.mkPen(panel.color.name(), width=2.5)
            for panel in self.panels
        }
        threshold = self.threshold_spin.value()
        for label, result in self.results.items():
            pen = pens.get(label, pg.mkPen("#cbd5e1", width=2.5))
            self.normalized_plot.plot(result.time, result.normalized_signal, pen=pen, name=label)
            self.raw_plot.plot(result.time, result.mean_intensity, pen=pen, name=label)
            if len(result.mean_intensity):
                baseline_count = baseline_sample_count(result.fps, len(result.mean_intensity))
                baseline = float(np.median(result.mean_intensity[:baseline_count]))
                self.raw_plot.addItem(
                    pg.InfiniteLine(
                        pos=baseline,
                        angle=0,
                        pen=pg.mkPen(pen.color(), width=1.5, style=Qt.PenStyle.DashLine),
                    )
                )
            self.derivative_plot.plot(
                result.time,
                temporal_derivative(result.time, result.normalized_signal),
                pen=pen,
                name=label,
            )
            if comparison_active:
                self.baseline_to_apex_plot.plot(
                    result.time,
                    baseline_to_apex_normalized_signal(result),
                    pen=pen,
                    name=label,
                )

        if self.results:
            max_time = max(result.time[-1] for result in self.results.values() if len(result.time))
            self.normalized_plot.setXRange(0, max_time, padding=0)
            self.normalized_plot.setYRange(0, 1.05, padding=0)
            self.derivative_plot.setXRange(0, max_time, padding=0)
            if comparison_active:
                threshold_line = pg.InfiniteLine(
                    pos=threshold,
                    angle=0,
                    pen=pg.mkPen("#e2e8f0", width=1, style=Qt.PenStyle.DashLine),
                )
                self.baseline_to_apex_plot.addItem(threshold_line)
                self.baseline_to_apex_plot.setXRange(0, max_time, padding=0)
                self.baseline_to_apex_plot.setYRange(0, 1.05, padding=0)
        self.refresh_parent_vessel_scaled_residence_plot()

        pre = self.results.get("Pre-deployment")
        post = self.results.get("Post-deployment")
        if pre is None and self.panels:
            pre = self.results.get(self.panels[0].label)
        if pre:
            self.pre_card.set_metric(format_seconds(pre.residence_time), self._metric_detail(pre))
        if post:
            self.post_card.set_metric(format_seconds(post.residence_time), self._metric_detail(post))
        if pre and post and pre.residence_time is not None and post.residence_time is not None:
            delta = post.residence_time - pre.residence_time
            self.delta_card.set_metric(f"{delta:+.2f} s", "post minus pre")

    def refresh_frame_brightness_plot(self) -> None:
        self._clear_frame_brightness_plots()
        if self.active_mode == MODE_LIVE and self._network_stream_display is not None:
            labels_and_colors = [("Live camera", PANEL_COLORS[0].name())]
        else:
            labels_and_colors = [(panel.label, panel.color.name()) for panel in self.panels]
        comparison_active = self.active_mode == MODE_COMPARISON and len(labels_and_colors) > 1
        plot = None
        for label, color in labels_and_colors:
            result = self.frame_brightness_results.get(label)
            if result is None:
                continue
            time, source_brightness, enhanced_brightness = result
            if plot is None or not comparison_active:
                plot = pg.PlotWidget(
                    title=("Comparison Average Frame Brightness" if comparison_active else f"{label} Average Frame Brightness")
                )
                plot.setBackground("#111827")
                plot.showGrid(x=True, y=True, alpha=0.25)
                plot.getAxis("bottom").setPen("#8aa0b8")
                plot.getAxis("left").setPen("#8aa0b8")
                plot.getAxis("bottom").setTextPen("#cbd5e1")
                plot.getAxis("left").setTextPen("#cbd5e1")
                plot.setLabel("bottom", "Time", units="s")
                plot.setLabel("left", "Mean pixel value")
                plot.addLegend(offset=(12, 12))
            if np.array_equal(source_brightness, enhanced_brightness):
                plot.plot(
                    time,
                    source_brightness,
                    pen=pg.mkPen(color, width=2.5),
                    name=f"{label} frame brightness" if comparison_active else "Frame brightness",
                )
            else:
                plot.plot(
                    time,
                    source_brightness,
                    pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine),
                    name=f"{label} original" if comparison_active else "Original",
                )
                plot.plot(
                    time,
                    enhanced_brightness,
                    pen=pg.mkPen(color, width=2.5),
                    name=f"{label} enhanced" if comparison_active else "Enhanced",
                )
            if len(time):
                if self.active_mode == MODE_LIVE:
                    plot.setXRange(-60, 0, padding=0)
                else:
                    current_maximum = plot.viewRange()[0][1]
                    plot.setXRange(0, max(current_maximum, time[-1]), padding=0)
            if not comparison_active:
                self.frame_brightness_layout.addWidget(plot)
                self.frame_brightness_plots[label] = plot
        if comparison_active and plot is not None:
            self.frame_brightness_layout.addWidget(plot)
            self.frame_brightness_plots["Comparison"] = plot

    def refresh_needle_brightness_plot(self) -> None:
        self.needle_brightness_plot.clear()
        plotted_brightness: list[np.ndarray] = []
        for panel in self.panels:
            result = self.needle_brightness_results.get(panel.label)
            if result is None:
                continue
            time, brightness = result
            plotted_brightness.append(brightness)
            self.needle_brightness_plot.plot(
                time,
                brightness,
                pen=pg.mkPen(panel.color.name(), width=2.5),
                name=panel.label,
            )
            baseline = self.needle_brightness_baselines.get(panel.label)
            if baseline is not None:
                self.needle_brightness_plot.addItem(
                    pg.InfiniteLine(
                        pos=baseline,
                        angle=0,
                        pen=pg.mkPen(panel.color.name(), width=1.5, style=Qt.PenStyle.DashLine),
                    )
                )
                plotted_brightness.append(np.asarray([baseline]))
        if self.needle_brightness_results:
            max_time = max(
                time[-1]
                for time, _brightness in self.needle_brightness_results.values()
                if len(time)
            )
            self.needle_brightness_plot.setXRange(0, max_time, padding=0)
        if plotted_brightness:
            values = np.concatenate(plotted_brightness)
            data_min = float(np.min(values))
            data_max = float(np.max(values))
            display_span = max(10.0, data_max - data_min)
            center = (data_min + data_max) / 2.0
            lower = max(0.0, min(255.0 - display_span, center - display_span / 2.0))
            self.needle_brightness_plot.setYRange(lower, lower + display_span, padding=0)

    def refresh_parent_vessel_roi_plot(self) -> None:
        self.parent_vessel_roi_plot.clear()
        self.analysis_tabs.setTabVisible(self.parent_vessel_roi_tab_index, bool(self.parent_vessel_roi_results))
        maximum_time = 0.0
        for panel in self.panels:
            result = self.parent_vessel_roi_results.get(panel.label)
            if result is None:
                continue
            time, darkness = result[:2]
            self.parent_vessel_roi_plot.plot(
                time,
                darkness,
                pen=pg.mkPen(panel.color.name(), width=2.5),
                name=panel.label,
            )
            finite_darkness = darkness[np.isfinite(darkness)]
            if len(finite_darkness):
                self.parent_vessel_roi_plot.addItem(
                    pg.InfiniteLine(
                        pos=float(np.min(finite_darkness)),
                        angle=0,
                        pen=pg.mkPen(panel.color.name(), width=1.5, style=Qt.PenStyle.DashLine),
                    )
                )
            if len(time):
                maximum_time = max(maximum_time, float(time[-1]))
        if maximum_time:
            self.parent_vessel_roi_plot.setXRange(0, maximum_time, padding=0)

    def refresh_background_roi_plot(self) -> None:
        self.background_roi_plot.clear()
        self.analysis_tabs.setTabVisible(self.background_roi_tab_index, bool(self.background_roi_results))
        maximum_time = 0.0
        for panel in self.panels:
            result = self.background_roi_results.get(panel.label)
            if result is None:
                continue
            time, brightness = result
            self.background_roi_plot.plot(
                time,
                brightness,
                pen=pg.mkPen(panel.color.name(), width=2.5),
                name=panel.label,
            )
            if len(time):
                maximum_time = max(maximum_time, float(time[-1]))
        if maximum_time:
            self.background_roi_plot.setXRange(0, maximum_time, padding=0)

    def refresh_parent_vessel_scaled_residence_plot(self) -> None:
        plot = self.parent_vessel_scaled_residence_plot
        plot.clear()
        available: list[tuple[VideoPanel, AnalysisResult, float]] = []
        for panel in self.panels:
            result = self.results.get(panel.label)
            parent_vessel_result = self.parent_vessel_roi_results.get(panel.label)
            if result is None or parent_vessel_result is None:
                continue
            parent_vessel_darkness = parent_vessel_result[1]
            finite_darkness = parent_vessel_darkness[np.isfinite(parent_vessel_darkness)]
            if len(finite_darkness):
                available.append((panel, result, float(np.min(finite_darkness))))
        self.analysis_tabs.setTabVisible(
            self.parent_vessel_scaled_residence_tab_index,
            bool(available),
        )
        if not available:
            return
        reference_minimum = available[0][2]
        if reference_minimum <= 0.0:
            return
        maximum_time = 0.0
        maximum_signal = 0.0
        for panel, result, parent_vessel_minimum in available:
            scaled_signal = result.normalized_signal * (parent_vessel_minimum / reference_minimum)
            plot.plot(
                result.time,
                scaled_signal,
                pen=pg.mkPen(panel.color.name(), width=2.5),
                name=panel.label,
            )
            if len(result.time):
                maximum_time = max(maximum_time, float(result.time[-1]))
            finite_signal = scaled_signal[np.isfinite(scaled_signal)]
            if len(finite_signal):
                maximum_signal = max(maximum_signal, float(np.max(finite_signal)))
        if maximum_time:
            plot.setXRange(0, maximum_time, padding=0)
        plot.setYRange(0, max(1.05, maximum_signal * 1.05), padding=0)

    def refresh_roi_needle_alignment_plot(self) -> None:
        plot = self.roi_needle_alignment_plot
        plot.clear()
        self.analysis_tabs.setTabVisible(
            self.roi_needle_alignment_tab_index,
            bool(self.roi_needle_alignment_results),
        )
        plotted_values: list[np.ndarray] = []
        maximum_time = 0.0
        baseline_position = 0.08
        for panel in self.panels:
            result = self.roi_needle_alignment_results.get(panel.label)
            if result is None:
                continue
            color = panel.color.name()
            before_color = QColor(color)
            before_color.setAlpha(150)
            plot.plot(
                result.time,
                result.roi_brightness_before,
                pen=pg.mkPen(before_color, width=1.8, style=Qt.PenStyle.DashLine),
                name=f"{panel.label} ROI before",
            )
            plot.plot(
                result.time,
                result.roi_brightness_after,
                pen=pg.mkPen(color, width=2.5),
                name=f"{panel.label} ROI after",
            )
            plotted_values.extend((result.roi_brightness_before, result.roi_brightness_after))
            if len(result.time):
                maximum_time = max(maximum_time, float(result.time[-1]))
            baseline_specs = (
                ("ROI before", result.roi_baseline_before, Qt.PenStyle.DashDotLine, before_color),
                ("Needle before", result.needle_baseline_before, Qt.PenStyle.DotLine, before_color),
                ("ROI after", result.roi_baseline_after, Qt.PenStyle.DashDotLine, QColor(color)),
                ("Needle after", result.needle_baseline_after, Qt.PenStyle.DotLine, QColor(color)),
            )
            for baseline_name, baseline, style, baseline_color in baseline_specs:
                plot.addItem(
                    pg.InfiniteLine(
                        pos=baseline,
                        angle=0,
                        pen=pg.mkPen(baseline_color, width=1.3, style=style),
                        label=f"{panel.label} {baseline_name}: {baseline:.1f}",
                        labelOpts={"position": baseline_position, "color": baseline_color},
                    )
                )
                baseline_position = min(0.92, baseline_position + 0.11)
                plotted_values.append(np.asarray([baseline]))
        if maximum_time > 0.0:
            plot.setXRange(0.0, maximum_time, padding=0)
        if plotted_values:
            values = np.concatenate(plotted_values)
            data_min = float(np.min(values))
            data_max = float(np.max(values))
            display_span = max(10.0, data_max - data_min)
            padding = max(2.0, display_span * 0.08)
            lower = max(0.0, data_min - padding)
            upper = min(255.0, data_max + padding)
            plot.setYRange(lower, max(lower + 10.0, upper), padding=0)

    def _clear_frame_brightness_plots(self) -> None:
        while self.frame_brightness_layout.count():
            item = self.frame_brightness_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.frame_brightness_plots.clear()

    def refresh_temporal_change_views(self) -> None:
        shared_scale = self.active_mode == MODE_COMPARISON and len(self.temporal_change_results) > 1
        cumulative_peak, rate_peak = temporal_change_heatmap_peaks(self.temporal_change_results) if shared_scale else (None, None)
        heatmap_drawers = self._stage_drawers("temporal_change_heatmap")
        active_drawer = next((drawer for drawer in heatmap_drawers if drawer.enable_button.isChecked()), None)
        colormap_combo = active_drawer.findChild(QComboBox, "heatmapColormap") if active_drawer is not None else None
        colormap = str(colormap_combo.currentData()) if colormap_combo is not None else "hot"
        for panel in self.panels:
            result = self.temporal_change_results.get(panel.label)
            if result is None:
                panel.set_temporal_change_heatmap(None)
                continue
            cumulative_change, change_rate = result
            panel.set_temporal_change_heatmap(
                render_temporal_change_heatmap(
                    cumulative_change,
                    change_rate,
                    cumulative_peak,
                    rate_peak,
                    colormap=colormap,
                )
            )
        self.temporal_change_view_check.setEnabled(bool(self.temporal_change_results) and self.active_mode != MODE_LIVE)
        if self.temporal_change_view_check.isChecked():
            for panel in self.panels:
                panel.set_temporal_change_display(True, self.current_frame_index)

    def _clear_temporal_change_views(self) -> None:
        self.temporal_change_view_check.blockSignals(True)
        self.temporal_change_view_check.setChecked(False)
        self.temporal_change_view_check.setEnabled(False)
        self.temporal_change_view_check.blockSignals(False)
        for panel in self.panels:
            set_heatmap = getattr(panel, "set_temporal_change_heatmap", None)
            if set_heatmap is not None:
                set_heatmap(None)
            panel.set_comparison(self.compare_view_check.isChecked(), self.current_frame_index)

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

    def export_pdf_report(self) -> None:
        if self.active_mode != MODE_COMPARISON or len(self.panels) != 2 or not self.results:
            return
        if any(panel.temporal_change_heatmap is None for panel in self.panels):
            if self.run_temporal_change_heatmap():
                self.statusBar().showMessage("Building comparison heatmaps in the background. Export again when analysis completes.")
            else:
                self.statusBar().showMessage("Comparison report export needs enhanced frames, heatmaps, and ROI residence results.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison report PDF", str(ROOT / "contrast_comparison_report.pdf"), "PDF files (*.pdf)")
        if not path:
            return
        report_path = Path(path)
        if report_path.suffix.lower() != ".pdf":
            report_path = report_path.with_suffix(".pdf")
        screen_report_path = report_path.with_name(f"{report_path.stem}_screen.pdf")
        print_report_path = report_path.with_name(f"{report_path.stem}_print.pdf")
        frame_index = maximum_contrast_frame(self.results)
        if frame_index is None:
            self.statusBar().showMessage("Comparison report export needs valid ROI residence results.")
            return
        screen_exported = render_comparison_report(
            screen_report_path,
            self.panels,
            self.results,
            frame_index,
            print_optimized=False,
        )
        print_exported = render_comparison_report(
            print_report_path,
            self.panels,
            self.results,
            frame_index,
            print_optimized=True,
        )
        if screen_exported and print_exported:
            self.statusBar().showMessage(f"Exported screen and print comparison reports to {report_path.parent}")
        else:
            self.statusBar().showMessage("Comparison report export needs enhanced frames, heatmaps, and ROI residence results.")

    def _drawer_control_values(self, drawer: StageDrawer) -> dict[str, bool | int | float | str]:
        values: dict[str, bool | int | float | str] = {}
        for widget in drawer.findChildren(QWidget):
            name = widget.objectName()
            if not name:
                continue
            if isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = str(widget.currentData())
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
        return values

    def _config_data(self) -> dict[str, object]:
        mode = self.active_mode if self.panels else MODE_SINGLE
        return {
            "version": CONFIG_VERSION,
            "videos": {
                "mode": mode,
                "paths": [str(panel.path) for panel in self.panels],
            },
            "pipeline": [
                {
                    "key": drawer.stage_key,
                    "enabled": drawer.enable_button.isChecked(),
                    "controls": self._drawer_control_values(drawer),
                }
                for drawer in self.pipeline_stage_drawers
            ],
            "view": {
                "show_source": self.compare_view_check.isChecked(),
                "compare_enabled": self.compare_view_check.isChecked(),
                "mask_overlay_enabled": self.overlay_mask_check.isChecked(),
                "playback_speed": self.speed_slider.value(),
                "loop": self.loop_check.isChecked(),
                "frame_index": self.current_frame_index,
                "roi_settings": self._roi_settings_data(),
                "parent_vessel_roi_settings": self._parent_vessel_roi_settings_data(),
                "background_roi_settings": self._background_roi_settings_data(),
            },
            "analysis": {"clearance_threshold": self.threshold_spin.value()},
        }

    def _pipeline_settings_data(self) -> dict[str, object]:
        return {
            "version": CONFIG_VERSION,
            "pipeline": [
                {
                    "key": drawer.stage_key,
                    "enabled": drawer.enable_button.isChecked(),
                    "controls": self._drawer_control_values(drawer),
                }
                for drawer in self.pipeline_stage_drawers
            ],
        }

    def _set_drawer_control_values(self, drawer: StageDrawer, values: dict[str, object]) -> None:
        for name, value in values.items():
            widget = drawer.findChild(QWidget, name)
            if widget is None:
                continue
            widget.blockSignals(True)
            try:
                if isinstance(widget, QCheckBox) and isinstance(value, bool):
                    widget.setChecked(value)
                elif isinstance(widget, QComboBox) and isinstance(value, str):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, QSpinBox) and isinstance(value, int) and not isinstance(value, bool):
                    widget.setValue(value)
                elif isinstance(widget, QDoubleSpinBox) and isinstance(value, (int, float)) and not isinstance(value, bool):
                    widget.setValue(float(value))
            finally:
                widget.blockSignals(False)

    def _validate_pipeline_settings_data(self, config: object) -> dict[str, object]:
        if not isinstance(config, dict) or config.get("version") != CONFIG_VERSION:
            raise ValueError(f"Unsupported configuration version. Expected version {CONFIG_VERSION}.")
        pipeline = config.get("pipeline")
        if not isinstance(pipeline, list):
            raise ValueError("Configuration must include a pipeline section.")
        for stage in pipeline:
            if not isinstance(stage, dict) or stage.get("key") not in self.stage_drawer_templates:
                raise ValueError("Configuration contains an unknown pipeline stage.")
            if not isinstance(stage.get("enabled"), bool):
                raise ValueError("Each pipeline stage must define an enabled state.")
            controls = stage.get("controls", {})
            if not isinstance(controls, dict):
                raise ValueError("Pipeline stage controls must be an object.")
        return config

    def _validate_config_data(self, config: object) -> tuple[dict[str, object], list[Path]]:
        config = self._validate_pipeline_settings_data(config)
        videos = config.get("videos")
        if not isinstance(videos, dict):
            raise ValueError("Configuration must include videos and pipeline sections.")

        video_paths: list[Path]
        if "paths" in videos:
            raw_paths = videos.get("paths")
            if not isinstance(raw_paths, list):
                raise ValueError("Configured video paths must be a list.")
            video_paths = [Path(str(path)).expanduser() for path in raw_paths]
        else:
            # Backward compatibility for older two-video config shape.
            pre_path = Path(str(videos.get("pre_deployment", ""))).expanduser()
            post_path = Path(str(videos.get("post_deployment", ""))).expanduser()
            video_paths = [pre_path, post_path]

        if len(video_paths) not in (0, 1, 2):
            raise ValueError("Configuration must include zero, one, or two videos.")
        configured_mode = videos.get("mode", MODE_COMPARISON if len(video_paths) > 1 else MODE_SINGLE)
        if configured_mode not in {MODE_SINGLE, MODE_COMPARISON, MODE_LIVE}:
            raise ValueError("Configuration contains an unknown video mode.")
        if configured_mode == MODE_LIVE and len(video_paths) != 1:
            raise ValueError("Live camera mode requires exactly one looping video source.")
        missing = [str(path) for path in video_paths if not path.is_file()]
        if missing:
            raise ValueError("Configured video files are unavailable: " + ", ".join(missing))
        return config, video_paths

    def _apply_pipeline_settings(self, pipeline: list[dict[str, object]]) -> None:
        for drawer in self.pipeline_stage_drawers:
            self._pipeline_layout_for(drawer).removeWidget(drawer)
            drawer.hide()
            drawer.setParent(None)
        self.pipeline_stage_drawers = []
        self.source_pipeline_stage_drawers = []
        self.live_pipeline_stage_drawers = []
        self._sync_pipeline_stage_lists()

        for stage in pipeline:
            drawer = self._add_pipeline_stage(cast(str, stage["key"]))
            self._set_drawer_control_values(drawer, cast(dict[str, object], stage.get("controls", {})))
            drawer.enable_button.blockSignals(True)
            drawer.enable_button.setChecked(cast(bool, stage["enabled"]))
            drawer.enable_button.blockSignals(False)

        self._sync_pipeline_stage_lists()
        self._refresh_pipeline_stage_ui()
        self._sync_active_denoise_controls()

    def _load_default_pipeline_settings(self) -> None:
        try:
            config = self._validate_pipeline_settings_data(json.loads(DEFAULT_PIPELINE_SETTINGS_FILE.read_text()))
        except (OSError, ValueError, json.JSONDecodeError):
            LOGGER.warning("Could not load default pipeline settings from %s", DEFAULT_PIPELINE_SETTINGS_FILE, exc_info=True)
            return
        self._loading_config = True
        try:
            self._apply_pipeline_settings(cast(list[dict[str, object]], config["pipeline"]))
        finally:
            self._loading_config = False

    def _remember_recent_config(self, path: Path) -> None:
        CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        RECENT_CONFIG_FILE.write_text(json.dumps({"path": str(path.resolve())}, indent=2) + "\n")

    def _most_recent_config_path(self) -> Path | None:
        try:
            recent = json.loads(RECENT_CONFIG_FILE.read_text())
            path = Path(str(recent["path"])).expanduser()
            if path.is_file():
                return path
        except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
            pass
        if not CONFIG_DIRECTORY.is_dir():
            return None
        configs = [path for path in CONFIG_DIRECTORY.glob("*.json") if path != RECENT_CONFIG_FILE]
        return max(configs, key=lambda path: path.stat().st_mtime, default=None)

    def _update_mode_selection_config_actions(self) -> None:
        recent_path = self._most_recent_config_path()
        self.load_recent_mode_button.setEnabled(recent_path is not None)
        tooltip = f"Load {recent_path.name}" if recent_path is not None else "No recent configuration available"
        self.load_recent_mode_button.setToolTip(tooltip)

    def save_config(self) -> None:
        if not self.panels:
            self.statusBar().showMessage("Load video files before saving a configuration.")
            return
        CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            str(CONFIG_DIRECTORY / "contrast_config.json"),
            "Contrast configuration (*.json)",
        )
        if not path:
            return
        config_path = Path(path).with_suffix(".json")
        try:
            config_path.write_text(json.dumps(self._config_data(), indent=2, sort_keys=True) + "\n")
            self._remember_recent_config(config_path)
            self._update_mode_selection_config_actions()
        except OSError as exc:
            LOGGER.exception("Could not save configuration to %s", config_path)
            QMessageBox.critical(self, "Could not save configuration", str(exc))
            return
        LOGGER.info("Saved configuration to %s", config_path)
        self.statusBar().showMessage(f"Saved configuration to {config_path}")

    def save_default_pipeline_settings(self) -> None:
        try:
            CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
            DEFAULT_PIPELINE_SETTINGS_FILE.write_text(
                json.dumps(self._pipeline_settings_data(), indent=2, sort_keys=True) + "\n"
            )
        except OSError as exc:
            LOGGER.exception("Could not save default pipeline settings to %s", DEFAULT_PIPELINE_SETTINGS_FILE)
            QMessageBox.critical(self, "Could not save default pipeline settings", str(exc))
            return
        LOGGER.info("Saved default pipeline settings to %s", DEFAULT_PIPELINE_SETTINGS_FILE)
        self.statusBar().showMessage(f"Saved default pipeline settings to {DEFAULT_PIPELINE_SETTINGS_FILE}")

    def load_config(self) -> None:
        CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load configuration",
            str(CONFIG_DIRECTORY),
            "Contrast configuration (*.json)",
        )
        if path:
            self._load_config_file(Path(path))

    def load_most_recent_config(self) -> None:
        path = self._most_recent_config_path()
        if path is None:
            self.statusBar().showMessage("No recent configuration found.")
            self._update_mode_selection_config_actions()
            return
        self._load_config_file(path)

    def _load_config_file(self, path: Path, show_error: bool = True) -> bool:
        try:
            config = json.loads(path.read_text())
            config, video_paths = self._validate_config_data(config)
            self._apply_config(config, video_paths)
            self._remember_recent_config(path)
            self._update_mode_selection_config_actions()
        except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            LOGGER.exception("Could not load configuration from %s", path)
            if show_error:
                QMessageBox.critical(self, "Could not load configuration", str(exc))
            return False
        self.statusBar().showMessage(f"Loaded configuration from {path}")
        LOGGER.info("Loaded configuration from %s", path)
        return True

    def _apply_config(self, config: dict[str, object], video_paths: list[Path]) -> None:
        self.pause()
        self._loading_config = True
        try:
            videos = cast(dict[str, object], config["videos"])
            self._set_video_panels(video_paths, live_input=videos.get("mode") == MODE_LIVE)
            pipeline = cast(list[dict[str, object]], config["pipeline"])
            self._apply_pipeline_settings(pipeline)

            view = config.get("view", {})
            if isinstance(view, dict):
                show_source = view.get("show_source", view.get("compare_enabled", True))
                self.compare_view_check.setChecked(bool(show_source))
                self.overlay_mask_check.setChecked(bool(view.get("mask_overlay_enabled", True)))
                self._apply_roi_settings_data(view.get("roi_settings"))
                self._apply_parent_vessel_roi_settings_data(view.get("parent_vessel_roi_settings"))
                self._apply_background_roi_settings_data(view.get("background_roi_settings"))
                playback_speed = view.get("playback_speed", 100)
                if isinstance(playback_speed, int):
                    self.speed_slider.setValue(playback_speed)
                self.loop_check.setChecked(bool(view.get("loop", False)))
            analysis = config.get("analysis", {})
            if isinstance(analysis, dict):
                threshold = analysis.get("clearance_threshold")
                if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                    self.threshold_spin.setValue(float(threshold))

            self._set_live_incompatible_stages_enabled(self.active_mode != MODE_LIVE)
            self._sync_trimmed_video_window()
            frame_index = view.get("frame_index", 0) if isinstance(view, dict) else 0
            self.set_frame_index(frame_index if isinstance(frame_index, int) else 0)
        finally:
            self._loading_config = False
        self.results.clear()
        self.clear_plots_and_metrics()
        self.on_pipeline_stages_changed()

    def closeEvent(self, event) -> None:  # noqa: ANN001
        LOGGER.info("Closing desktop application")
        self.enhancement_poll_timer.stop()
        self.network_stream_poll_timer.stop()
        if self._enhancement_cancel is not None:
            self._enhancement_cancel.set()
        if self._analysis_cancel is not None:
            self._analysis_cancel.set()
        self._enhancement_executor.shutdown(wait=True, cancel_futures=True)
        self._analysis_executor.shutdown(wait=True, cancel_futures=True)
        for panel in list(self.panels):
            panel.close()
        for denoiser in self.deep_denoisers.values():
            close = getattr(denoiser, "close", None)
            if close is not None:
                close()
        if self._stream_service is not None:
            close = getattr(self._stream_service, "close", None)
            if close is not None:
                close()
        if self._stream_server is not None:
            self._stream_server.shutdown()
            self._stream_server.server_close()
        if self._stream_server_thread is not None:
            self._stream_server_thread.join()
        super().closeEvent(event)


def analyze_gray_frames(
    label: str,
    path: Path,
    fps: float,
    roi: QRect,
    roi_mask: np.ndarray | None,
    gray_frames: list[np.ndarray],
    threshold_fraction: float,
    gain_corrected: bool,
    camera_view_mask: np.ndarray | None = None,
) -> AnalysisResult:
    means: list[float] = []
    references: list[float] = []

    for gray in gray_frames:
        means.append(roi_mean(gray, roi, roi_mask, camera_view_mask))
        references.append(reference_mean(gray, roi, roi_mask, camera_view_mask))

    roi_intensity = np.asarray(means, dtype=float)
    reference_intensity = np.asarray(references, dtype=float)
    measurement_intensity = smooth_temporal_signal(roi_intensity, fps)

    return build_analysis_result(label, path, fps, roi, measurement_intensity, reference_intensity, threshold_fraction, gain_corrected)


def build_analysis_result(
    label: str,
    path: Path,
    fps: float,
    roi: QRect,
    mean_intensity: np.ndarray,
    reference_intensity: np.ndarray,
    threshold_fraction: float,
    gain_corrected: bool,
    normalization_peak: float | None = None,
) -> AnalysisResult:
    time = np.arange(len(mean_intensity), dtype=float) / fps
    if len(mean_intensity) == 0:
        empty = np.asarray([], dtype=float)
        return AnalysisResult(label, path, fps, roi, empty, empty, empty, empty, empty, gain_corrected, threshold_fraction, 0.0, None, None, None, None, 0.0, 0.0)

    baseline_count = baseline_sample_count(fps, len(mean_intensity))
    baseline = float(np.median(mean_intensity[:baseline_count]))
    contrast_signal = np.clip(baseline - mean_intensity, 0, None)
    peak_signal = float(np.max(contrast_signal))
    scale_peak = peak_signal if normalization_peak is None else normalization_peak
    normalized = contrast_signal / scale_peak if scale_peak > 0 else np.zeros_like(contrast_signal)
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
        residence_time = max(0.0, clear_time - peak_time)

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


def recompute_threshold_metrics(
    result: AnalysisResult,
    threshold_fraction: float,
    normalization_peak: float | None = None,
) -> AnalysisResult:
    return build_analysis_result(
        result.label,
        result.path,
        result.fps,
        result.roi,
        result.mean_intensity,
        result.reference_intensity,
        threshold_fraction,
        result.gain_corrected,
        normalization_peak,
    )


def normalize_analysis_results(
    results: dict[str, AnalysisResult],
    threshold_fraction: float,
) -> dict[str, AnalysisResult]:
    return {
        label: recompute_threshold_metrics(result, threshold_fraction)
        for label, result in results.items()
    }


def run_headless(config_path: Path) -> None:
    from stream_server import RawFrameRecorder, LiveStreamProcessor, StreamService, create_http_server, load_stream_configuration

    LOGGER.info("Loading headless configuration from %s", config_path)
    settings, stages, parameters, noise_sigma, auto_crop_enabled, denoiser_settings = load_stream_configuration(str(config_path))
    denoiser: FrameDenoiser | None = None
    if "denoise" in stages.enabled_stage_order:
        if denoiser_settings.mode == "non-local-means":
            denoiser = NonLocalMeansDenoiser()
        elif denoiser_settings.mode == "tensor-nlm-ngc":
            from container_denoiser import ContainerDenoiser

            denoiser = ContainerDenoiser(
                "tensor-nlm",
                None,
                denoiser_settings.batch_size,
                denoiser_settings.precision,
            )
        elif denoiser_settings.mode.endswith("-ngc"):
            from container_denoiser import ContainerDenoiser

            denoiser = ContainerDenoiser(
                "ffdnet",
                ROOT / "models" / "ffdnet_gray.pth",
                denoiser_settings.batch_size,
                denoiser_settings.precision,
            )
        elif denoiser_settings.mode == "ffdnet-native":
            from deep_denoiser import FFDNetDenoiser

            denoiser = FFDNetDenoiser(ROOT / "models" / "ffdnet_gray.pth", denoiser_settings.precision)
        else:
            raise ValueError(f"Unsupported headless denoiser mode: {denoiser_settings.mode}")
    processor = LiveStreamProcessor(
        stages,
        parameters,
        noise_sigma,
        settings.crop_sample_frames,
        settings.jpeg_quality,
        auto_crop_enabled,
        denoiser,
        dsa_mask_delay_frames=round(settings.recording_fps),
    )
    service = StreamService(
        processor,
        settings.max_frame_bytes,
        RawFrameRecorder(settings.recording_directory, settings.recording_fps),
    )
    server = create_http_server(settings, service)
    LOGGER.info("Contrast stream service listening on http://%s:%s", settings.host, settings.port)
    LOGGER.info("POST JPEG frames to /ingest; read enhanced MJPEG from /egress.mjpg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Headless stream service interrupted")
    finally:
        LOGGER.info("Shutting down headless stream service")
        service.close()
        server.server_close()
        if denoiser is not None:
            close = getattr(denoiser, "close", None)
            if close is not None:
                close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Contrast fluoroscopy enhancement")
    parser.add_argument("--headless", action="store_true", help="Run the HTTP live-stream service without the desktop UI")
    parser.add_argument("--config", type=Path, help="Configuration JSON used by --headless")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs" / "contrast.log")
    arguments = parser.parse_args(argv)
    configure_logging(arguments.log_level, arguments.log_file)
    install_exception_logging()
    LOGGER.info("Starting Contrast in %s mode", "headless" if arguments.headless else "desktop")
    if arguments.headless:
        if arguments.config is None:
            parser.error("--headless requires --config PATH")
        run_headless(arguments.config)
        return
    if arguments.config is not None:
        parser.error("--config is only supported with --headless")
    app = QApplication(sys.argv)
    window = ContrastWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
