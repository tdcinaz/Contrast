from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS = {
    "Pre-deployment": ROOT / "PPI150_PreDeployment_Contrast.mov",
    "Post-deployment": ROOT / "PPI150_PostDeployment_Contrast.mov",
}


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


def first_frame_median(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        if not ok:
            return 128.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.median(gray))
    finally:
        capture.release()


def enhance_frame_for_display(frame: np.ndarray, target_median: float) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_median = max(1.0, float(np.median(gray)))
    gain = float(np.clip(target_median / current_median, 0.55, 1.85))
    stabilized = np.clip(gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    smoothed = cv2.GaussianBlur(stabilized, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    enhanced = clahe.apply(smoothed)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


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
        self._pixmap = QPixmap()
        self._roi: QRect | None = None
        self._drag_origin: QPoint | None = None
        self._display_rect = QRect()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #0e1116; border: 1px solid #253044; border-radius: 8px;")

    def set_frame(self, frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        bytes_per_line = channels * width
        image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.frame_size = (width, height)
        self._pixmap = QPixmap.fromImage(image)
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

        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            self._display_rect = QRect(x, y, scaled.width(), scaled.height())
            painter.drawPixmap(self._display_rect, scaled)
        else:
            self._display_rect = QRect()
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
        self.capture = cv2.VideoCapture(str(path))
        self.current_frame: np.ndarray | None = None
        self.current_frame_index = -1
        self.enhance_display = False
        self.target_median = first_frame_median(path)
        self.enhanced_frames: list[np.ndarray] | None = None

        self.display = VideoDisplay(label, color)
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
        progress_callback: callable[[int, int], bool] | None = None,
    ) -> bool:
        if self.enhanced_frames is not None and len(self.enhanced_frames) == self.info.frame_count:
            return True

        capture = cv2.VideoCapture(str(self.path))
        try:
            enhanced_frames: list[np.ndarray] = []
            for frame_index in range(self.info.frame_count):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not precompute enhancement for video: {self.path}")
                enhanced_frames.append(enhance_frame_for_display(frame, self.target_median))
                if progress_callback is not None and not progress_callback(frame_index + 1, self.info.frame_count):
                    return False
            self.enhanced_frames = enhanced_frames
            return True
        finally:
            capture.release()

    def _metadata_text(self) -> str:
        return f"{self.info.width}x{self.info.height} | {self.info.fps:.1f} fps | {self.info.duration:.1f} s"

    def _display_frame(self, frame: np.ndarray, apply_enhancement: bool | None = None) -> None:
        self.current_frame = frame
        if apply_enhancement is None:
            apply_enhancement = self.enhance_display
        can_enhance = self.enhanced_frames is not None and 0 <= self.current_frame_index < len(self.enhanced_frames)
        display_frame = self.enhanced_frames[self.current_frame_index] if apply_enhancement and can_enhance else frame
        self.display.set_frame(display_frame)

    def read_next(self, playback: bool = False) -> bool:
        if self.current_frame_index >= self.info.frame_count - 1:
            return False
        ok, frame = self.capture.read()
        if ok:
            self.current_frame_index += 1
            self._display_frame(frame, apply_enhancement=self.enhance_display)
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
            self._display_frame(frame)
        return ok

    def roi(self) -> QRect | None:
        return self.display.roi()

    def set_video(self, path: Path) -> None:
        self.capture.release()
        self.path = path
        self.info = probe_video(path)
        self.capture = cv2.VideoCapture(str(path))
        self.current_frame_index = -1
        self.target_median = first_frame_median(path)
        self.enhanced_frames = None
        self.path_label.setText(path.name)
        self.meta_label.setText(self._metadata_text())
        self.display.clear_roi()
        self.seek(0)

    def set_enhancement(self, enabled: bool, frame_index: int) -> None:
        self.enhance_display = enabled
        self.seek(frame_index)

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
        toolbar = QToolBar("Playback")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

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

        toolbar.addWidget(self.play_button)
        toolbar.addWidget(self.step_back_button)
        toolbar.addWidget(self.step_forward_button)
        toolbar.addWidget(self.frame_slider)
        toolbar.addWidget(QLabel("Frame"))
        toolbar.addWidget(self.frame_spin)
        toolbar.addWidget(QLabel("Speed"))
        toolbar.addWidget(self.speed_slider)
        toolbar.addWidget(self.speed_label)
        toolbar.addWidget(self.time_label)

        video_row = QWidget()
        video_layout = QHBoxLayout(video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(14)
        video_layout.addWidget(self.pre_panel)
        video_layout.addWidget(self.post_panel)

        analysis_panel = self._build_analysis_panel()
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(video_row)
        splitter.addWidget(analysis_panel)
        splitter.setSizes([560, 320])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 14, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(splitter)
        self.setCentralWidget(central)
        self.loading_overlay = LoadingOverlay(central)
        self.loading_overlay.setGeometry(central.rect())
        self.setStatusBar(QStatusBar())

    def _build_analysis_panel(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        controls = QGroupBox("ROI analysis")
        controls.setMaximumWidth(360)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setSpacing(12)

        hint = QLabel("Draw a box over the aneurysm sac in each video. The signal is computed as baseline brightness minus ROI brightness, so darker contrast produces a positive curve.")
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        controls_layout.addWidget(hint)

        self.enhance_button = QPushButton("Enable video enhancement")
        self.enhance_button.setCheckable(True)
        self.enhance_button.setChecked(False)
        self.enhance_button.clicked.connect(self.on_enhance_clicked)
        controls_layout.addWidget(self.enhance_button)

        self.gain_correct_check = QCheckBox("Correct gain drift in analysis")
        self.gain_correct_check.setChecked(True)
        self.gain_correct_check.stateChanged.connect(lambda: self.on_analysis_filter_changed())
        controls_layout.addWidget(self.gain_correct_check)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Clearance threshold"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.95)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.20)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.valueChanged.connect(self.refresh_analysis_from_existing)
        threshold_row.addWidget(self.threshold_spin)
        controls_layout.addLayout(threshold_row)

        self.analyze_button = QPushButton("Analyze ROIs")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.analyze_button)

        self.export_button = QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_csv)
        self.export_button.setEnabled(False)
        controls_layout.addWidget(self.export_button)
        controls_layout.addStretch()

        self.pre_card = MetricCard("Pre residence")
        self.post_card = MetricCard("Post residence")
        self.delta_card = MetricCard("Difference")
        controls_layout.addWidget(self.pre_card)
        controls_layout.addWidget(self.post_card)
        controls_layout.addWidget(self.delta_card)

        plot_group = QFrame()
        plot_group.setObjectName("plotPanel")
        plot_layout = QGridLayout(plot_group)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)

        self.normalized_plot = pg.PlotWidget(title="Normalized Contrast Residence")
        self.raw_plot = pg.PlotWidget(title="ROI Mean Brightness")
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

        layout.addWidget(controls)
        layout.addWidget(plot_group, 1)
        return container

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
            QSpinBox, QDoubleSpinBox { background: #111827; border: 1px solid #334155; border-radius: 6px; padding: 5px; color: #e5edf6; }
            QGroupBox { background: #111827; border: 1px solid #253044; border-radius: 8px; margin-top: 12px; padding: 12px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #f8fafc; }
            QFrame#videoPanel, QFrame#plotPanel { background: #111827; border: 1px solid #253044; border-radius: 8px; }
            QLabel#panelTitle { font-size: 16px; font-weight: 700; color: #f8fafc; }
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

        self.enhance_button.blockSignals(True)
        self.enhance_button.setChecked(False)
        self.enhance_button.setText("Enable video enhancement")
        self.enhance_button.blockSignals(False)
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

    def on_enhance_clicked(self, checked: bool) -> None:
        if checked:
            if not self.ensure_enhancement_ready():
                self.enhance_button.blockSignals(True)
                self.enhance_button.setChecked(False)
                self.enhance_button.blockSignals(False)
                self.enhance_button.setText("Enable video enhancement")
                return
            self.set_display_enhancement(True)
            self.enhance_button.setText("Disable video enhancement")
            return

        self.set_display_enhancement(False)
        self.enhance_button.setText("Enable video enhancement")

    def ensure_enhancement_ready(self) -> bool:
        pending_panels = [panel for panel in self.panels if panel.enhanced_frames is None]
        if not pending_panels:
            return True

        self.pause()
        total_frames = sum(panel.info.frame_count for panel in pending_panels)
        self.loading_overlay.begin("Enhancing video display. Please wait...", total_frames)

        completed = 0
        try:
            for panel in pending_panels:
                def progress_callback(done: int, _total: int) -> bool:
                    self.loading_overlay.set_progress(completed + done)
                    return True

                prepared = panel.prepare_enhanced_frames(progress_callback)
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
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        if hasattr(self, "loading_overlay") and self.centralWidget() is not None:
            self.loading_overlay.setGeometry(self.centralWidget().rect())


def analyze_video(
    label: str,
    path: Path,
    info: VideoInfo,
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        baseline_reference = float(np.median(reference_intensity[:baseline_count]))
        reference_safe = np.clip(reference_intensity, 1.0, None)
        gain = np.clip(baseline_reference / reference_safe, 0.55, 1.85)
        measurement_intensity = roi_intensity * gain

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
