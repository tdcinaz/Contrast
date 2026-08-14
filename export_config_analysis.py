from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from main import (
    AnalysisResult,
    ContrastWindow,
    configure_logging,
    overlay_needle_mask,
    overlay_roi_regions,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATTERN = "TF_*_Norm.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "final_reports" / "excel_analysis"
VIDEO_FIELDS = (
    "video_file_name",
    "parent_vessel_roi_apex_min_level",
    "parent_vessel_roi_darkest_10_percent_median",
    "aneurysm_residence_time_s",
    "time_s",
    "aneurysm_roi_absolute_average_pixel_darkness",
    "background_roi_average_pixel_brightness",
    "needle_average_pixel_brightness",
)
SUMMARY_FIELDS = VIDEO_FIELDS[:3]


def parent_vessel_minimum(parent_vessel_result: tuple[np.ndarray, np.ndarray]) -> float | None:
    finite_values = parent_vessel_result[1][np.isfinite(parent_vessel_result[1])]
    return float(np.min(finite_values)) if len(finite_values) else None


def _csv_value(value: float | None) -> float | str:
    return "" if value is None or not math.isfinite(value) else value


def lowest_aneurysm_brightness_frame(result: AnalysisResult) -> int | None:
    brightness = np.asarray(result.mean_intensity, dtype=float)
    valid = np.isfinite(brightness)
    if not np.any(valid):
        return None
    return int(np.argmin(np.where(valid, brightness, np.inf)))


def _footer_text(frame_width: int, video_name: str, frame_number: int, time_s: float) -> tuple[str, float]:
    text = f"{video_name} | Frame {frame_number} | {time_s:.3f} s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    for scale in np.linspace(0.7, 0.35, num=8):
        text_width = cv2.getTextSize(text, font, float(scale), 1)[0][0]
        if text_width <= frame_width - 24:
            return text, float(scale)
    return text, 0.35


def _draw_roi_outlines(
    image: np.ndarray,
    panel,
    parent_vessel_roi: tuple[int, int, float] | None,
    background_roi: tuple[int, int] | None,
) -> None:
    roi = getattr(panel, "roi", lambda: None)()
    roi_mask = getattr(panel, "roi_mask", lambda: None)()
    if roi is not None and roi.isValid():
        color = (panel.color.blue(), panel.color.green(), panel.color.red())
        if roi_mask is None:
            cv2.rectangle(image, (roi.x(), roi.y()), (roi.right(), roi.bottom()), color, 2)
        else:
            contours, _ = cv2.findContours(roi_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(image, [contour + np.asarray([[[roi.x(), roi.y()]]])], -1, color, 2)
    if parent_vessel_roi is not None:
        center_x, center_y, rotation = parent_vessel_roi
        corners = cv2.boxPoints(((float(center_x), float(center_y)), (50.0, 50.0), float(rotation))).round().astype(np.int32)
        cv2.polylines(image, [corners], True, (21, 204, 250), 2)
    if background_roi is not None:
        center_x, center_y = background_roi
        cv2.rectangle(image, (center_x - 75, center_y - 75), (center_x + 75, center_y + 75), (94, 197, 34), 2)


def write_video_image(
    output_path: Path,
    result: AnalysisResult,
    panel,
    parent_vessel_roi: tuple[int, int, float] | None = None,
    background_roi: tuple[int, int] | None = None,
) -> bool:
    frame_index = lowest_aneurysm_brightness_frame(result)
    encoded_frames = getattr(panel, "report_encoded_frames", None) or panel.enhanced_frames
    if frame_index is None or encoded_frames is None or frame_index >= len(encoded_frames):
        return False
    enhanced = cv2.imdecode(encoded_frames[frame_index], cv2.IMREAD_GRAYSCALE)
    if enhanced is None:
        return False

    image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    masks = panel.roi_region_masks
    if masks is not None and frame_index < len(masks):
        regions = cv2.imdecode(masks[frame_index], cv2.IMREAD_GRAYSCALE)
        if regions is not None:
            image = overlay_roi_regions(
                image,
                regions,
                (panel.color.blue(), panel.color.green(), panel.color.red()),
            )
    if panel.needle_segmentation_mask is not None:
        image = overlay_needle_mask(image, panel.needle_segmentation_mask)
    _draw_roi_outlines(image, panel, parent_vessel_roi, background_roi)

    footer_height = max(36, round(image.shape[0] * 0.06))
    footer = np.full((footer_height, image.shape[1], 3), (14, 17, 22), dtype=np.uint8)
    text, scale = _footer_text(image.shape[1], result.path.name, frame_index + 1, float(result.time[frame_index]))
    text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][1]
    cv2.putText(
        footer,
        text,
        (12, (footer_height + text_height) // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), np.vstack((image, footer)))


def write_video_csv(
    output_path: Path,
    result: AnalysisResult,
    parent_minimum: float | None,
    parent_vessel_result: tuple[np.ndarray, np.ndarray] | None,
    background_result: tuple[np.ndarray, np.ndarray] | None,
    needle_result: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    parent_vessel_darkness = parent_vessel_result[1] if parent_vessel_result is not None else None
    background_brightness = background_result[1] if background_result is not None else None
    needle_brightness = needle_result[1] if needle_result is not None else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=VIDEO_FIELDS)
        writer.writeheader()
        for index, (time_s, darkness) in enumerate(zip(result.time, result.mean_intensity, strict=True)):
            parent_vessel_value = (
                _csv_value(float(parent_vessel_darkness[index]))
                if parent_vessel_darkness is not None and index < len(parent_vessel_darkness)
                else ""
            )
            background_value = (
                _csv_value(float(background_brightness[index]))
                if background_brightness is not None and index < len(background_brightness)
                else ""
            )
            needle_value = (
                _csv_value(float(needle_brightness[index]))
                if needle_brightness is not None and index < len(needle_brightness)
                else ""
            )
            writer.writerow(
                {
                    "video_file_name": result.path.name,
                    "parent_vessel_roi_apex_min_level": _csv_value(parent_minimum),
                    "parent_vessel_roi_darkest_10_percent_median": parent_vessel_value,
                    "aneurysm_residence_time_s": _csv_value(result.residence_time),
                    "time_s": float(time_s),
                    "aneurysm_roi_absolute_average_pixel_darkness": float(darkness),
                    "background_roi_average_pixel_brightness": background_value,
                    "needle_average_pixel_brightness": needle_value,
                }
            )


class ConfigAnalysisExporter:
    def __init__(
        self,
        app: QApplication,
        window: ContrastWindow,
        config_paths: list[Path],
        output_directory: Path,
    ) -> None:
        self.app = app
        self.window = window
        self.config_paths = config_paths
        self.output_directory = output_directory
        self.summary_rows: list[dict[str, float | str]] = []
        self.failures: list[str] = []
        self.config_index = 0

    def start(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        QTimer.singleShot(0, self._load_next_config)

    def _load_next_config(self) -> None:
        if self.config_index >= len(self.config_paths):
            self._finish()
            return
        config_path = self.config_paths[self.config_index]
        print(f"[{self.config_index + 1}/{len(self.config_paths)}] Processing {config_path.name}", flush=True)
        if not self.window._load_config_file(config_path, show_error=False):
            self.failures.append(f"{config_path}: configuration could not be loaded")
            self.config_index += 1
            QTimer.singleShot(0, self._load_next_config)
            return
        QTimer.singleShot(100, self._wait_for_analysis)

    def _wait_for_analysis(self) -> None:
        if self.window._background_pipeline_work_active():
            QTimer.singleShot(100, self._wait_for_analysis)
            return
        self._export_current_config()
        self.config_index += 1
        QTimer.singleShot(0, self._load_next_config)

    def _export_current_config(self) -> None:
        config_path = self.config_paths[self.config_index]
        for panel in self.window.panels:
            result = self.window.results.get(panel.label)
            parent_result = self.window.parent_vessel_roi_results.get(panel.label)
            if result is None or parent_result is None:
                self.failures.append(f"{config_path}: analysis results are missing for {panel.label}")
                continue
            parent_minimum = parent_vessel_minimum(parent_result)
            background_result = self.window.background_roi_results.get(panel.label)
            needle_result = self.window.needle_brightness_results.get(panel.label)
            output_path = self.output_directory / f"{result.path.stem}_analysis.csv"
            write_video_csv(output_path, result, parent_minimum, parent_result, background_result, needle_result)
            image_path = self.output_directory / f"{result.path.stem}_analysis.png"
            panel_index = self.window.panels.index(panel)
            parent_vessel_rois = getattr(self.window, "_parent_vessel_rois", [])
            background_rois = getattr(self.window, "_background_rois", [])
            parent_vessel_roi = parent_vessel_rois[panel_index] if panel_index < len(parent_vessel_rois) else None
            background_roi = background_rois[panel_index] if panel_index < len(background_rois) else None
            if not write_video_image(image_path, result, panel, parent_vessel_roi, background_roi):
                self.failures.append(f"{config_path}: image export failed for {panel.label}")
                continue
            self.summary_rows.append(
                {
                    "video_file_name": result.path.name,
                    "parent_vessel_roi_apex_min_level": _csv_value(parent_minimum),
                    "aneurysm_residence_time_s": _csv_value(result.residence_time),
                }
            )
            print(f"  Wrote {output_path.name}", flush=True)
            print(f"  Wrote {image_path.name}", flush=True)

    def _finish(self) -> None:
        summary_path = self.output_directory / "video_metrics_summary.csv"
        with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(self.summary_rows)
        print(f"Wrote {summary_path}", flush=True)
        if self.failures:
            print("Failures:", file=sys.stderr)
            for failure in self.failures:
                print(f"  {failure}", file=sys.stderr)
        self.window.close()
        self.app.exit(1 if self.failures else 0)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export saved comparison-config analysis to Excel-friendly CSV files")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help=f"Config files to process (default: configs/{DEFAULT_CONFIG_PATTERN})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config_paths = arguments.configs or sorted((ROOT / "configs").glob(DEFAULT_CONFIG_PATTERN))
    if not config_paths:
        print("No comparison configs were found.", file=sys.stderr)
        return 2
    missing = [str(path) for path in config_paths if not path.is_file()]
    if missing:
        print("Config files do not exist: " + ", ".join(missing), file=sys.stderr)
        return 2
    configure_logging(arguments.log_level, ROOT / "logs" / "contrast_export.log")
    app = QApplication(sys.argv[:1])
    window = ContrastWindow()
    exporter = ConfigAnalysisExporter(app, window, config_paths, arguments.output_dir)
    exporter.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())