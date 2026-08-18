from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QColor

from export_config_analysis import (
    ConfigAnalysisExporter,
    NO_QUANTUM_MOTTLE_IMAGE_NOTE,
    lowest_aneurysm_brightness_frame,
    write_video_csv,
    write_video_image,
)
from main import build_analysis_result


class ConfigAnalysisExportTests(unittest.TestCase):
    def test_writes_darkest_aneurysm_frame_with_overlays_and_footer(self) -> None:
        result = build_analysis_result(
            "Pre-deployment",
            Path("example_pre.avi"),
            2.0,
            QRect(0, 0, 1, 1),
            np.asarray([100.0, 40.0, 60.0]),
            np.asarray([], dtype=float),
            0.2,
            False,
        )
        frames = [
            cv2.imencode(".png", np.full((40, 60), level, dtype=np.uint8))[1]
            for level in (80, 100, 120)
        ]
        roi_mask = np.zeros((40, 60), dtype=np.uint8)
        roi_mask[8:18, 8:18] = 2
        needle_mask = np.zeros((40, 60), dtype=np.uint8)
        needle_mask[20:34, 30:44] = 255
        panel = type("Panel", (), {
            "enhanced_frames": frames,
            "report_encoded_frames": None,
            "roi_region_masks": [cv2.imencode(".png", roi_mask)[1]] * 3,
            "needle_segmentation_mask": needle_mask,
            "color": QColor("#38bdf8"),
        })()

        self.assertEqual(lowest_aneurysm_brightness_frame(result), 1)
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "example_pre_analysis.png"
            self.assertTrue(write_video_image(output_path, result, panel))
            image = cv2.imread(str(output_path))

        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual(image.shape[1], 60)
        self.assertGreater(image.shape[0], 40)
        self.assertFalse(np.array_equal(image[10, 10], (100, 100, 100)))
        self.assertGreater(int(image[27, 37, 2]), int(image[27, 37, 0]))
        self.assertGreater(int(np.max(image[40:])), 22)

    def test_writes_no_mottle_note_in_image_footer(self) -> None:
        result = build_analysis_result(
            "Pre-deployment",
            Path("example_pre.avi"),
            2.0,
            QRect(0, 0, 1, 1),
            np.asarray([100.0, 40.0, 60.0]),
            np.asarray([], dtype=float),
            0.2,
            False,
        )
        frames = [cv2.imencode(".png", np.full((80, 320), 100, dtype=np.uint8))[1]] * 3
        panel = type("Panel", (), {
            "enhanced_frames": frames,
            "report_encoded_frames": None,
            "roi_region_masks": None,
            "needle_segmentation_mask": None,
            "color": QColor("#38bdf8"),
        })()

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "example_pre_analysis.png"
            self.assertTrue(write_video_image(output_path, result, panel, note=NO_QUANTUM_MOTTLE_IMAGE_NOTE))
            image = cv2.imread(str(output_path))

        self.assertIsNotNone(image)
        assert image is not None
        self.assertEqual(image.shape[0], 116)
        footer = image[80:]
        self.assertTrue(np.any((footer[:, :, 0] > 50) & (footer[:, :, 1] > 150) & (footer[:, :, 2] > 200)))

    def test_disables_all_quantum_mottle_stages_only(self) -> None:
        class Button:
            def __init__(self, checked: bool) -> None:
                self.checked = checked

            def isChecked(self) -> bool:
                return self.checked

            def blockSignals(self, _blocked: bool) -> None:
                pass

            def setChecked(self, checked: bool) -> None:
                self.checked = checked

        class Drawer:
            def __init__(self, checked: bool) -> None:
                self.enable_button = Button(checked)

        mottle_drawers = [Drawer(True), Drawer(True)]
        unrelated_drawer = Drawer(True)
        window = type("Window", (), {
            "pipeline_rebuilt": False,
            "_stage_drawers": lambda self, key: mottle_drawers if key == "quantum_mottle_filter" else [unrelated_drawer],
            "on_pipeline_stages_changed": lambda self: setattr(self, "pipeline_rebuilt", True),
        })()
        exporter = ConfigAnalysisExporter.__new__(ConfigAnalysisExporter)
        exporter.window = window

        exporter._disable_quantum_mottle_reduction()

        self.assertTrue(window.pipeline_rebuilt)
        self.assertTrue(all(not drawer.enable_button.isChecked() for drawer in mottle_drawers))
        self.assertTrue(unrelated_drawer.enable_button.isChecked())

    def test_writes_one_excel_friendly_row_per_video_frame(self) -> None:
        result = build_analysis_result(
            "Pre-deployment",
            Path("example_pre.avi"),
            2.0,
            QRect(10, 20, 30, 40),
            np.asarray([100.0, 60.0, 95.0]),
            np.asarray([], dtype=float),
            0.2,
            False,
        )
        parent_result = (
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([np.nan, 82.0, 75.0]),
            np.asarray([100.0, 101.5, np.nan]),
        )
        background_result = (
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([120.0, 118.0, np.nan]),
        )
        needle_result = (
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([25.0, np.nan, 27.0]),
        )

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "example.csv"
            write_video_csv(output_path, result, parent_result, background_result, needle_result)
            with output_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["video_file_name"], "example_pre.avi")
        self.assertEqual(
            [row["parent_vessel_roi_apex_average_pixel_brightness"] for row in rows],
            ["100.0", "101.5", ""],
        )
        self.assertEqual(
            [row["parent_vessel_roi_darkest_10_percent_median"] for row in rows],
            ["", "82.0", "75.0"],
        )
        self.assertEqual(rows[0]["aneurysm_residence_time_s"], "0.5")
        self.assertEqual([row["time_s"] for row in rows], ["0.0", "0.5", "1.0"])
        self.assertEqual(
            [row["aneurysm_roi_absolute_average_pixel_darkness"] for row in rows],
            ["100.0", "60.0", "95.0"],
        )
        self.assertEqual(
            [row["background_roi_average_pixel_brightness"] for row in rows],
            ["120.0", "118.0", ""],
        )
        self.assertEqual(
            [row["needle_average_pixel_brightness"] for row in rows],
            ["25.0", "", "27.0"],
        )

    def test_writes_blank_background_column_when_stage_is_not_enabled(self) -> None:
        result = build_analysis_result(
            "Pre-deployment",
            Path("example_pre.avi"),
            2.0,
            QRect(10, 20, 30, 40),
            np.asarray([100.0, 60.0]),
            np.asarray([], dtype=float),
            0.2,
            False,
        )

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "example.csv"
            write_video_csv(output_path, result, None, None, None)
            with output_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            [row["background_roi_average_pixel_brightness"] for row in rows],
            ["", ""],
        )
        self.assertEqual(
            [row["parent_vessel_roi_darkest_10_percent_median"] for row in rows],
            ["", ""],
        )
        self.assertEqual(
            [row["needle_average_pixel_brightness"] for row in rows],
            ["", ""],
        )


if __name__ == "__main__":
    unittest.main()