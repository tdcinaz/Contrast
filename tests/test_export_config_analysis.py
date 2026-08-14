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
    lowest_aneurysm_brightness_frame,
    parent_vessel_minimum,
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
            minimum = parent_vessel_minimum(parent_result)
            write_video_csv(output_path, result, minimum, parent_result, background_result, needle_result)
            with output_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(minimum, 75.0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["video_file_name"], "example_pre.avi")
        self.assertEqual(rows[0]["parent_vessel_roi_apex_min_level"], "75.0")
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
            write_video_csv(output_path, result, None, None, None, None)
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