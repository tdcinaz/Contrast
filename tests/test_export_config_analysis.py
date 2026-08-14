from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PySide6.QtCore import QRect

from export_config_analysis import parent_vessel_minimum, write_video_csv
from main import build_analysis_result


class ConfigAnalysisExportTests(unittest.TestCase):
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

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "example.csv"
            minimum = parent_vessel_minimum(parent_result)
            write_video_csv(output_path, result, minimum, background_result)
            with output_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(minimum, 75.0)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["video_file_name"], "example_pre.avi")
        self.assertEqual(rows[0]["parent_vessel_roi_apex_min_level"], "75.0")
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
            write_video_csv(output_path, result, None, None)
            with output_path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            [row["background_roi_average_pixel_brightness"] for row in rows],
            ["", ""],
        )


if __name__ == "__main__":
    unittest.main()