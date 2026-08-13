from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dicom_source import DicomVideoCapture, is_dicom_video, probe_dicom_video


class DicomSourceTests(unittest.TestCase):
    def _write_cine(self, path: Path) -> np.ndarray:
        frames = np.asarray(
            [
                [[40, 50, 99], [100, 149, 160]],
                [[60, 70, 109], [110, 139, 150]],
            ],
            dtype=np.uint16,
        )
        sop_class_uid = "1.2.840.10008.5.1.4.1.1.12.2"
        sop_instance_uid = generate_uid()
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = sop_class_uid
        dataset.SOPInstanceUID = sop_instance_uid
        dataset.Rows = frames.shape[1]
        dataset.Columns = frames.shape[2]
        dataset.NumberOfFrames = frames.shape[0]
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.CineRate = 15
        dataset.WindowCenter = 100
        dataset.WindowWidth = 100
        dataset.PixelData = frames.tobytes()
        dataset.save_as(path, enforce_file_format=True)
        return frames

    def test_probe_and_capture_multiframe_dicom_with_fixed_window(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "IMG1"
            self._write_cine(path)

            self.assertTrue(is_dicom_video(path))
            info = probe_dicom_video(path)
            self.assertEqual((info.frame_count, info.width, info.height), (2, 3, 2))
            self.assertEqual(info.fps, 15.0)
            self.assertEqual((info.window_center, info.window_width), (100.0, 100.0))

            capture = DicomVideoCapture(path)
            ok, first = capture.read()
            self.assertTrue(ok)
            assert first is not None
            expected_first = np.asarray([[0, 0, 126], [128, 255, 255]], dtype=np.uint8)
            np.testing.assert_array_equal(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), expected_first)
            self.assertEqual(capture.get(cv2.CAP_PROP_POS_FRAMES), 1.0)

            ok, second = capture.read()
            self.assertTrue(ok)
            assert second is not None
            self.assertFalse(np.array_equal(first, second))
            self.assertFalse(capture.read()[0])

            self.assertTrue(capture.set(cv2.CAP_PROP_POS_FRAMES, 0))
            ok, repeated = capture.read()
            self.assertTrue(ok)
            assert repeated is not None
            np.testing.assert_array_equal(repeated, first)


if __name__ == "__main__":
    unittest.main()