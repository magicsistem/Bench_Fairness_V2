import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_fitzpatrick.v2 import (
    BBox,
    atomic_json,
    clean_skin_mask,
    colour_metrics,
    highlight_mask,
    q95,
    required_symmetric_margin,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from roi import expanded  # noqa: E402
from scripts.prepare_data import assert_development_path, folds


class V2CoreTest(unittest.TestCase):
    def test_roi_margin_expands_symmetrically_and_clips(self):
        self.assertEqual(expanded([10, 20, 30, 40], 0.5, 35, 45), [0, 10, 35, 45])
    def test_margin_is_minimum_symmetric_fraction(self):
        self.assertEqual(required_symmetric_margin(BBox(10, 10, 30, 50), BBox(8, 9, 34, 55)), 0.2)
        self.assertEqual(q95([0.0] * 19 + [1.0]), 0.05000000000000071)

    def test_clean_skin_has_no_fallback_or_highlight(self):
        rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
        rgb[0, 0] = 255
        lesion = np.zeros((40, 40), dtype=np.uint8)
        lesion[15:25, 15:25] = 1
        hair = np.zeros_like(lesion)
        hair[:, 5] = 1
        clean, metadata = clean_skin_mask(rgb, lesion, hair, 0.05, minimum_pixels=1)
        self.assertEqual(metadata["color_status"], "available")
        self.assertFalse(clean[0, 0])
        self.assertFalse(clean[:, 5].any())
        self.assertFalse(clean[13:27, 15:25].any())
        self.assertFalse(clean[15:25, 13:27].any())
        stats = colour_metrics(rgb, clean)
        self.assertAlmostEqual(stats["rgb_median"][0], 100.0)

    def test_degenerate_masks_are_unavailable(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        clean, metadata = clean_skin_mask(rgb, np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8), 0)
        self.assertIsNone(clean)
        self.assertEqual(metadata["color_status"], "unavailable_degenerate_lesion_mask")

    def test_highlight_rule_is_not_dilated(self):
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        rgb[1, 1] = [248, 240, 242]
        mask = highlight_mask(rgb)
        self.assertEqual(int(mask.sum()), 1)

    def test_atomic_json_is_sorted_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})

    def test_test_paths_are_rejected_before_freeze(self):
        with self.assertRaises(SystemExit):
            assert_development_path(Path("/sealed/ISIC2018_Test_Input"))

    def test_folds_are_deterministic_and_complete_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"count": 11, "records": [{"image_id": f"I{i}"} for i in range(11)]}))
            first, second = root / "first.json", root / "second.json"
            folds(manifest, first, None, 7); folds(manifest, second, None, 7)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            value = json.loads(first.read_text())
            self.assertEqual(len(value["assignments"]), 11)
            self.assertLessEqual(max(value["counts"]) - min(value["counts"]), 1)


if __name__ == "__main__":
    unittest.main()
