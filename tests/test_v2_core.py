import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from thesis_fitzpatrick.v2 import (
    BBox,
    atomic_json,
    clean_skin_mask,
    colour_metrics,
    highlight_mask,
    lab_to_srgb_unclipped,
    minimum_support_pixels,
    q95,
    required_symmetric_margin,
    srgb_to_lab,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from roi import build, build_no_ground_truth, expanded  # noqa: E402
from scripts.prepare_data import assert_development_path, folds


class V2CoreTest(unittest.TestCase):
    def test_d41_lab_inverse_round_trip(self):
        rgb = np.array([[[32, 128, 224], [220, 180, 80]]], dtype=np.uint8)
        restored = lab_to_srgb_unclipped(srgb_to_lab(rgb))
        np.testing.assert_allclose(restored, rgb / 255.0, atol=2e-6)
    def test_roi_margin_expands_symmetrically_and_clips(self):
        self.assertEqual(expanded([10, 20, 30, 40], 0.5, 35, 45), [0, 10, 35, 45])

    def test_no_detection_uses_full_condition_image(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); labels = root/"labels"; labels.mkdir()
            mask = root/"mask.png"; Image.fromarray(np.pad(np.ones((2, 2), np.uint8)*255, 1)).save(mask)
            manifest = root/"manifest.json"; manifest.write_text(json.dumps({"split": "test_mst", "records": [{"image_id": "I__MST_01", "width": 4, "height": 4, "mask": str(mask), "bbox_xyxy_half_open": [1, 1, 3, 3]}]}))
            margin = root/"margin.json"; margin.write_text(json.dumps({"margin_fraction": .1}))
            output = root/"rois.json"; build(manifest, labels, margin, output)
            record = json.loads(output.read_text())["records"][0]
            self.assertEqual(record["detector_status"], "valid_no_detection")
            self.assertEqual(record["roi_bbox"], [0, 0, 4, 4])

    def test_mskcc_no_ground_truth_detected_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); labels = root/"labels"; labels.mkdir()
            (labels/"MSK_DETECTED.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="ascii")
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"split": "mskcc_census", "records": [
                {"image_id": "MSK_DETECTED", "image": "/tmp/MSK_DETECTED.jpg", "width": 100, "height": 100,
                 "patient_id": "P1", "lesion_id": "L1"}]}))
            margin = root/"margin.json"; margin.write_text(json.dumps({"margin_fraction": .1}))
            output = root/"rois.json"; build_no_ground_truth(manifest, labels, margin, output)
            value = json.loads(output.read_text()); record = value["records"][0]
            self.assertEqual(record["detector_status"], "detected")
            self.assertEqual(record["selected_bbox"], [40, 40, 60, 60])
            self.assertEqual(record["roi_bbox"], [38, 38, 62, 62])
            self.assertIsNone(record["bbox_containment"])
            self.assertIsNone(record["lesion_pixel_containment"])
            self.assertFalse(value["ground_truth_available"])

    def test_mskcc_no_ground_truth_no_detection_uses_full_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); labels = root/"labels"; labels.mkdir()
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"split": "mskcc_census", "records": [
                {"image_id": "MSK_NO_DETECTION", "image": "/tmp/MSK_NO_DETECTION.jpg", "width": 80, "height": 60,
                 "patient_id": "P2", "lesion_id": "L2"}]}))
            margin = root/"margin.json"; margin.write_text(json.dumps({"margin_fraction": .1}))
            output = root/"rois.json"; build_no_ground_truth(manifest, labels, margin, output)
            value = json.loads(output.read_text()); record = value["records"][0]
            self.assertEqual(record["detector_status"], "valid_no_detection")
            self.assertIsNone(record["selected_bbox"])
            self.assertEqual(record["roi_bbox"], [0, 0, 80, 60])
            self.assertFalse(value["ground_truth_available"])

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

    def test_d37_minimum_support(self):
        self.assertEqual(minimum_support_pixels(8 * 8), 256)
        self.assertEqual(minimum_support_pixels(100_001), 501)

    def test_mst_transforms_full_image_and_writes_lossless_png(self):
        from PIL import Image
        try:
            import mst
        except ModuleNotFoundError as error:
            if error.name != "skimage": raise
            color = types.ModuleType("skimage.color"); color.deltaE_ciede2000 = lambda a, b: np.linalg.norm(a-b, axis=-1)
            package = types.ModuleType("skimage"); package.color = color
            sys.modules["skimage"], sys.modules["skimage.color"] = package, color
            import mst
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = np.full((12, 12, 3), 120, np.uint8); truth = np.zeros((12, 12), np.uint8); truth[5:7, 5:7] = 255
            image_path, mask_path = root/"source.jpg", root/"mask.png"
            Image.fromarray(image).save(image_path); Image.fromarray(truth).save(mask_path)
            config = root/"config.json"; config.write_text(json.dumps({"mst": [["MST_01", 40, 35, 30]], "minimum_clean_skin": {"pixels": 1, "area_fraction": 0}}))
            margin = root/"margin.json"; margin.write_text(json.dumps({"lesion_safety_margin_fraction_q95": 0}))
            rois = root/"Test_rois.json"; rois.write_text(json.dumps({"count": 1, "records": [{"image_id": "I1", "image": str(image_path), "mask": str(mask_path), "mask_sha256": "x", "width": 12, "height": 12, "bbox_xyxy_half_open": [5, 5, 7, 7], "roi_bbox": [4, 4, 8, 8]}]}))
            original_detector = mst.detect_hair_mask
            try:
                mst.detect_hair_mask = lambda rgb, _: (np.zeros(rgb.shape[:2], np.uint8), {})
                mst.generate(rois, config, margin, root/"mst", workers=2)
                mst.generate(rois, config, margin, root/"mst", workers=2)
            finally: mst.detect_hair_mask = original_detector
            manifest = json.loads((root/"mst/manifest.json").read_text()); record = manifest["records"][0]
            with Image.open(record["image"]) as opened: result = np.asarray(opened.convert("RGB"))
            self.assertEqual(result.shape, image.shape)
            self.assertFalse(np.array_equal(result[0, 0], image[0, 0]))
            self.assertEqual(manifest["synthesis_domain"], "full_image")
            self.assertEqual(manifest["png_compress_level"], 6)
            self.assertTrue(record["reused_png"])

    def test_atomic_json_is_sorted_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            atomic_json(path, {"b": 2, "a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1, "b": 2})

    def test_segmenter_progress_requires_matching_mask_hash(self):
        from evaluate_segmenter import append_progress, load_progress
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); mask = root / "mask.png"; mask.write_bytes(b"valid")
            progress = root / "progress.jsonl"
            progress.write_text(json.dumps({"method_id": "m", "roi_manifest_sha256": "r", "schema_version": 2}) + "\n")
            record = {"image_id": "I1", "mask": str(mask), "mask_sha256": __import__("hashlib").sha256(b"valid").hexdigest()}
            append_progress(progress, record)
            self.assertEqual(load_progress(progress, "m", "r")["I1"], record)
            mask.write_bytes(b"partial")
            self.assertEqual(load_progress(progress, "m", "r"), {})

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
