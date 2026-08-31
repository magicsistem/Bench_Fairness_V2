#!/usr/bin/env python3
"""Create the frozen MST palette and D45-D46 full-ROI synthetic variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000

from thesis_fitzpatrick.v2 import atomic_json, detect_hair_mask, highlight_mask, lab_to_srgb_unclipped, minimum_support_pixels, sha256_file, srgb_to_lab

from colorimetry import HAIR


def palette(config: Path, output: Path) -> None:
    colors = json.loads(config.read_text(encoding="utf-8"))["mst"]
    image = np.zeros((100, 1000, 3), np.uint8)
    for index, (_, red, green, blue) in enumerate(colors): image[:, index*100:(index+1)*100] = (red, green, blue)
    output.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image).save(output, compress_level=0)
    atomic_json(output.with_suffix(".json"), {"schema_version": 2, "colors": colors, "png_sha256": sha256_file(output)})


def generate(rois_path: Path, config: Path, margin_path: Path, output: Path) -> None:
    if "test" not in str(rois_path).lower(): raise SystemExit("MST generation is restricted to sealed Test")
    rois = json.loads(rois_path.read_text(encoding="utf-8")); settings = json.loads(config.read_text(encoding="utf-8")); colors = settings["mst"]
    minimum = settings["minimum_clean_skin"]
    margin = float(json.loads(margin_path.read_text(encoding="utf-8"))["lesion_safety_margin_fraction_q95"])
    targets = {name: srgb_to_lab(np.array(rgb, np.uint8)) for name, *rgb in colors}
    records, unavailable = [], []
    for item in rois["records"]:
        x0, y0, x1, y1 = item["roi_bbox"]
        with Image.open(item["image"]) as opened: full = np.asarray(opened.convert("RGB"), np.uint8)
        with Image.open(item["mask"]) as opened: gt = np.asarray(opened.convert("L")) >= 128
        roi, lesion = full[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        hair, _ = detect_hair_mask(roi, HAIR); radius = int(np.ceil(margin * min(lesion.shape)))
        if radius:
            distance = cv2.distanceTransform((~lesion).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            lesion = distance <= radius
        support = ~lesion & ~hair.astype(bool) & ~highlight_mask(roi).astype(bool)
        support_pixels = int(support.sum())
        required_pixels = minimum_support_pixels(support.size, int(minimum["pixels"]), float(minimum["area_fraction"]))
        if support_pixels < required_pixels:
            unavailable.extend({"image_id": f"{item['image_id']}__{name}", "source_image_id": item["image_id"],
                                "condition": name, "status": "unavailable_insufficient_d45_support",
                                "support_pixels": support_pixels, "required_pixels": required_pixels}
                               for name, *_ in colors)
            continue
        lab = srgb_to_lab(roi); source = np.median(lab[support], axis=0)
        for name, target in targets.items():
            shifted = lab + (target-source); unclipped = lab_to_srgb_unclipped(shifted)
            clipped = np.clip(unclipped, 0, 1); result = np.rint(clipped*255).astype(np.uint8)
            full_result = full.copy(); full_result[y0:y1, x0:x1] = result
            folder = output/item["image_id"]/name; folder.mkdir(parents=True, exist_ok=True)
            path = folder/"image.png"; Image.fromarray(full_result).save(path, compress_level=0)
            measured = np.median(srgb_to_lab(result)[support], axis=0)
            records.append({"image_id": f"{item['image_id']}__{name}", "source_image_id": item["image_id"], "condition": name,
                            "image": str(path.resolve()), "mask": item["mask"], "mask_sha256": item["mask_sha256"],
                            "width": item["width"], "height": item["height"], "roi_bbox": item["roi_bbox"],
                            "detector_status": item["detector_status"],
                            "image_sha256": sha256_file(path), "source_lab": source.tolist(), "target_lab": target.tolist(),
                            "measured_lab": measured.tolist(), "delta_lab": (target-source).tolist(),
                            "ciede2000": float(deltaE_ciede2000(target[None, :], measured[None, :])[0]),
                            "clipped_channel_fraction": float(np.mean((unclipped <= 0) | (unclipped >= 1))),
                            "shape": list(result.shape), "gt_sha256": item["mask_sha256"]})
    atomic_json(output/"manifest.json", {"schema_version": 2, "split": "test_mst", "count": len(records),
                                          "unavailable_count": len(unavailable), "expected_count": rois["count"] * len(colors),
                                          "source_count": rois["count"], "available_source_count": len(records) // len(colors),
                                          "unavailable_source_count": len(unavailable) // len(colors),
                                          "conditions": [name for name, *_ in colors], "records": records,
                                          "unavailable_records": unavailable})


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("palette"); one.add_argument("--config", type=Path, required=True); one.add_argument("--output", type=Path, required=True)
    two = sub.add_parser("generate"); two.add_argument("--rois", type=Path, required=True); two.add_argument("--config", type=Path, required=True); two.add_argument("--margin", type=Path, required=True); two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); palette(args.config, args.output) if args.command == "palette" else generate(args.rois, args.config, args.margin, args.output)


if __name__ == "__main__": main()
