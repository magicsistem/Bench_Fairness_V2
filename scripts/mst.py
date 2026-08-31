#!/usr/bin/env python3
"""Create the frozen MST palette and D45-D46 full-image synthetic variants."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000

from thesis_fitzpatrick.v2 import atomic_json, detect_hair_mask, highlight_mask, lab_to_srgb_unclipped, minimum_support_pixels, sha256_file, srgb_to_lab

from colorimetry import HAIR


def pixel_sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def configure_worker(opencv_threads: int) -> None:
    cv2.setNumThreads(opencv_threads)


def save_or_validate_png(path: Path, rgb: np.ndarray) -> bool:
    expected = pixel_sha256(rgb)
    if path.is_file():
        try:
            with Image.open(path) as opened: decoded = np.asarray(opened.convert("RGB"), np.uint8)
            if decoded.shape == rgb.shape and pixel_sha256(decoded) == expected:
                return True
        except OSError:
            pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(rgb).save(temporary, format="PNG", compress_level=6)
    os.replace(temporary, path)
    with Image.open(path) as opened: decoded = np.asarray(opened.convert("RGB"), np.uint8)
    if decoded.shape != rgb.shape or pixel_sha256(decoded) != expected:
        raise RuntimeError(f"lossless PNG invariant failed: {path}")
    return False


def generate_source(item: dict, colors: list, targets: dict, minimum: dict, margin: float, output: Path) -> tuple[list, list]:
    x0, y0, x1, y1 = item["roi_bbox"]
    with Image.open(item["image"]) as opened: full = np.asarray(opened.convert("RGB"), np.uint8)
    with Image.open(item["mask"]) as opened: gt = np.asarray(opened.convert("L")) >= 128
    lesion = gt.copy(); hair, _ = detect_hair_mask(full, HAIR); radius = int(np.ceil(margin * min(y1-y0, x1-x0)))
    if radius:
        distance = cv2.distanceTransform((~lesion).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        lesion = distance <= radius
    support = ~lesion & ~hair.astype(bool) & ~highlight_mask(full).astype(bool)
    support_pixels = int(support.sum())
    required_pixels = minimum_support_pixels(support.size, int(minimum["pixels"]), float(minimum["area_fraction"]))
    if support_pixels < required_pixels:
        return [], [{"image_id": f"{item['image_id']}__{name}", "source_image_id": item["image_id"],
                     "condition": name, "status": "unavailable_insufficient_d45_support",
                     "support_pixels": support_pixels, "required_pixels": required_pixels} for name, *_ in colors]
    lab = srgb_to_lab(full); source = np.median(lab[support], axis=0); records = []
    for name, target in targets.items():
        shifted = lab + (target-source); unclipped = lab_to_srgb_unclipped(shifted)
        clipped = np.clip(unclipped, 0, 1); result = np.rint(clipped*255).astype(np.uint8)
        path = output/"images"/f"{item['image_id']}__{name}.png"; reused = save_or_validate_png(path, result)
        measured = np.median(srgb_to_lab(result)[support], axis=0)
        records.append({"image_id": f"{item['image_id']}__{name}", "source_image_id": item["image_id"], "condition": name,
                        "image": str(path.resolve()), "mask": item["mask"], "mask_sha256": item["mask_sha256"],
                        "width": item["width"], "height": item["height"],
                        "bbox_xyxy_half_open": item["bbox_xyxy_half_open"], "original_roi_bbox": item["roi_bbox"],
                        "image_sha256": sha256_file(path), "source_lab": source.tolist(), "target_lab": target.tolist(),
                        "measured_lab": measured.tolist(), "delta_lab": (target-source).tolist(),
                        "ciede2000": float(deltaE_ciede2000(target[None, :], measured[None, :])[0]),
                        "clipped_channel_fraction": float(np.mean((unclipped <= 0) | (unclipped >= 1))),
                        "shape": list(result.shape), "pixel_sha256": pixel_sha256(result),
                        "gt_sha256": item["mask_sha256"], "reused_png": reused})
    return records, []


def palette(config: Path, output: Path) -> None:
    colors = json.loads(config.read_text(encoding="utf-8"))["mst"]
    image = np.zeros((100, 1000, 3), np.uint8)
    for index, (_, red, green, blue) in enumerate(colors): image[:, index*100:(index+1)*100] = (red, green, blue)
    output.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(image).save(output, compress_level=6)
    atomic_json(output.with_suffix(".json"), {"schema_version": 2, "colors": colors, "png_sha256": sha256_file(output)})


def generate(rois_path: Path, config: Path, margin_path: Path, output: Path, workers: int = 1) -> None:
    if "test" not in str(rois_path).lower(): raise SystemExit("MST generation is restricted to sealed Test")
    rois = json.loads(rois_path.read_text(encoding="utf-8")); settings = json.loads(config.read_text(encoding="utf-8")); colors = settings["mst"]
    minimum = settings["minimum_clean_skin"]
    margin = float(json.loads(margin_path.read_text(encoding="utf-8"))["lesion_safety_margin_fraction_q95"])
    targets = {name: srgb_to_lab(np.array(rgb, np.uint8)) for name, *rgb in colors}
    if workers < 1: raise SystemExit("workers must be positive")
    (output/"images").mkdir(parents=True, exist_ok=True)
    opencv_threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) // workers)
    records, unavailable = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=configure_worker,
                                                initargs=(opencv_threads,)) as executor:
        futures = [executor.submit(generate_source, item, colors, targets, minimum, margin, output) for item in rois["records"]]
        for future in concurrent.futures.as_completed(futures):
            complete, missing = future.result(); records.extend(complete); unavailable.extend(missing)
    records.sort(key=lambda row: row["image_id"]); unavailable.sort(key=lambda row: row["image_id"])
    atomic_json(output/"manifest.json", {"schema_version": 2, "split": "test_mst", "count": len(records),
                                          "synthesis_domain": "full_image", "png_compress_level": 6,
                                          "unavailable_count": len(unavailable), "expected_count": rois["count"] * len(colors),
                                          "source_count": rois["count"], "available_source_count": len(records) // len(colors),
                                          "unavailable_source_count": len(unavailable) // len(colors),
                                          "conditions": [name for name, *_ in colors], "records": records,
                                          "unavailable_records": unavailable})


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("palette"); one.add_argument("--config", type=Path, required=True); one.add_argument("--output", type=Path, required=True)
    two = sub.add_parser("generate"); two.add_argument("--rois", type=Path, required=True); two.add_argument("--config", type=Path, required=True); two.add_argument("--margin", type=Path, required=True); two.add_argument("--output", type=Path, required=True); two.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); palette(args.config, args.output) if args.command == "palette" else generate(args.rois, args.config, args.margin, args.output, args.workers)


if __name__ == "__main__": main()
