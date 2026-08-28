#!/usr/bin/env python3
"""Calibrate D09 from OOF YOLO labels and create deterministic ROI manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_fitzpatrick.v2 import atomic_json, sha256_file


def detections(path: Path, width: int, height: int) -> list[tuple[float, list[int]]]:
    found = []
    if not path.is_file():
        return found
    for line in path.read_text(encoding="ascii").splitlines():
        values = [float(value) for value in line.split()]
        if len(values) not in (5, 6) or int(values[0]) != 0:
            raise SystemExit(f"invalid YOLO label: {path}")
        _, cx, cy, w, h, *confidence = values
        box = [max(0, round((cx - w / 2) * width)), max(0, round((cy - h / 2) * height)),
               min(width, round((cx + w / 2) * width)), min(height, round((cy + h / 2) * height))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise SystemExit(f"empty YOLO box: {path}")
        found.append((confidence[0] if confidence else 1.0, box))
    return sorted(found, reverse=True)


def expanded(box: list[int], margin: float, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    dx, dy = margin * (x1 - x0), margin * (y1 - y0)
    return [max(0, int(np.floor(x0 - dx))), max(0, int(np.floor(y0 - dy))),
            min(width, int(np.ceil(x1 + dx))), min(height, int(np.ceil(y1 + dy)))]


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("records"):
        raise SystemExit(f"empty manifest: {path}")
    return value


def calibrate(manifest_path: Path, folds_path: Path, labels_root: Path, output: Path) -> None:
    manifest, folds = load_manifest(manifest_path), json.loads(folds_path.read_text(encoding="utf-8"))
    margins, records = [], []
    for item in manifest["records"]:
        identifier, width, height = item["image_id"], item["width"], item["height"]
        fold = folds["assignments"][identifier]
        label = labels_root / f"fold-{fold}" / "labels" / f"{identifier}.txt"
        boxes = detections(label, width, height)
        if not boxes:
            records.append({"image_id": identifier, "fold": fold, "status": "valid_no_detection"})
            continue
        confidence, box = boxes[0]
        gx0, gy0, gx1, gy1 = item["bbox_xyxy_half_open"]
        x0, y0, x1, y1 = box
        value = max(0, (x0-gx0)/width, (gx1-x1)/width, (y0-gy0)/height, (gy1-y1)/height)
        margins.append(value)
        records.append({"image_id": identifier, "fold": fold, "status": "detected", "confidence": confidence,
                        "selected_bbox": box, "minimum_margin": value})
    if not margins:
        raise SystemExit("D09 requires at least one valid OOF detection")
    margin = float(np.quantile(margins, 0.95, method="linear"))
    atomic_json(output, {"schema_version": 2, "rule": "D09_Q95_valid_OOF", "margin_fraction": margin,
                         "valid_detection_count": len(margins), "no_detection_count": len(records)-len(margins),
                         "minimum_margin_summary": {"median": float(np.median(margins)), "q95": margin},
                         "manifest_sha256": sha256_file(manifest_path), "folds_sha256": sha256_file(folds_path),
                         "records": records})


def build(manifest_path: Path, labels_root: Path, margin_path: Path, output: Path) -> None:
    manifest = load_manifest(manifest_path)
    margin_value = json.loads(margin_path.read_text(encoding="utf-8"))
    margin = float(margin_value["margin_fraction"])
    records = []
    for item in manifest["records"]:
        identifier, width, height = item["image_id"], item["width"], item["height"]
        boxes = detections(labels_root / f"{identifier}.txt", width, height)
        if boxes:
            confidence, box = boxes[0]
            roi, status = expanded(box, margin, width, height), "detected"
        else:
            confidence, box, roi, status = None, None, [0, 0, width, height], "valid_no_detection"
        records.append({**item, "detector_status": status, "selected_bbox": box, "confidence": confidence,
                        "roi_bbox": roi, "margin_fraction": margin})
    atomic_json(output, {"schema_version": 2, "split": manifest["split"], "count": len(records),
                         "manifest_sha256": sha256_file(manifest_path), "margin_sha256": sha256_file(margin_path),
                         "records": records})


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("calibrate")
    one.add_argument("--manifest", type=Path, required=True); one.add_argument("--folds", type=Path, required=True)
    one.add_argument("--labels-root", type=Path, required=True); one.add_argument("--output", type=Path, required=True)
    two = commands.add_parser("build")
    two.add_argument("--manifest", type=Path, required=True); two.add_argument("--labels-root", type=Path, required=True)
    two.add_argument("--margin", type=Path, required=True); two.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate": calibrate(args.manifest, args.folds, args.labels_root, args.output)
    else: build(args.manifest, args.labels_root, args.margin, args.output)


if __name__ == "__main__":
    main()
