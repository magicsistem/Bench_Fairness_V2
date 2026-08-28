#!/usr/bin/env python3
"""Calibrate D35 and generate independent D33-D42 colour outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from thesis_fitzpatrick.v2 import (atomic_json, clean_skin_mask, colour_metrics, detect_hair_mask,
                                   safety_margin_fraction, sha256_file)


HAIR = {"line_length_fraction": .035, "line_thickness_fraction": .002, "orientations_degrees": [0, 45, 90, 135],
        "response_percentile": 90., "minimum_elongation": 2.5, "maximum_component_thickness_fraction": .025,
        "dilation_fraction": .003, "maximum_coverage_fraction": .18}


def inputs(rois_path: Path, selection_path: Path, results_root: Path) -> tuple[dict, dict, list[str], dict]:
    rois, selection = json.loads(rois_path.read_text(encoding="utf-8")), json.loads(selection_path.read_text(encoding="utf-8"))
    top3 = selection["top3"]; results = {}
    for method in top3:
        report = json.loads((results_root/method/"results.json").read_text(encoding="utf-8"))
        results[method] = {item["image_id"]: item for item in report["records"]}
    return rois, selection, top3, results


def calibrate(rois_path: Path, selection_path: Path, results_root: Path, output: Path) -> None:
    rois, _, top3, results = inputs(rois_path, selection_path, results_root)
    observations = []
    for item in rois["records"]:
        x0, y0, x1, y1 = item["roi_bbox"]
        with Image.open(item["mask"]) as opened: truth = (np.asarray(opened.convert("L")) >= 128)[y0:y1, x0:x1]
        for method in top3:
            with Image.open(results[method][item["image_id"]]["mask"]) as opened:
                pred = (np.asarray(opened.convert("L")) >= 128)[y0:y1, x0:x1]
            if pred.any() and not pred.all():
                observations.append({"image_id": item["image_id"], "method_id": method,
                                     "normalized_q95_distance": safety_margin_fraction(pred, truth)})
    if not observations: raise SystemExit("D35 has no valid TOP-3 observations")
    margin = float(np.quantile([item["normalized_q95_distance"] for item in observations], .95, method="linear"))
    atomic_json(output, {"schema_version": 2, "rule": "D35_TOP3_pair_Q95_then_global_Q95",
                         "lesion_safety_margin_fraction_q95": margin, "observation_count": len(observations),
                         "top3": top3, "selection_sha256": sha256_file(selection_path), "records": observations})


def measure(rois_path: Path, selection_path: Path, results_root: Path, margin_path: Path, output: Path) -> None:
    rois, _, top3, results = inputs(rois_path, selection_path, results_root)
    margin_value = json.loads(margin_path.read_text(encoding="utf-8")); margin = margin_value["lesion_safety_margin_fraction_q95"]
    records = []
    for item in rois["records"]:
        x0, y0, x1, y1 = item["roi_bbox"]
        with Image.open(item["image"]) as opened: rgb = np.asarray(opened.convert("RGB"), np.uint8)[y0:y1, x0:x1]
        hair, hair_meta = detect_hair_mask(rgb, HAIR)
        for method in top3:
            with Image.open(results[method][item["image_id"]]["mask"]) as opened:
                lesion = (np.asarray(opened.convert("L")) >= 128)[y0:y1, x0:x1]
            clean, metadata = clean_skin_mask(rgb, lesion, hair, margin)
            color = None if clean is None else colour_metrics(rgb, clean)
            mask_path = None
            if clean is not None:
                mask_path = output/method/"clean_masks"/f"{item['image_id']}.png"; mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(clean*255).save(mask_path)
            records.append({"image_id": item["image_id"], "method_id": method, "roi_bbox": item["roi_bbox"],
                            "hair": hair_meta, "clean_skin": metadata, "color_metrics": color,
                            "clean_mask": str(mask_path.resolve()) if mask_path else None,
                            "clean_mask_sha256": sha256_file(mask_path) if mask_path else None})
    atomic_json(output/"colorimetry.json", {"schema_version": 2, "top3": top3, "count": len(records),
                                             "margin_sha256": sha256_file(margin_path), "records": records})


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "measure"):
        item = sub.add_parser(name); item.add_argument("--rois", type=Path, required=True)
        item.add_argument("--selection", type=Path, required=True); item.add_argument("--results", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        if name == "measure": item.add_argument("--margin", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate": calibrate(args.rois, args.selection, args.results, args.output)
    else: measure(args.rois, args.selection, args.results, args.margin, args.output)


if __name__ == "__main__": main()
