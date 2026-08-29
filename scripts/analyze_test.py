#!/usr/bin/env python3
"""D54 real-image robustness using independent GT-supported continuous ITA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from colorimetry import HAIR
from thesis_fitzpatrick.v2 import atomic_json, colour_metrics, detect_hair_mask, highlight_mask


def run(rois_path: Path, selection_path: Path, results_root: Path, margin_path: Path, output: Path, seed: int) -> None:
    rois = json.loads(rois_path.read_text(encoding="utf-8")); top3 = json.loads(selection_path.read_text(encoding="utf-8"))["top3"]
    margin = json.loads(margin_path.read_text(encoding="utf-8"))["lesion_safety_margin_fraction_q95"]
    tone = {}
    for item in rois["records"]:
        x0, y0, x1, y1 = item["roi_bbox"]
        with Image.open(item["image"]) as opened: rgb = np.asarray(opened.convert("RGB"), np.uint8)[y0:y1, x0:x1]
        with Image.open(item["mask"]) as opened: gt = (np.asarray(opened.convert("L")) >= 128)[y0:y1, x0:x1]
        radius = int(np.ceil(margin*min(gt.shape)))
        if radius:
            distance = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE); gt = distance <= radius
        hair, _ = detect_hair_mask(rgb, HAIR); support = ~gt & ~hair.astype(bool) & ~highlight_mask(rgb).astype(bool)
        required = max(256, int(np.ceil(.005*support.size)))
        tone[item["image_id"]] = None if support.sum() < required else colour_metrics(rgb, support.astype(np.uint8))["ita_degrees"]
    reports, generator = [], np.random.default_rng(seed)
    for method in top3:
        records = json.loads((results_root/method/"results.json").read_text(encoding="utf-8"))["records"]
        usable = [(tone[r["image_id"]], r) for r in records if tone[r["image_id"]] is not None]
        if not usable: raise SystemExit(f"no independent ITA observations for {method}")
        item = {"method_id": method, "n": len(usable), "associations": {}, "metric_summaries": {}}
        ita = np.array([x[0] for x in usable]); indices = generator.integers(0, len(usable), size=(10_000, len(usable)))
        for metric in ("jaccard", "dice", "boundary_f1"):
            values = np.array([x[1]["metrics"][metric] for x in usable])
            rho = float(spearmanr(ita, values).statistic)
            boot = np.array([spearmanr(ita[index], values[index]).statistic for index in indices])
            boot = boot[np.isfinite(boot)]
            item["associations"][metric] = {"spearman_rho": rho, "ci95": [float(x) for x in np.percentile(boot, (2.5, 97.5))] if len(boot) else None}
            means = values[indices].mean(axis=1)
            item["metric_summaries"][metric] = {"mean": float(values.mean()), "ci95": [float(x) for x in np.percentile(means, (2.5, 97.5))]}
        reports.append(item)
    atomic_json(output, {"schema_version": 2, "interpretation": "robustness versus image-estimated continuous ITA; not demographic fairness",
                         "top3": top3, "tone_by_image": tone, "methods": reports, "bootstrap_repetitions": 10_000, "seed": seed})


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--rois", type=Path, required=True); p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True); p.add_argument("--margin", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260828); a = p.parse_args(); run(a.rois, a.selection, a.results, a.margin, a.output, a.seed)


if __name__ == "__main__": main()
