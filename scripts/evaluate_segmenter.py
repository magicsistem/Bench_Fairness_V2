#!/usr/bin/env python3
"""Run one V2 segmenter on a frozen ROI manifest and evaluate without mask cleanup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from thesis_fitzpatrick.metrics import segmentation_metrics
from thesis_fitzpatrick.v2 import atomic_json, sha256_file


def roster(path: Path) -> list[dict]:
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    return [*models, {"id": "grabcut", "name": "GrabCut", "adapter_command": None}]


def grabcut(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    mx, my = max(1, round(width * .05)), max(1, round(height * .05))
    if width - 2*mx < 2 or height - 2*my < 2:
        raise RuntimeError("ROI too small for GrabCut")
    labels = np.zeros((height, width), np.uint8)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, (mx, my, width-2*mx, height-2*my),
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_RECT)
    return np.isin(labels, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)


def command(model: dict, root: Path, image: Path, output: Path, device: str) -> tuple[list[str], Path | None]:
    values = {"repo_root": str(root), "image": str(image), "lesion_mask": str(output), "device": device,
              "conda": "conda"}
    raw = [part.format(**values) for part in model["adapter_command"]]
    if "python" not in raw:
        raise RuntimeError(f"adapter has no python entry: {model['id']}")
    raw = [sys.executable, *raw[raw.index("python") + 1:]]
    checkpoint = next((Path(raw[index+1]) for index, part in enumerate(raw[:-1]) if part == "--checkpoint"), None)
    return raw, checkpoint


def run(args: argparse.Namespace) -> None:
    root = args.repo.resolve(); methods = roster(root / "configs/segmentation_models.json")
    if not 0 <= args.method_index < len(methods): raise SystemExit("method-index outside 0..15")
    model = methods[args.method_index]
    roi_manifest = json.loads(args.rois.read_text(encoding="utf-8"))
    out = args.output / model["id"]; (out / "masks").mkdir(parents=True, exist_ok=True)
    records = []
    for item in roi_manifest["records"]:
        identifier = item["image_id"]; started = time.perf_counter()
        with Image.open(item["image"]) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        x0, y0, x1, y1 = item["roi_bbox"]; crop = rgb[y0:y1, x0:x1]
        if crop.size == 0: raise RuntimeError(f"empty ROI: {identifier}")
        with tempfile.TemporaryDirectory(prefix="v2-roi-") as temporary:
            roi_path, native_path = Path(temporary)/"roi.png", Path(temporary)/"mask.png"
            Image.fromarray(crop).save(roi_path)
            checkpoint = None
            if model["id"] == "grabcut": native = grabcut(crop)
            else:
                call, checkpoint = command(model, root, roi_path, native_path, args.device)
                environment = {**os.environ, "THESIS_ADAPTER_METRICS_PATH": str(Path(temporary)/"runtime.json")}
                subprocess.run(call, check=True, cwd=root, env=environment)
                with Image.open(native_path) as opened: native = (np.asarray(opened.convert("L")) >= 128).astype(np.uint8)
        if native.shape != crop.shape[:2]: raise RuntimeError(f"native shape mismatch: {identifier}")
        restored = np.zeros(rgb.shape[:2], np.uint8); restored[y0:y1, x0:x1] = native
        mask_path = out / "masks" / f"{identifier}.png"; Image.fromarray(restored * 255).save(mask_path)
        with Image.open(item["mask"]) as opened: truth = np.asarray(opened.convert("L")) >= 128
        metrics = segmentation_metrics(restored, truth)
        records.append({"image_id": identifier, "method_id": model["id"], "status": "complete",
                        "roi_bbox": item["roi_bbox"], "detector_status": item["detector_status"],
                        "mask": str(mask_path.resolve()), "mask_sha256": sha256_file(mask_path),
                        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
                        "elapsed_ms": (time.perf_counter()-started)*1000, "metrics": metrics})
    atomic_json(out / "results.json", {"schema_version": 2, "method_id": model["id"], "count": len(records),
                                        "roi_manifest_sha256": sha256_file(args.rois), "records": records})
    print(f"V2_SEGMENTER_COMPLETE={model['id']} count={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(".")); parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--method-index", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__": main()
