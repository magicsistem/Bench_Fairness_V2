#!/usr/bin/env python3
"""Run one V2 segmenter on a frozen ROI manifest and evaluate without mask cleanup."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
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


def adapter(model: dict, root: Path, device: str):
    values = {"repo_root": str(root), "image": "{image}", "lesion_mask": "{output}", "device": device, "conda": "conda"}
    raw = [part.format(**values) for part in model["adapter_command"]]
    if "python" not in raw:
        raise RuntimeError(f"adapter has no python entry: {model['id']}")
    raw = raw[raw.index("python") + 1:]; script = Path(raw[0]); options = dict(zip(raw[1::2], raw[2::2]))
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location(f"v2_adapter_{model['id'].replace('-', '_')}", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.load_model = functools.lru_cache(maxsize=1)(module.load_model)
    checkpoint = Path(options["--checkpoint"]) if "--checkpoint" in options else None
    def infer(image: Path, output: Path) -> None:
        kwargs = {"image_path": image, "output_path": output, "device": device}
        for option, name in (("--source", "source"), ("--checkpoint", "checkpoint"), ("--dataset", "dataset"), ("--variant", "variant")):
            if option in options: kwargs[name] = Path(options[option]) if option in ("--source", "--checkpoint") else options[option]
        module.infer(**kwargs)
    return infer, checkpoint


def run(args: argparse.Namespace) -> None:
    root = args.repo.resolve(); methods = roster(root / "configs/segmentation_models.json")
    if args.method_id:
        matches = [item for item in methods if item["id"] == args.method_id]
        if len(matches) != 1: raise SystemExit(f"unknown method-id: {args.method_id}")
        model = matches[0]
    else:
        if args.method_index is None or not 0 <= args.method_index < len(methods): raise SystemExit("method-index outside 0..15")
        model = methods[args.method_index]
    roi_manifest = json.loads(args.rois.read_text(encoding="utf-8"))
    out = args.output / model["id"]; (out / "masks").mkdir(parents=True, exist_ok=True)
    records = []; infer_adapter, checkpoint = (None, None) if model["id"] == "grabcut" else adapter(model, root, args.device)
    for item in roi_manifest["records"]:
        identifier = item["image_id"]; started = time.perf_counter()
        with Image.open(item["image"]) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        x0, y0, x1, y1 = item["roi_bbox"]; crop = rgb[y0:y1, x0:x1]
        if crop.size == 0: raise RuntimeError(f"empty ROI: {identifier}")
        with tempfile.TemporaryDirectory(prefix="v2-roi-") as temporary:
            roi_path, native_path = Path(temporary)/"roi.png", Path(temporary)/"mask.png"
            Image.fromarray(crop).save(roi_path)
            if model["id"] == "grabcut": native = grabcut(crop)
            else:
                os.environ["THESIS_ADAPTER_METRICS_PATH"] = str(Path(temporary)/"runtime.json")
                infer_adapter(roi_path, native_path)
                with Image.open(native_path) as opened: native = (np.asarray(opened.convert("L")) >= 128).astype(np.uint8)
        if native.shape != crop.shape[:2]: raise RuntimeError(f"native shape mismatch: {identifier}")
        restored = np.zeros(rgb.shape[:2], np.uint8); restored[y0:y1, x0:x1] = native
        mask_path = out / "masks" / f"{identifier}.png"; Image.fromarray(restored * 255).save(mask_path)
        metrics = None
        if not args.no_ground_truth:
            with Image.open(item["mask"]) as opened: truth = np.asarray(opened.convert("L")) >= 128
            metrics = segmentation_metrics(restored, truth)
        records.append({"image_id": identifier, "method_id": model["id"], "status": "complete",
                        "roi_bbox": item["roi_bbox"], "detector_status": item["detector_status"],
                        "selected_bbox": item.get("selected_bbox"), "confidence": item.get("confidence"),
                        "bbox_containment": item.get("bbox_containment"),
                        "lesion_pixel_containment": item.get("lesion_pixel_containment"),
                        "roi_area_inflation": item.get("roi_area_inflation"),
                        "mask": str(mask_path.resolve()), "mask_sha256": sha256_file(mask_path),
                        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
                        "elapsed_ms": (time.perf_counter()-started)*1000, "metrics": metrics})
    atomic_json(out / "results.json", {"schema_version": 2, "method_id": model["id"], "count": len(records),
                                        "roi_manifest_sha256": sha256_file(args.rois), "records": records})
    print(f"V2_SEGMENTER_COMPLETE={model['id']} count={len(records)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(".")); parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True); choice.add_argument("--method-index", type=int); choice.add_argument("--method-id")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-ground-truth", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__": main()
