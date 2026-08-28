#!/usr/bin/env python3
"""Build V2 development manifests, grouped folds and YOLO labels on CEDIA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from thesis_fitzpatrick.v2 import atomic_json, bbox_from_binary, sha256_file


def image_id(path: Path) -> str:
    name = path.stem
    return name.removesuffix("_segmentation")


def assert_development_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise SystemExit(f"sealed Test path rejected before freeze: {path}")


def files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    return {image_id(path): path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in suffixes}


def manifest(images_dir: Path, masks_dir: Path, output: Path, split: str) -> None:
    assert_development_path(images_dir)
    assert_development_path(masks_dir)
    images, masks = files(images_dir, {".jpg", ".jpeg", ".png"}), files(masks_dir, {".png"})
    if set(images) != set(masks):
        raise SystemExit(f"unpaired data: images_only={len(set(images)-set(masks))} masks_only={len(set(masks)-set(images))}")
    records = []
    for identifier in sorted(images):
        with Image.open(images[identifier]) as image, Image.open(masks[identifier]) as mask_image:
            width, height = image.size
            mask = np.asarray(mask_image.convert("L")) >= 128
            if mask.shape != (height, width):
                raise SystemExit(f"shape mismatch: {identifier}")
            bbox = bbox_from_binary(mask)
        records.append({
            "image_id": identifier,
            "image": str(images[identifier].resolve()),
            "mask": str(masks[identifier].resolve()),
            "width": width,
            "height": height,
            "bbox_xyxy_half_open": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
            "image_sha256": sha256_file(images[identifier]),
            "mask_sha256": sha256_file(masks[identifier]),
        })
    atomic_json(output, {"schema_version": 2, "split": split, "count": len(records), "records": records})


def metadata_groups(metadata_dir: Path | None) -> dict[str, str]:
    groups = {}
    if metadata_dir is None or not metadata_dir.is_dir():
        return groups
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        identifier = str(value.get("isic_id") or value.get("image_id") or path.stem)
        source = value.get("metadata", value)
        group = source.get("patient_id") or source.get("lesion_id") or source.get("duplicate_group_id")
        if group not in (None, ""):
            groups[identifier] = str(group)
    return groups


def folds(manifest_path: Path, output: Path, metadata_dir: Path | None, seed: int) -> None:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups_by_image = metadata_groups(metadata_dir)
    members: dict[str, list[str]] = defaultdict(list)
    missing = []
    for record in value["records"]:
        identifier = record["image_id"]
        group = groups_by_image.get(identifier)
        if group is None:
            group, missing = f"image:{identifier}", [*missing, identifier]
        members[group].append(identifier)
    ordered = sorted(members, key=lambda key: (-len(members[key]), hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()))
    fold_members = [[] for _ in range(5)]
    fold_counts = [0] * 5
    for group in ordered:
        fold = min(range(5), key=lambda index: (fold_counts[index], index))
        fold_members[fold].extend(sorted(members[group]))
        fold_counts[fold] += len(members[group])
    assignments = {identifier: fold for fold, identifiers in enumerate(fold_members) for identifier in identifiers}
    if len(assignments) != value["count"]:
        raise SystemExit("fold assignment is incomplete")
    atomic_json(output, {
        "schema_version": 2,
        "seed": seed,
        "fold_count": 5,
        "counts": fold_counts,
        "group_fields": ["patient_id", "lesion_id", "duplicate_group_id"],
        "metadata_missing_count": len(missing),
        "metadata_missing_image_ids": missing,
        "assignments": assignments,
    })


def yolo(manifest_path: Path, folds_path: Path, output: Path) -> None:
    manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold_value = json.loads(folds_path.read_text(encoding="utf-8"))
    labels = output / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in manifest_value["records"]:
        x0, y0, x1, y1 = record["bbox_xyxy_half_open"]
        width, height = record["width"], record["height"]
        line = f"0 {(x0+x1)/(2*width):.10f} {(y0+y1)/(2*height):.10f} {(x1-x0)/width:.10f} {(y1-y0)/height:.10f}\n"
        target = labels / f"{record['image_id']}.txt"
        target.write_text(line, encoding="ascii")
        rows.append({"image_id": record["image_id"], "fold": fold_value["assignments"][record["image_id"]], "label_sha256": sha256_file(target)})
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "fold", "label_sha256"])
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    make_manifest = sub.add_parser("manifest")
    make_manifest.add_argument("--images", type=Path, required=True); make_manifest.add_argument("--masks", type=Path, required=True)
    make_manifest.add_argument("--output", type=Path, required=True); make_manifest.add_argument("--split", choices=["training", "validation"], required=True)
    make_folds = sub.add_parser("folds")
    make_folds.add_argument("--manifest", type=Path, required=True); make_folds.add_argument("--output", type=Path, required=True)
    make_folds.add_argument("--metadata", type=Path); make_folds.add_argument("--seed", type=int, default=20260828)
    make_yolo = sub.add_parser("yolo")
    make_yolo.add_argument("--manifest", type=Path, required=True); make_yolo.add_argument("--folds", type=Path, required=True); make_yolo.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest": manifest(args.images, args.masks, args.output, args.split)
    elif args.command == "folds": folds(args.manifest, args.output, args.metadata, args.seed)
    else: yolo(args.manifest, args.folds, args.output)


if __name__ == "__main__":
    main()

