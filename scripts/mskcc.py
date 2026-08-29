#!/usr/bin/env python3
"""Prepare the complete MSKCC census and analyze D53 agreement by patient block."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import kendalltau, spearmanr

from thesis_fitzpatrick.v2 import atomic_json, sha256_file


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream: return list(csv.DictReader(stream))


def number(value: str | None) -> float | None:
    try: return float(value) if value not in (None, "") else None
    except ValueError: return None


def prepare(root: Path, output: Path) -> None:
    metadata = rows(root/"metadata.csv"); sites = {r["lesion_id"]: r for r in rows(root/"supplements/s2.csv") if r.get("lesion_id")}
    tones = {r["tag_id"]: r for r in rows(root/"supplements/s3.csv")}; colors = {r["tag_id"]: r for r in rows(root/"supplements/s4.csv")}
    direct = {r["isic_id"]: r for r in rows(root/"supplements/s7.csv")}
    images = {path.stem: path for path in root.rglob("*.jpg")}
    if len(metadata) != 4879 or len(images) != 4879: raise SystemExit(f"D51 expected 4879 rows/images, got {len(metadata)}/{len(images)}")
    records = []
    for item in metadata:
        identifier = item["isic_id"]; path = images.get(identifier)
        if path is None: raise SystemExit(f"missing image: {identifier}")
        site = sites.get(item["lesion_id"], {}); tag = site.get("tag_id"); tone, color, observed = tones.get(tag, {}), colors.get(tag, {}), direct.get(identifier, {})
        with Image.open(path) as image: width, height = image.size
        records.append({"image_id": identifier, "image": str(path.resolve()), "image_sha256": sha256_file(path),
            "width": width, "height": height, "patient_id": item["patient_id"], "lesion_id": item["lesion_id"], "tag_id": tag,
            "anatom_site_general": item["anatom_site_general"], "anatomic_site": site.get("anatomic_site"),
            "dermoscopic_type": item["dermoscopic_type"], "image_type": item["image_type"],
            "mst": number(tone.get("mst_r1")), "reference_l": number(observed.get("average_l") or color.get("average_l")),
            "reference_a": number(observed.get("average_a") or color.get("average_a")),
            "reference_b": number(observed.get("average_b") or color.get("average_b")),
            "reference_ita": number(observed.get("average_ita") or color.get("average_ita"))})
    atomic_json(output, {"schema_version": 2, "split": "mskcc_census", "count": len(records),
                         "patient_count": len({r['patient_id'] for r in records}), "records": records})


def icc_absolute(reference: np.ndarray, estimate: np.ndarray) -> float:
    values = np.column_stack((reference, estimate)); n, k = values.shape
    if n < 2: return float("nan")
    grand = values.mean(); row_mean, col_mean = values.mean(1), values.mean(0)
    msr = k*np.sum((row_mean-grand)**2)/(n-1); msc = n*np.sum((col_mean-grand)**2)/(k-1)
    mse = np.sum((values-row_mean[:, None]-col_mean[None, :]+grand)**2)/((n-1)*(k-1))
    denominator = msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/denominator) if denominator else float("nan")


def stats(records: list[dict], generator: np.random.Generator, repetitions: int) -> dict:
    usable = [r for r in records if r["estimate"] is not None and r["reference"] is not None]
    if len(usable) < 2: return {"n": len(usable), "status": "insufficient"}
    by_patient = defaultdict(list)
    for record in usable: by_patient[record["patient_id"]].append(record)
    patients = sorted(by_patient); reference = np.array([r["reference"] for r in usable]); estimate = np.array([r["estimate"] for r in usable]); delta = estimate-reference
    boot = []
    for _ in range(repetitions):
        selected = generator.choice(patients, len(patients), replace=True); sampled = [r for patient in selected for r in by_patient[patient]]
        boot.append(icc_absolute(np.array([r["reference"] for r in sampled]), np.array([r["estimate"] for r in sampled])))
    finite = np.array(boot)[np.isfinite(boot)]
    return {"status": "complete", "n": len(usable), "patients": len(patients), "icc_absolute_agreement": icc_absolute(reference, estimate),
            "icc_ci95_patient_bootstrap": [float(x) for x in np.percentile(finite, (2.5, 97.5))],
            "bias": float(delta.mean()), "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta**2))),
            "bland_altman_limits": [float(delta.mean()-1.96*delta.std(ddof=1)), float(delta.mean()+1.96*delta.std(ddof=1))]}


def analyze(census_path: Path, color_path: Path, output: Path, repetitions: int, seed: int) -> None:
    census = {r["image_id"]: r for r in json.loads(census_path.read_text())["records"]}; colors = json.loads(color_path.read_text())
    generator = np.random.default_rng(seed); methods = []
    for method in colors["top3"]:
        extracted = {r["image_id"]: r for r in colors["records"] if r["method_id"] == method}
        joined = []
        for identifier, item in census.items():
            metric = (extracted.get(identifier) or {}).get("color_metrics") or {}
            joined.append({**item, "estimate": metric.get("ita_degrees"), "reference": item["reference_ita"]})
        groups = {}
        for field in ("anatom_site_general", "anatomic_site", "dermoscopic_type", "image_type"):
            groups[field] = {str(level): stats([r for r in joined if r.get(field) == level], generator, repetitions)
                             for level in sorted({r.get(field) for r in joined if r.get(field) not in (None, "")})}
        ordinal = [r for r in joined if r["estimate"] is not None and r["mst"] is not None]
        methods.append({"method_id": method, "continuous": stats(joined, generator, repetitions), "groups": groups,
            "mst_association": {"n": len(ordinal), "kendall_tau_b": float(kendalltau([r['mst'] for r in ordinal], [r['estimate'] for r in ordinal]).statistic) if ordinal else None,
                                "spearman_rho": float(spearmanr([r['mst'] for r in ordinal], [r['estimate'] for r in ordinal]).statistic) if ordinal else None}})
    atomic_json(output, {"schema_version": 2, "primary": "absolute-agreement ICC for ITA", "bootstrap_unit": "patient",
                         "repetitions": repetitions, "seed": seed, "methods": methods})


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--root", type=Path, required=True); a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("analyze"); b.add_argument("--census", type=Path, required=True); b.add_argument("--color", type=Path, required=True); b.add_argument("--output", type=Path, required=True); b.add_argument("--repetitions", type=int, default=10_000); b.add_argument("--seed", type=int, default=20260828)
    x = p.parse_args(); prepare(x.root, x.output) if x.command == "prepare" else analyze(x.census, x.color, x.output, x.repetitions, x.seed)


if __name__ == "__main__": main()
