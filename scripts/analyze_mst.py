#!/usr/bin/env python3
"""D47-D49 paired MST sensitivity summaries without hypothesis tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_fitzpatrick.v2 import atomic_json


def run(original_root: Path, mst_root: Path, selection: Path, output: Path, seed: int) -> None:
    top3 = json.loads(selection.read_text(encoding="utf-8"))["top3"]; generator = np.random.default_rng(seed); reports = []
    for method in top3:
        original = {r["image_id"]: r for r in json.loads((original_root/method/"results.json").read_text())["records"]}
        variants = json.loads((mst_root/method/"results.json").read_text())["records"]
        by_condition = {}
        for record in variants:
            source, condition = record["image_id"].rsplit("__", 1); base = original[source]
            by_condition.setdefault(condition, []).append({"source_image_id": source,
                **{f"delta_{metric}": record["metrics"][metric]-base["metrics"][metric]
                   for metric in ("jaccard", "threshold_jaccard", "dice", "boundary_f1")}})
        conditions = []
        for condition in sorted(by_condition):
            rows = by_condition[condition]; item = {"condition": condition, "n": len(rows), "metrics": {}}
            indices = generator.integers(0, len(rows), size=(10_000, len(rows)))
            for metric in ("jaccard", "threshold_jaccard", "dice", "boundary_f1"):
                values = np.array([r[f"delta_{metric}"] for r in rows]); means = values[indices].mean(axis=1)
                item["metrics"][f"delta_{metric}"] = {"mean": float(values.mean()), "median": float(np.median(values)),
                    "q1": float(np.percentile(values, 25)), "q3": float(np.percentile(values, 75)),
                    "ci95": [float(x) for x in np.percentile(means, (2.5, 97.5))], "worst_source_drop": float(values.min())}
            conditions.append(item)
        reports.append({"method_id": method, "conditions": conditions})
    atomic_json(output, {"schema_version": 2, "primary": "paired delta continuous Jaccard versus ORIGINAL", "top3": top3,
                         "bootstrap_repetitions": 10_000, "hypothesis_tests": False, "methods": reports})


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--original", type=Path, required=True); p.add_argument("--mst", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=20260828)
    a = p.parse_args(); run(a.original, a.mst, a.selection, a.output, a.seed)


if __name__ == "__main__": main()
