#!/usr/bin/env python3
"""Apply D28-D32 to complete Validation results and freeze a new V2 TOP-3."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from thesis_fitzpatrick.metrics import holm_adjust, paired_permutation_pvalue
from thesis_fitzpatrick.v2 import atomic_json, sha256_file


ORDER = (("threshold_jaccard", -1), ("jaccard", -1), ("dice", -1), ("boundary_f1", -1), ("hd95_normalized", 1))


def summarize(values: np.ndarray, samples: np.ndarray) -> dict:
    finite, sample_finite = values[np.isfinite(values)], samples[np.isfinite(samples)]
    if not len(finite) or not len(sample_finite): return {"n": 0, "mean": None, "median": None, "q1": None, "q3": None, "mean_ci95": None}
    low, high = np.percentile(sample_finite, (2.5, 97.5))
    return {"n": len(finite), "mean": float(finite.mean()), "median": float(np.median(finite)),
            "q1": float(np.percentile(finite, 25)), "q3": float(np.percentile(finite, 75)),
            "mean_ci95": [float(low), float(high)]}


def run(input_root: Path, output: Path, repetitions: int, seed: int) -> None:
    files = sorted(input_root.glob("*/results.json"))
    if len(files) != 16: raise SystemExit(f"D57 requires 16 complete methods, found {len(files)}")
    values, hashes, identifiers = {}, {}, None
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8")); records = report["records"]
        ids = [record["image_id"] for record in records]
        if len(records) != report["count"] or any(record["status"] != "complete" for record in records):
            raise SystemExit(f"incomplete method: {report['method_id']}")
        if identifiers is None: identifiers = ids
        if ids != identifiers: raise SystemExit(f"unpaired method: {report['method_id']}")
        values[report["method_id"]] = {metric: np.array([r["metrics"][metric] for r in records], float) for metric, _ in ORDER}
        hashes[report["method_id"]] = sha256_file(path)
    generator = np.random.default_rng(seed); count = len(identifiers)
    indices = generator.integers(0, count, size=(repetitions, count))
    means = {method: {metric: np.nanmean(data[metric][indices], axis=1) for metric, _ in ORDER} for method, data in values.items()}
    def key(method: str) -> tuple:
        result = []
        for metric, direction in ORDER:
            value = np.nanmean(values[method][metric])
            result.append(direction * value if np.isfinite(value) else float("inf"))
        return tuple(result)
    ranking = sorted(values, key=key)
    primary_matrix = np.stack([means[method]["threshold_jaccard"] for method in ranking])
    positions = np.empty_like(primary_matrix, dtype=int)
    for replicate in range(repetitions):
        positions[np.argsort(-primary_matrix[:, replicate], kind="stable"), replicate] = np.arange(1, len(ranking)+1)
    summaries = []
    for row, method in enumerate(ranking):
        item = {"rank": row+1, "method_id": method, "metrics": {}}
        for metric, _ in ORDER: item["metrics"][metric] = summarize(values[method][metric], means[method][metric])
        item["rank_bootstrap"] = {"median": float(np.median(positions[row])),
                                  "ci95": [float(x) for x in np.percentile(positions[row], (2.5, 97.5))],
                                  "probability_first": float(np.mean(positions[row] == 1)),
                                  "probability_top3": float(np.mean(positions[row] <= 3))}
        summaries.append(item)
    comparisons, pvalues = [], []
    for first, second in itertools.combinations(ranking, 2):
        a, b = values[first]["threshold_jaccard"], values[second]["threshold_jaccard"]
        delta = a-b; sample = delta[indices].mean(axis=1)
        p = paired_permutation_pvalue(a, b, repetitions=repetitions, seed=seed)
        comparisons.append({"first": first, "second": second, "mean_difference": float(delta.mean()),
                            "ci95": [float(x) for x in np.percentile(sample, (2.5, 97.5))], "p_raw": p})
        pvalues.append(p)
    for item, adjusted in zip(comparisons, holm_adjust(pvalues)): item["p_holm"] = adjusted
    atomic_json(output, {"schema_version": 2, "rule": "D28-D32", "repetitions": repetitions, "seed": seed,
                         "method_count": 16, "image_count": count, "top3": ranking[:3], "ranking": summaries,
                         "paired_comparisons": comparisons, "input_hashes": hashes})
    print("V2_TOP3_COMPLETE=" + ",".join(ranking[:3]))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828); args = parser.parse_args()
    run(args.input, args.output, args.repetitions, args.seed)


if __name__ == "__main__": main()
