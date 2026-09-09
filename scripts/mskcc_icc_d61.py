#!/usr/bin/env python3
"""D61: three ICC models and anatomical-site agreement from existing MSKCC pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ICC_KEYS = ("icc_1_1_oneway", "icc_2_1_absolute_agreement", "icc_3_1_consistency")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _not_estimable(reason: str) -> dict:
    return {"estimate": None, "icc_status": "not_estimable", "reason": reason}


def icc_estimates(pipeline: np.ndarray, colorimeter: np.ndarray) -> dict[str, dict]:
    """Shrout-Fleiss ICC1/2/3 single-measure estimates for paired columns."""
    pipeline = np.asarray(pipeline, dtype=np.float64)
    colorimeter = np.asarray(colorimeter, dtype=np.float64)
    if pipeline.shape != colorimeter.shape or pipeline.ndim != 1 or pipeline.size < 2:
        return {key: _not_estimable("insufficient_pairs") for key in ICC_KEYS}
    values = np.column_stack((pipeline, colorimeter))
    if not np.all(np.isfinite(values)):
        return {key: _not_estimable("not_estimable") for key in ICC_KEYS}
    n, k = values.shape
    grand = values.mean()
    row_mean, col_mean = values.mean(1), values.mean(0)
    msr = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    msc = n * np.sum((col_mean - grand) ** 2) / (k - 1)
    mse = np.sum((values - row_mean[:, None] - col_mean[None, :] + grand) ** 2) / ((n - 1) * (k - 1))
    msw = np.sum((values - row_mean[:, None]) ** 2) / (n * (k - 1))
    scale = max(1.0, abs(msr), abs(msc), abs(mse), abs(msw))
    if abs(msr) <= np.finfo(np.float64).eps * scale:
        return {key: _not_estimable("zero_between_target_variance") for key in ICC_KEYS}
    denominators = {
        "icc_1_1_oneway": msr + (k - 1) * msw,
        "icc_2_1_absolute_agreement": msr + (k - 1) * mse + (k / n) * (msc - mse),
        "icc_3_1_consistency": msr + (k - 1) * mse,
    }
    numerators = {
        "icc_1_1_oneway": msr - msw,
        "icc_2_1_absolute_agreement": msr - mse,
        "icc_3_1_consistency": msr - mse,
    }
    output = {}
    for key in ICC_KEYS:
        denominator = denominators[key]
        if not np.isfinite(denominator) or abs(denominator) <= np.finfo(np.float64).eps * scale:
            output[key] = _not_estimable("anova_degenerate")
        else:
            estimate = numerators[key] / denominator
            output[key] = ({"estimate": float(estimate), "icc_status": "complete", "reason": None}
                           if np.isfinite(estimate) else _not_estimable("not_estimable"))
    return output


def _scope(records: list[dict], generator: np.random.Generator, repetitions: int) -> dict:
    usable = [r for r in records if r.get("estimate") is not None and r.get("reference") is not None]
    patients = sorted({str(r["patient_id"]) for r in usable})
    pipeline = np.asarray([r["estimate"] for r in usable], dtype=np.float64)
    colorimeter = np.asarray([r["reference"] for r in usable], dtype=np.float64)
    point = (icc_estimates(pipeline, colorimeter) if len(patients) >= 2
             else {key: _not_estimable("insufficient_patients") for key in ICC_KEYS})
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(usable):
        by_patient[str(record["patient_id"])].append(index)
    bootstrap = {key: [] for key in ICC_KEYS}
    invalid = {key: Counter() for key in ICC_KEYS}
    if len(patients) < 2:
        for key in ICC_KEYS:
            invalid[key]["insufficient_patients"] = repetitions
    else:
        blocks = [np.asarray(by_patient[patient], dtype=np.int64) for patient in patients]
        for _ in range(repetitions):
            chosen = generator.integers(0, len(blocks), size=len(blocks))
            indices = np.concatenate([blocks[index] for index in chosen])
            estimates = icc_estimates(pipeline[indices], colorimeter[indices])
            for key, result in estimates.items():
                if result["estimate"] is None:
                    invalid[key][result["reason"]] += 1
                else:
                    bootstrap[key].append(result["estimate"])
    for key in ICC_KEYS:
        values = np.asarray(bootstrap[key], dtype=np.float64)
        point[key].update({
            "ci95_low": float(np.percentile(values, 2.5)) if values.size else None,
            "ci95_high": float(np.percentile(values, 97.5)) if values.size else None,
            "bootstrap_total": repetitions,
            "bootstrap_valid": int(values.size),
            "bootstrap_invalid": repetitions - int(values.size),
            "bootstrap_invalid_reasons": dict(sorted(invalid[key].items())),
        })
    delta = pipeline - colorimeter
    sd = float(delta.std(ddof=1)) if delta.size >= 2 else None
    bias = float(delta.mean()) if delta.size else None
    estimates = [point[key]["estimate"] for key in ICC_KEYS]
    return {
        "status": "complete" if all(value is not None for value in estimates) else ("partial" if any(value is not None for value in estimates) else "not_estimable"),
        "n_images": len(usable),
        "n_patients": len(patients),
        "n_effective_pairs": len(usable),
        **point,
        "delta_icc_3_1_minus_2_1": (point["icc_3_1_consistency"]["estimate"] - point["icc_2_1_absolute_agreement"]["estimate"]
                                         if point["icc_3_1_consistency"]["estimate"] is not None and point["icc_2_1_absolute_agreement"]["estimate"] is not None else None),
        "bias": bias,
        "mae": float(np.abs(delta).mean()) if delta.size else None,
        "rmse": float(np.sqrt(np.mean(delta ** 2))) if delta.size else None,
        "sd_differences": sd,
        "bland_altman": {
            "lower_loa": bias - 1.96 * sd if bias is not None and sd is not None else None,
            "upper_loa": bias + 1.96 * sd if bias is not None and sd is not None else None,
        },
    }


def _format(value) -> str:
    return "NA" if value is None else f"{value:.10f}" if isinstance(value, float) else str(value)


def _ci(value: dict) -> str:
    return "NA" if value["ci95_low"] is None else f"[{value['ci95_low']:.10f}, {value['ci95_high']:.10f}]"


def _summary_row(method: str, site: str | None, value: dict) -> dict:
    row = {"Method": method}
    if site is not None:
        row["Anatomical site"] = site
    row.update({
        "N images": value["n_images"], "N patients": value["n_patients"], "N effective pairs": value["n_effective_pairs"],
        "ICC(1,1)": _format(value["icc_1_1_oneway"]["estimate"]), "95% CI ICC(1,1)": _ci(value["icc_1_1_oneway"]),
        "ICC(2,1) absolute agreement": _format(value["icc_2_1_absolute_agreement"]["estimate"]), "95% CI ICC(2,1)": _ci(value["icc_2_1_absolute_agreement"]),
        "ICC(3,1) consistency": _format(value["icc_3_1_consistency"]["estimate"]), "95% CI ICC(3,1)": _ci(value["icc_3_1_consistency"]),
        "Delta ICC3-ICC2": _format(value["delta_icc_3_1_minus_2_1"]), "Bias ITA": _format(value["bias"]),
        "MAE ITA": _format(value["mae"]), "RMSE ITA": _format(value["rmse"]), "SD differences": _format(value["sd_differences"]),
        "Lower LoA": _format(value["bland_altman"]["lower_loa"]), "Upper LoA": _format(value["bland_altman"]["upper_loa"]),
    })
    return row


def _diagnostics(method: str, scope: str, site: str | None, value: dict) -> list[dict]:
    labels = {"icc_1_1_oneway": "ICC(1,1)", "icc_2_1_absolute_agreement": "ICC(2,1)/ICC(A,1)",
              "icc_3_1_consistency": "ICC(3,1)/ICC(C,1)"}
    return [{"Method": method, "Scope": scope, "Anatomical site": site or "NA", "ICC type": labels[key],
             "Point estimate": _format(value[key]["estimate"]), "CI low": _format(value[key]["ci95_low"]),
             "CI high": _format(value[key]["ci95_high"]), "Bootstrap valid": value[key]["bootstrap_valid"],
             "Bootstrap invalid": value[key]["bootstrap_invalid"], "Status": value[key]["icc_status"],
             "Reason": value[key]["reason"] or "NA",
             "Bootstrap invalid reasons": json.dumps(value[key]["bootstrap_invalid_reasons"], sort_keys=True)} for key in ICC_KEYS]


def _write_csv(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _verify_invariants(manifest_path: Path) -> dict[str, str]:
    manifest = _load(manifest_path); expected = manifest["sha256"]
    for relative, digest in expected.items():
        target = ROOT / relative
        if not target.is_file() or sha256_file(target) != digest:
            raise SystemExit(f"D61 historical invariant mismatch: {relative}")
    return expected


def analyze(args: argparse.Namespace) -> None:
    outputs = [args.output, args.global_csv, args.site_csv, args.diagnostics_csv, args.manifest, args.provenance]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit("D61 refuses to overwrite existing outputs: " + ", ".join(existing))
    invariants = _verify_invariants(args.invariants)
    census_value, color_value, historical = _load(args.census), _load(args.color), _load(args.historical)
    census = {r["image_id"]: r for r in census_value["records"]}
    historical_by_method = {m["method_id"]: m for m in historical["methods"]}
    generator = np.random.default_rng(args.seed)
    methods = []
    for method in color_value["top3"]:
        extracted = {r["image_id"]: r for r in color_value["records"] if r["method_id"] == method}
        joined = []
        for identifier, item in census.items():
            metric = (extracted.get(identifier) or {}).get("color_metrics") or {}
            joined.append({**item, "estimate": metric.get("ita_degrees"), "reference": item.get("reference_ita")})
        global_result = _scope(joined, generator, args.repetitions)
        sites = sorted({str(r["anatomic_site"]) for r in joined if r.get("anatomic_site") not in (None, "")})
        by_site = {site: _scope([r for r in joined if str(r.get("anatomic_site")) == site], generator, args.repetitions) for site in sites}
        old = historical_by_method[method]["continuous"]["icc_absolute_agreement"]
        new = global_result["icc_2_1_absolute_agreement"]["estimate"]
        difference = None if new is None else float(new - old)
        matched = difference is not None and abs(difference) <= args.historical_tolerance
        methods.append({"method_id": method, "global": global_result, "by_anatomical_site": by_site,
                        "historical_icc_2_1_reproduction": {"historical": old, "recalculated": new,
                                                            "difference": difference, "tolerance": args.historical_tolerance,
                                                            "matched": matched}})
    mismatches = [m["method_id"] for m in methods if not m["historical_icc_2_1_reproduction"]["matched"]]
    if mismatches:
        raise SystemExit("D61 ICC(2,1) historical mismatch; results not written: " + ", ".join(mismatches))
    result = {
        "schema_version": 3, "decision": "D61", "status": "complete",
        "primary": "icc_2_1_absolute_agreement", "primary_alias": "icc_a_1",
        "secondary": "icc_3_1_consistency", "secondary_alias": "icc_c_1", "exploratory": "icc_1_1_oneway",
        "measurement_order": ["ita_pipeline", "ita_colorimeter"], "anatomical_site_field": "anatomic_site",
        "bootstrap_unit": "patient_id", "bootstrap_repetitions": args.repetitions,
        "bootstrap_percentiles": [2.5, 97.5], "seed": args.seed,
        "reference_validation": {
            "library": "pingouin", "version": "0.5.5", "function": "intraclass_corr",
            "mapping": {"ICC1": "icc_1_1_oneway", "ICC2": "icc_2_1_absolute_agreement", "ICC3": "icc_3_1_consistency"},
            "automated_test": "tests/test_d61_icc.py::test_pingouin_0_5_5_reference_fixture",
        },
        "source_hashes": {_relative(args.census): sha256_file(args.census), _relative(args.color): sha256_file(args.color),
                          _relative(args.historical): sha256_file(args.historical)},
        "historical_invariants": invariants, "methods": methods,
    }
    global_rows = [_summary_row(m["method_id"], None, m["global"]) for m in methods]
    site_rows = [_summary_row(m["method_id"], site, value) for m in methods for site, value in m["by_anatomical_site"].items()]
    diagnostics = [row for m in methods for row in _diagnostics(m["method_id"], "global", None, m["global"])]
    diagnostics.extend(row for m in methods for site, value in m["by_anatomical_site"].items()
                       for row in _diagnostics(m["method_id"], "by_anatomical_site", site, value))
    with tempfile.TemporaryDirectory(prefix="d61-", dir=args.output.parent) as directory:
        temporary = Path(directory)
        staged = {path: temporary / path.name for path in outputs}
        atomic_json(staged[args.output], result)
        _write_csv(staged[args.global_csv], global_rows); _write_csv(staged[args.site_csv], site_rows); _write_csv(staged[args.diagnostics_csv], diagnostics)
        result_outputs = [args.output, args.global_csv, args.site_csv, args.diagnostics_csv]
        atomic_json(staged[args.manifest], {
            "schema_version": 1, "decision": "D61", "status": "complete", "method_count": len(methods),
            "anatomical_site_count": len(site_rows), "diagnostic_row_count": len(diagnostics),
            "inputs": result["source_hashes"],
            "outputs": {_relative(path): sha256_file(staged[path]) for path in result_outputs},
        })
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        output_hashes = {_relative(path): sha256_file(staged[path]) for path in outputs[:-1]}
        atomic_json(staged[args.provenance], {
            "schema_version": 1, "decision": "D61", "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "configuration": {"path": "configs/methodology_v2.json", "sha256": sha256_file(ROOT / "configs/methodology_v2.json")},
            "inputs": result["source_hashes"], "historical_invariants": invariants, "outputs": output_hashes,
        })
        for destination in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True); os.replace(staged[destination], destination)
    print("D61_MSKCC_ICC_ANALYSIS_COMPLETE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True); parser.add_argument("--color", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True); parser.add_argument("--invariants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--site-csv", type=Path, required=True); parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260828); parser.add_argument("--historical-tolerance", type=float, default=1e-12)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
