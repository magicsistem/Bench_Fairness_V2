#!/usr/bin/env python3
"""Fail-closed artifact gates for the V2 pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    target = ROOT / path
    if not target.is_file():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(target.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def repository() -> None:
    require((ROOT / "GATES.md").is_file(), "missing GATES.md")
    require((ROOT / "configs/methodology_v2.json").is_file(), "missing methodology config")
    require(len(load("configs/segmentation_models.json")["models"]) == 15, "roster must contain 15 checkpoints")
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip()
    require(remote == "https://github.com/magicsistem/Bench_Fairness_V2.git", "unexpected origin")
    print("V2 repository gate passed")


def detector() -> None:
    margin, rois = load("artifacts/yolov7/roi_margin.json"), load("artifacts/yolov7/validation_rois.json")
    require(margin.get("rule") == "D09_Q95_valid_OOF" and margin.get("valid_detection_count", 0) > 0, "invalid D09 calibration")
    require(rois.get("count") == 100 and (ROOT/"artifacts/yolov7/final/weights/best.pt").is_file(), "incomplete final detector")
    print("V2 detector gate passed")


def selection() -> None:
    value = load("artifacts/selection/top3.json")
    require(value.get("method_count") == 16 and len(value.get("top3", [])) == 3, "invalid V2 TOP-3")
    require(len(list((ROOT/"results/validation").glob("*/results.json"))) == 16, "missing 16-method validation results")
    print("V2 selection gate passed")


def freeze() -> None:
    value = load("scientific_freeze.json")
    require(value.get("status") == "frozen" and value.get("test_absent_at_freeze") is True, "invalid freeze")
    print("V2 freeze gate passed")


def mst() -> None:
    value, rois, analysis = load("artifacts/test/mst/manifest.json"), load("artifacts/test/mst_rois.json"), load("results/test_mst/analysis.json")
    require(value.get("source_count") == 1000 and value.get("expected_count") == 10000, "invalid MST census")
    require(value.get("synthesis_domain") == "full_image" and value.get("png_compress_level") == 6, "invalid MST image contract")
    require(value.get("count", 0) + value.get("unavailable_count", 0) == 10000, "incomplete MST availability census")
    require(value.get("unavailable_count", 0) == len(value.get("unavailable_records", [])), "invalid MST unavailable records")
    require(rois.get("count") == value.get("count") and all(r.get("detector_status") in ("detected", "valid_no_detection") for r in rois.get("records", [])), "invalid per-condition YOLO census")
    require(all("pixel_sha256" in r and r.get("shape") == [r.get("height"), r.get("width"), 3] for r in value.get("records", [])), "invalid lossless MST records")
    require(analysis.get("hypothesis_tests") is False and len(analysis.get("methods", [])) == 3, "invalid MST analysis")
    require(all("detector" in c for m in analysis.get("methods", []) for c in m.get("conditions", [])), "missing MST detector analysis")
    print("V2 MST gate passed")


def mskcc() -> None:
    census, analysis = load("artifacts/mskcc/census.json"), load("results/mskcc_analysis.json")
    require(census.get("count") == 4879 and census.get("patient_count") == 64, "invalid MSKCC census")
    require(analysis.get("primary") == "absolute-agreement ICC for ITA" and len(analysis.get("methods", [])) == 3, "invalid MSKCC analysis")
    print("V2 MSKCC gate passed")


def pipeline() -> None:
    run = (ROOT / "run.sh").read_text(encoding="utf-8")
    require("set -Eeuo pipefail" in run, "run.sh is not fail-fast")
    jobs = list((ROOT / "scripts/stages").glob("*.slurm"))
    require(jobs, "no Slurm stages")
    for job in jobs:
        text = job.read_text(encoding="utf-8")
        require("#SBATCH --nodelist=compute-0-2" in text, f"{job}: node not pinned")
        require("#SBATCH --exclude=compute-0-1" in text, f"{job}: bad node not excluded")
    print("V2 pipeline gate passed")


def provenance() -> None:
    manifest = load("artifacts/final/provenance.json")
    required = {"git_commit", "config_sha256", "dataset_manifest_sha256", "jobs", "artifacts"}
    require(required <= manifest.keys(), "provenance fields missing")
    print("V2 provenance gate passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["repository", "detector", "selection", "freeze", "mst", "mskcc", "pipeline", "provenance"])
    stage = parser.parse_args().stage
    if stage == "repository": repository()
    elif stage == "detector": detector()
    elif stage == "selection": selection()
    elif stage == "freeze": freeze()
    elif stage == "mst": mst()
    elif stage == "mskcc": mskcc()
    elif stage == "pipeline": pipeline()
    else: provenance()


if __name__ == "__main__":
    main()
