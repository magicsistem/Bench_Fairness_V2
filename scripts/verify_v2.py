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


def artifact(stage: str, path: str, marker: str) -> None:
    value = load(path)
    require(value.get("schema_version") == 2, f"{stage}: wrong schema")
    require(value.get("status") == "complete", f"{stage}: not complete")
    print(marker)


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
    elif stage == "detector": artifact(stage, "artifacts/detector/manifest.json", "V2 detector gate passed")
    elif stage == "selection": artifact(stage, "artifacts/selection/manifest.json", "V2 selection gate passed")
    elif stage == "freeze": artifact(stage, "artifacts/freeze/scientific_freeze.json", "V2 freeze gate passed")
    elif stage == "mst": artifact(stage, "artifacts/mst/manifest.json", "V2 MST gate passed")
    elif stage == "mskcc": artifact(stage, "artifacts/mskcc/manifest.json", "V2 MSKCC gate passed")
    elif stage == "pipeline": pipeline()
    else: provenance()


if __name__ == "__main__":
    main()

