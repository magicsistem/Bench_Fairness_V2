#!/usr/bin/env python3
"""Create the final machine-auditable V2 provenance index."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from thesis_fitzpatrick.v2 import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = ["configs/methodology_v2.json", "artifacts/development/training_manifest.json",
        "artifacts/development/validation_manifest.json", "artifacts/yolov7/roi_margin.json",
        "artifacts/yolov7/final/weights/best.pt", "artifacts/selection/top3.json", "artifacts/color/lesion_margin.json",
        "scientific_freeze.json", "artifacts/test/access_manifest.json", "results/test_original/robustness_ita.json",
        "results/test_mst/analysis.json", "artifacts/mskcc/census.json", "results/mskcc_analysis.json"]
    missing = [item for item in required if not (ROOT/item).is_file()]
    if missing: raise SystemExit("final provenance missing: " + ", ".join(missing))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    logs = sorted((ROOT/"logs").glob("slurm-*.out")); jobs = sorted({path.stem.rsplit("-", 1)[-1] for path in logs})
    atomic_json(ROOT/"artifacts/final/provenance.json", {"schema_version": 2, "status": "complete", "git_commit": commit,
        "config_sha256": sha256_file(ROOT/required[0]), "dataset_manifest_sha256": sha256_file(ROOT/required[1]),
        "jobs": jobs, "artifacts": {item: sha256_file(ROOT/item) for item in required},
        "test_open_event": json.loads((ROOT/"artifacts/test/open_event.json").read_text())})
    print("V2_FINAL_PROVENANCE_COMPLETE")


if __name__ == "__main__": main()
