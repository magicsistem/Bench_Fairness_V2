#!/usr/bin/env python3
"""Materialize and verify the D55 scientific-freeze contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from thesis_fitzpatrick.v2 import atomic_json, sha256_file


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=True).stdout.strip()


def create(root: Path, output: Path) -> None:
    root = root.resolve()
    if git(root, "status", "--porcelain", "--untracked-files=no"): raise SystemExit("freeze rejects dirty tracked worktree")
    commit = git(root, "rev-parse", "HEAD")
    if git(root, "rev-parse", "origin/main") != commit: raise SystemExit("freeze commit is not published at origin/main")
    test_path = root/"data/raw/isic2018_task1/ISIC2018_Task1-2_Test_Input"
    if test_path.exists(): raise SystemExit("freeze rejects previously opened Test")
    required = [root/"artifacts/selection/top3.json", root/"artifacts/yolov7/roi_margin.json",
                root/"artifacts/yolov7/final/weights/best.pt", root/"artifacts/color/lesion_margin.json",
                root/"artifacts/mst/palette.png", root/"artifacts/environment/pip-freeze.txt",
                root/"artifacts/environment/container.sha256"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise SystemExit("freeze missing artifacts: " + ", ".join(missing))
    selection = json.loads(required[0].read_text(encoding="utf-8"))
    if len(selection.get("top3", [])) != 3 or selection.get("method_count") != 16: raise SystemExit("invalid TOP-3 selection")
    tracked = [root/path for path in git(root, "ls-files", "configs", "scripts", "src", "tests").splitlines()]
    atomic_json(output, {"schema_version": 2, "status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
                         "scientific_commit": commit, "top3": selection["top3"], "seed": 20260828,
                         "test_absent_at_freeze": True, "code_and_config_hashes": {str(p.relative_to(root)): sha256_file(p) for p in tracked},
                         "artifact_hashes": {str(p.relative_to(root)): sha256_file(p) for p in required},
                         "sealed_protocol": "D55-D56", "post_freeze_changes_allowed": "identical-hash technical replay only"})
    print(f"V2_SCIENTIFIC_FREEZE_CREATED={output}")


def verify(root: Path, freeze: Path) -> None:
    value = json.loads(freeze.read_text(encoding="utf-8"))
    if value.get("status") != "frozen" or len(value.get("top3", [])) != 3: raise SystemExit("invalid freeze")
    for relative, expected in {**value["code_and_config_hashes"], **value["artifact_hashes"]}.items():
        path = root/relative
        if not path.is_file() or sha256_file(path) != expected: raise SystemExit(f"freeze hash mismatch: {relative}")
    print("V2_SCIENTIFIC_FREEZE_VERIFIED")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        item = sub.add_parser(name); item.add_argument("--repo", type=Path, default=Path(".")); item.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args(); create(args.repo, args.freeze) if args.command == "create" else verify(args.repo, args.freeze)


if __name__ == "__main__": main()
