#!/usr/bin/env python3
"""Fail-closed acceptance oracle for D61 MSKCC ICC extension."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT.parent / "Guia" / "10_METHODOLOGY_V2_FROM_ZERO.md"
ICC_KEYS = ("icc_1_1_oneway", "icc_2_1_absolute_agreement", "icc_3_1_consistency")
METHODS = {"avit", "delightsam-dermoscopy", "vmunet-isic17"}
DISPLAY_NAMES = {"avit": "AViT", "delightsam-dermoscopy": "DeLightSAM-Dermoscopy", "vmunet-isic17": "VM-UNet ISIC17"}
SITES = {"abdomen", "dorsal forearm", "head/neck", "lateral torso", "lower back", "lower leg",
         "palms/soles", "upper arm", "upper back", "upper chest", "upper leg", "ventral forearm"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load(relative: str) -> dict:
    path = ROOT / relative
    require(path.is_file(), f"missing D61 artifact: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def csv_rows(relative: str) -> list[dict]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def display_number(value: str) -> str:
    return "NA" if value in {"", "NA"} else f"{float(value):.6f}"


def display_ci(value: str) -> str:
    if value in {"", "NA"}:
        return "NA"
    low, high = value.strip("[]").split(", ")
    return f"[{float(low):.6f}, {float(high):.6f}]"


def verify_scope(value: dict, label: str) -> None:
    require(value["n_images"] == value["n_effective_pairs"] and value["n_patients"] >= 0, f"{label}: invalid sample counts")
    for field in ("bias", "mae", "rmse", "sd_differences", "bland_altman"):
        require(field in value, f"{label}: missing {field}")
    require({"lower_loa", "upper_loa"} <= value["bland_altman"].keys(), f"{label}: missing LoA")
    for key in ICC_KEYS:
        icc = value[key]
        require(icc["bootstrap_total"] == 10_000, f"{label}/{key}: wrong bootstrap total")
        require(icc["bootstrap_valid"] + icc["bootstrap_invalid"] == 10_000, f"{label}/{key}: bootstrap accounting")
        require(isinstance(icc["bootstrap_invalid_reasons"], dict), f"{label}/{key}: missing invalid reasons")
        if icc["estimate"] is None:
            require(icc["icc_status"] == "not_estimable" and icc["reason"] in {
                "insufficient_patients", "insufficient_pairs", "zero_between_target_variance", "anova_degenerate", "not_estimable"
            }, f"{label}/{key}: NA without status/reason")
        else:
            require(icc["icc_status"] == "complete" and icc["reason"] is None, f"{label}/{key}: estimable status mismatch")
            require(icc["ci95_low"] is not None and icc["ci95_high"] is not None, f"{label}/{key}: missing CI")


def main() -> None:
    config = load("configs/methodology_v2.json")["mskcc_icc_d61"]
    require(config["primary"] == "icc_2_1_absolute_agreement" and config["secondary"] == "icc_3_1_consistency"
            and config["exploratory"] == "icc_1_1_oneway", "D61 roles are ambiguous")
    require(config["bootstrap_unit"] == "patient_id" and config["bootstrap_repetitions"] == 10_000,
            "D61 bootstrap contract mismatch")
    result = load("results/mskcc_icc_d61.json")
    require(result["decision"] == "D61" and result["status"] == "complete", "D61 result incomplete")
    require(result["primary"] == config["primary"] and result["secondary"] == config["secondary"]
            and result["exploratory"] == config["exploratory"], "D61 output roles mismatch")
    require(result["bootstrap_unit"] == "patient_id" and result["anatomical_site_field"] == "anatomic_site",
            "D61 cluster/site mismatch")
    require(result["reference_validation"]["library"] == "pingouin"
            and result["reference_validation"]["mapping"] == {
                "ICC1": "icc_1_1_oneway", "ICC2": "icc_2_1_absolute_agreement", "ICC3": "icc_3_1_consistency"
            }, "D61 recognized-library validation missing")
    methods = {method["method_id"]: method for method in result["methods"]}
    require(set(methods) == METHODS, "D61 TOP-3 mismatch")
    for method, value in methods.items():
        verify_scope(value["global"], f"{method}/global")
        require(set(value["by_anatomical_site"]) == SITES, f"{method}: incomplete official anatomical sites")
        for site, scope in value["by_anatomical_site"].items():
            verify_scope(scope, f"{method}/{site}")
        reproduction = value["historical_icc_2_1_reproduction"]
        require(reproduction["matched"] and abs(reproduction["difference"]) <= reproduction["tolerance"],
                f"{method}: historical ICC(2,1) not reproduced")
    global_rows = csv_rows("results/mskcc_icc_d61_global.csv")
    site_rows = csv_rows("results/mskcc_icc_d61_by_anatomical_site.csv")
    diagnostics = csv_rows("results/mskcc_icc_d61_diagnostics.csv")
    require(len(global_rows) == 3 and len(site_rows) == 36 and len(diagnostics) == 117, "D61 table cardinality mismatch")
    require({"ICC(1,1)", "ICC(2,1) absolute agreement", "ICC(3,1) consistency", "SD differences"} <= global_rows[0].keys(),
            "D61 global table columns missing")
    require({"Anatomical site", "Lower LoA", "Upper LoA"} <= site_rows[0].keys(), "D61 site table columns missing")
    require({"ICC type", "Bootstrap valid", "Bootstrap invalid", "Status", "Reason"} <= diagnostics[0].keys(),
            "D61 diagnostics columns missing")
    manifest = load("artifacts/mskcc/icc_d61_manifest.json")
    require(manifest["decision"] == "D61" and manifest["status"] == "complete" and manifest["method_count"] == 3
            and manifest["anatomical_site_count"] == 36 and manifest["diagnostic_row_count"] == 117,
            "D61 result manifest mismatch")
    for relative, expected in manifest["outputs"].items():
        require(digest(ROOT / relative) == expected, f"D61 manifest hash mismatch: {relative}")
    invariants = load("configs/d61_historical_invariants.json")["sha256"]
    require(result["historical_invariants"] == invariants, "D61 invariant manifest mismatch")
    for relative, expected in invariants.items():
        path = ROOT / relative
        if path.is_file():
            require(digest(path) == expected, f"historical artifact changed: {relative}")
    provenance = load("artifacts/final/provenance_d61.json")
    require(provenance["decision"] == "D61" and provenance["status"] == "complete", "D61 provenance incomplete")
    require(str(provenance["slurm_job_id"] or "").isdigit() and provenance["slurm_node"] == "compute-0-2",
            "D61 provenance lacks valid Slurm compute-0-2 evidence")
    require(provenance["historical_invariants"] == invariants, "D61 provenance invariants mismatch")
    for relative, expected in provenance["outputs"].items():
        require(digest(ROOT / relative) == expected, f"D61 output hash mismatch: {relative}")
    guide = GUIDE.read_text(encoding="utf-8")
    report = (ROOT / "docs/METODOLOGIA_Y_RESULTADOS_V2.md").read_text(encoding="utf-8")
    for token in ("| D61 |", "ICC(1,1)", "ICC(2,1)/ICC(A,1)", "ICC(3,1)/ICC(C,1)",
                  "10.1037/0033-2909.86.2.420", "10.1037/1082-989X.1.1.30"):
        require(token in guide, f"guide missing {token}")
        require(token in report, f"report missing {token}")
    for token in ("Concordancia MSKCC por sitio anatómico", "Acuerdo absoluto frente a consistencia", "D01–D61",
                  "0.3356501648804588", "0.44896757002331095", "0.5144750357866749"):
        require(token in report, f"report missing {token}")
    for row in global_rows:
        rendered = (f"| {DISPLAY_NAMES[row['Method']]} | {row['N images']} | {row['N patients']} | "
                    f"{display_number(row['ICC(1,1)'])} {display_ci(row['95% CI ICC(1,1)'])} | "
                    f"{display_number(row['ICC(2,1) absolute agreement'])} {display_ci(row['95% CI ICC(2,1)'])} | "
                    f"{display_number(row['ICC(3,1) consistency'])} {display_ci(row['95% CI ICC(3,1)'])} | "
                    f"{display_number(row['Delta ICC3-ICC2'])} | {display_number(row['Bias ITA'])} | "
                    f"{display_number(row['MAE ITA'])} | {display_number(row['RMSE ITA'])} | "
                    f"{display_number(row['SD differences'])} | "
                    f"[{display_number(row['Lower LoA'])}, {display_number(row['Upper LoA'])}] |")
        require(rendered in report, f"report/global mismatch: {row['Method']}")
    for row in site_rows:
        rendered = (f"| {DISPLAY_NAMES[row['Method']]} | {row['Anatomical site']} | {row['N images']} | {row['N patients']} | "
                    f"{display_number(row['ICC(1,1)'])}" + ("" if row['ICC(1,1)'] == "NA" else f" {display_ci(row['95% CI ICC(1,1)'])}") + " | "
                    f"{display_number(row['ICC(2,1) absolute agreement'])}" + ("" if row['ICC(2,1) absolute agreement'] == "NA" else f" {display_ci(row['95% CI ICC(2,1)'])}") + " | "
                    f"{display_number(row['ICC(3,1) consistency'])}" + ("" if row['ICC(3,1) consistency'] == "NA" else f" {display_ci(row['95% CI ICC(3,1)'])}") + " | "
                    f"{display_number(row['Delta ICC3-ICC2'])} | {display_number(row['Bias ITA'])} | "
                    f"{display_number(row['MAE ITA'])} | {display_number(row['RMSE ITA'])} | "
                    f"{display_number(row['Lower LoA'])} | {display_number(row['Upper LoA'])} |")
        require(rendered in report, f"report/site mismatch: {row['Method']}/{row['Anatomical site']}")
    for row in diagnostics:
        scope = "global" if row["Scope"] == "global" else "sitio"
        rendered = (f"| {DISPLAY_NAMES[row['Method']]} | {scope} | {row['Anatomical site']} | {row['ICC type']} | "
                    f"{display_number(row['Point estimate'])} | {display_number(row['CI low'])} | "
                    f"{display_number(row['CI high'])} | {row['Bootstrap valid']} | {row['Bootstrap invalid']} | "
                    f"{row['Status']} | {row['Reason']} |")
        require(rendered in report, f"report/diagnostic mismatch: {row['Method']}/{scope}/{row['Anatomical site']}/{row['ICC type']}")
    environment = dict(os.environ); environment["PYTHONPATH"] = str(ROOT / "src")
    check = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_d61_icc.py", "-v"],
                           cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    require(check.returncode == 0 and "Ran 7 tests" in check.stdout and "OK" in check.stdout, "D61 statistical tests failed")
    print("D61_MSKCC_ICC_EXTENSION_COMPLETE")


if __name__ == "__main__":
    main()
