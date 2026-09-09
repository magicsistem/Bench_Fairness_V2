#!/usr/bin/env python3
import json
import re
from pathlib import Path


report = Path("docs/METODOLOGIA_Y_RESULTADOS_V2.md")
text = report.read_text(encoding="utf-8")
guide = Path("../Guia/10_METHODOLOGY_V2_FROM_ZERO.md").read_text(encoding="utf-8")
config = json.loads(Path("configs/methodology_v2.json").read_text(encoding="utf-8"))
evidence = Path("docs/REPORT_EVIDENCE_V2.tsv").read_text(encoding="utf-8")
required = [
    "# Metodología y resultados de la tesis — V2",
    "## Metodología paso a paso",
    "## Resultados reales",
    "## Interpretación de los resultados",
    "## Referencias",
    "### Valores finales retenidos en cada etapa",
    "### Calibración OOF del margen de ROI",
    "### Colorimetría en Validation y margen D35",
    "#### Detector YOLOv7 por condición MST",
    "#### Detalle MST — AViT",
    "#### Detalle MST — DeLightSAM-Dermoscopy",
    "#### Detalle MST — VM-UNet ISIC17",
    "D01–D61",
    "#### Concordancia MSKCC por sitio anatómico",
    "#### Acuerdo absoluto frente a consistencia",
    "#### Tabla C — Diagnóstico ICC y bootstrap",
    "25663",
    "0e61764577c2b2793e169fe8b64b8b530b362404",
    "V2_COMPLETE",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"Missing report sections/evidence: {missing}")
if text.count("|---") < 15 or text.count("$$") < 40:
    raise SystemExit("Report lacks the expected result tables or mathematical notation")
if any(delimiter in text for delimiter in ("\\[", "\\]", "\\(", "\\)")):
    raise SystemExit("Report contains Markdown math delimiters unsupported by GitHub")

# The report must cover every canonical decision and every canonical reference.
for decision in (f"D{i:02d}" for i in range(1, 62)):
    if not re.search(rf"^\| {decision} \|", text, re.MULTILINE):
        raise SystemExit(f"Missing canonical decision row: {decision}")
for reference in range(1, 40):
    if not re.search(rf"^\[{reference}\] ", text, re.MULTILINE):
        raise SystemExit(f"Missing canonical reference: [{reference}]")

# Executable values must remain synchronized with the canonical JSON.
canonical_values = [
    config["seed"], config["bootstrap_repetitions"], config["permutation_repetitions"],
    config["yolov7"]["image_size"], config["yolov7"]["physical_batch"],
    config["yolov7"]["epochs"], config["yolov7"]["confidence_threshold"],
    config["yolov7"]["nms_iou_threshold"], config["minimum_clean_skin"]["pixels"],
]
for value in canonical_values:
    if str(value).lower() not in text.lower():
        raise SystemExit(f"Missing executable canonical value: {value}")
if guide.count("| FIJADA |") < 61:
    raise SystemExit("Canonical guide no longer exposes all fixed decisions")

# Reconcile the report with the immutable CEDIA evidence manifest.
for line in evidence.splitlines()[1:]:
    artifact, digest, counts = line.split("\t")
    if artifact not in text or digest not in text:
        raise SystemExit(f"Missing CEDIA provenance for {artifact}")
    for item in counts.split(";"):
        if item.split("=", 1)[1] not in text:
            raise SystemExit(f"Missing evidence count {item} for {artifact}")

checks = {
    "validation model rows": len(re.findall(r"^\| (?:[1-9]|1[0-6]) \| [a-z]", text, re.MULTILINE)),
    "paired comparisons": len(re.findall(r"^\| [a-z][^|]+ \| [a-z][^|]+ \| -?\d+\.\d{6} \| \[", text, re.MULTILINE)),
    "MST model-condition rows": len(re.findall(r"^\| (?:avit|delightsam-dermoscopy|vmunet-isic17) \| MST_\d{2} \|", text, re.MULTILINE)),
}
expected = {"validation model rows": 16, "paired comparisons": 120, "MST model-condition rows": 30}
if checks != expected:
    raise SystemExit(f"Result-table cardinality mismatch: {checks} != {expected}")
for token in ("Matriz completa D01–D61", "Matriz de ramas operativas", "Matriz de lectura y uso de las 39 referencias", "Resumen destacado de cada etapa y rama"):
    if token not in text:
        raise SystemExit(f"Missing exhaustive-report section: {token}")
print("V2 methodology report gate passed")
print("V2 exhaustive methodology report gate passed")
