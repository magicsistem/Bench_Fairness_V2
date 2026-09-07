#!/usr/bin/env python3
from pathlib import Path


report = Path("docs/METODOLOGIA_Y_RESULTADOS_V2.md")
text = report.read_text(encoding="utf-8")
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
    "D01–D60",
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
print("V2 methodology report gate passed")
