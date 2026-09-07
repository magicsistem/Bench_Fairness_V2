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
    "D01–D60",
    "0e61764577c2b2793e169fe8b64b8b530b362404",
    "V2_COMPLETE",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"Missing report sections/evidence: {missing}")
if text.count("|---") < 8 or text.count("\\[") < 20:
    raise SystemExit("Report lacks the expected result tables or mathematical notation")
print("V2 methodology report gate passed")
