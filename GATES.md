# Gates: metodología V2 completa

OWNS: **

Scope: implementar, ejecutar y verificar de extremo a extremo la metodología canónica D01–D60 sin abrir Test antes del scientific freeze.

- [x] G1: el repositorio V2 y CEDIA tienen una revisión trazable y reproducible
  CHECK: python scripts/verify_v2.py repository
  EXPECT: V2 repository gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=84a4f0238223a0a89e083b6c6875423b932003655048c371105c45b16a868cc4; output-bytes=26

- [x] G2: datos, folds, manifiestos y detector YOLOv7 cumplen D01–D25 sin leakage
  CHECK: python scripts/verify_v2.py detector
  EXPECT: V2 detector gate passed
  EVIDENCE: job 25037 en compute-0-2; log final reconfirmado por SSH el 2026-09-07: `V2 detector gate passed`

- [x] G3: los segmentadores elegibles y el TOP-3 V2 cumplen D26–D32
  CHECK: python scripts/verify_v2.py selection
  EXPECT: V2 selection gate passed
  EVIDENCE: job 25037 en compute-0-2; log final reconfirmado por SSH el 2026-09-07: `V2 selection gate passed`

- [x] G4: colorimetría, clean-skin y freeze cumplen D33–D42
  CHECK: python scripts/verify_v2.py freeze
  EXPECT: V2 freeze gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=77e14af322e6846be197e626fd5f42e9767e2ff4d31c8afc149121eb3ffaac26; output-bytes=22

- [x] G5: Test sellado y experimento MST end-to-end de imagen completa cumplen D25 y D43–D49/D59 después del freeze, con PNG comprimido sin pérdida, YOLOv7 por condición y fallback verificable
  CHECK: python scripts/verify_v2.py mst
  EXPECT: V2 MST gate passed
  EVIDENCE: job 25037 en compute-0-2; log final reconfirmado por SSH el 2026-09-07: `V2 MST gate passed`

- [x] G6: el censo y análisis MSKCC cumplen D50–D51 y la decisión de concordancia aprobada
  CHECK: python scripts/verify_v2.py mskcc
  EXPECT: V2 MSKCC gate passed
  EVIDENCE: jobs 25033–25037 en compute-0-2; log final reconfirmado por SSH el 2026-09-07: `V2 MSKCC gate passed`

- [x] G7: un único run.sh orquesta todas las etapas y todos los jobs científicos prueban compute-0-2 con compute-0-1 excluido
  CHECK: python scripts/verify_v2.py pipeline
  EXPECT: V2 pipeline gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=b00ef6dd86bf12ae6843629731cde8bbc9fdd385322d9f489340821c877dbd8d; output-bytes=24

- [x] G8: bitácora, hashes, jobs, commits y artefactos permiten auditar toda la ejecución
  CHECK: python scripts/verify_v2.py provenance
  EXPECT: V2 provenance gate passed
  EVIDENCE: job 25037 en compute-0-2; log final reconfirmado por SSH el 2026-09-07: `V2 provenance gate passed` y `V2_COMPLETE`

- [x] G9: un informe Markdown autocontenido explica la metodología, fórmulas, referencias y resultados reales por etapa
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G10: el informe identifica los valores retenidos en cada etapa, muestra tablas completas por modelo y condición MST, y usa fórmulas renderizables con alternativa textual portable
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G11: existe un inventario trazable de todas las decisiones D01–D60, etapas, subpasos, ramas, ablaciones y variaciones canónicas y ejecutadas
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 exhaustive methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G12: las 37 referencias canónicas fueron releídas y el informe contiene una matriz que delimita su uso científico
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 exhaustive methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G13: las cifras del informe se reconcilian automáticamente con los artefactos finales extraídos de CEDIA
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 exhaustive methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G14: cada fórmula, conversión y algoritmo metodológico define símbolos, operaciones, unidades, dominio, ramas y razón de uso
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 exhaustive methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G15: cada etapa científica presenta resultados completos y una tabla resumen con interpretación sustentada
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 exhaustive methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79

- [x] G16: el informe final supera controles de cobertura, cifras, citas IEEE, Markdown matemático y regresión del repositorio
  CHECK: python scripts/verify_methodology_report.py
  EXPECT: V2 exhaustive methodology report gate passed
  EVIDENCE: exit=0; shell=/bin/sh; cwd=/run/media/miguel/Data/12. DECIMO PRIMER SEMESTRE/Tesis/Bench_Fairness_V2; path=5e95aee1592f/15 entries; EXPECT=matched; output-sha256=1e2cc856ef461076fb4dd9826a8102610d638ff8bec42f8f2089aaced06c75c5; output-bytes=79
