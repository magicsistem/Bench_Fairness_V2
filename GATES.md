# Gates: metodología V2 completa

OWNS: **

Scope: implementar, ejecutar y verificar de extremo a extremo la metodología canónica D01–D60 sin abrir Test antes del scientific freeze.

- [x] G1: el repositorio V2 y CEDIA tienen una revisión trazable y reproducible
  CHECK: python scripts/verify_v2.py repository
  EXPECT: V2 repository gate passed
  EVIDENCE: commit 56c1447; verificación final CEDIA 25037 y repositorio local aprobados

- [x] G2: datos, folds, manifiestos y detector YOLOv7 cumplen D01–D25 sin leakage
  CHECK: python scripts/verify_v2.py detector
  EXPECT: V2 detector gate passed
  EVIDENCE: job 25037, `V2 detector gate passed`

- [x] G3: los segmentadores elegibles y el TOP-3 V2 cumplen D26–D32
  CHECK: python scripts/verify_v2.py selection
  EXPECT: V2 selection gate passed
  EVIDENCE: job 25037, `V2 selection gate passed`

- [x] G4: colorimetría, clean-skin y freeze cumplen D33–D42
  CHECK: python scripts/verify_v2.py freeze
  EXPECT: V2 freeze gate passed
  EVIDENCE: job 25037, `V2 freeze gate passed`

- [x] G5: Test sellado y experimento MST end-to-end de imagen completa cumplen D25 y D43–D49/D59 después del freeze, con PNG comprimido sin pérdida, YOLOv7 por condición y fallback verificable
  CHECK: python scripts/verify_v2.py mst
  EXPECT: V2 MST gate passed
  EVIDENCE: job 25037, `V2 MST gate passed`

- [x] G6: el censo y análisis MSKCC cumplen D50–D51 y la decisión de concordancia aprobada
  CHECK: python scripts/verify_v2.py mskcc
  EXPECT: V2 MSKCC gate passed
  EVIDENCE: jobs 25033–25036 y 25037, `V2 MSKCC gate passed`

- [x] G7: un único run.sh orquesta todas las etapas y todos los jobs científicos prueban compute-0-2 con compute-0-1 excluido
  CHECK: python scripts/verify_v2.py pipeline
  EXPECT: V2 pipeline gate passed
  EVIDENCE: pipeline gate aprobado; sacct confirma compute-0-2 en la cadena final

- [x] G8: bitácora, hashes, jobs, commits y artefactos permiten auditar toda la ejecución
  CHECK: python scripts/verify_v2.py provenance
  EXPECT: V2 provenance gate passed
  EVIDENCE: job 25037, `V2 provenance gate passed` y artifacts/final/provenance.json
