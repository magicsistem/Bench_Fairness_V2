# Gates: metodología V2 completa

OWNS: **

Scope: implementar, ejecutar y verificar de extremo a extremo la metodología canónica D01–D52 sin abrir Test antes del scientific freeze.

- [ ] G1: el repositorio V2 y CEDIA tienen una revisión trazable y reproducible
  CHECK: python scripts/verify_v2.py repository
  EXPECT: V2 repository gate passed
  EVIDENCE: pending

- [ ] G2: datos, folds, manifiestos y detector YOLOv7 cumplen D01–D25 sin leakage
  CHECK: python scripts/verify_v2.py detector
  EXPECT: V2 detector gate passed
  EVIDENCE: pending

- [ ] G3: los segmentadores elegibles y el TOP-3 V2 cumplen D26–D32
  CHECK: python scripts/verify_v2.py selection
  EXPECT: V2 selection gate passed
  EVIDENCE: pending

- [ ] G4: colorimetría, clean-skin y freeze cumplen D33–D42
  CHECK: python scripts/verify_v2.py freeze
  EXPECT: V2 freeze gate passed
  EVIDENCE: pending

- [ ] G5: Test sellado y experimento MST cumplen D43–D49 después del freeze
  CHECK: python scripts/verify_v2.py mst
  EXPECT: V2 MST gate passed
  EVIDENCE: pending

- [ ] G6: el censo y análisis MSKCC cumplen D50–D51 y la decisión de concordancia aprobada
  CHECK: python scripts/verify_v2.py mskcc
  EXPECT: V2 MSKCC gate passed
  EVIDENCE: pending

- [ ] G7: un único run.sh orquesta todas las etapas y todos los jobs científicos prueban compute-0-2 con compute-0-1 excluido
  CHECK: python scripts/verify_v2.py pipeline
  EXPECT: V2 pipeline gate passed
  EVIDENCE: pending

- [ ] G8: bitácora, hashes, jobs, commits y artefactos permiten auditar toda la ejecución
  CHECK: python scripts/verify_v2.py provenance
  EXPECT: V2 provenance gate passed
  EVIDENCE: pending

