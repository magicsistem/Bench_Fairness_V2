# Trazabilidad de reutilización

Fuente: `Thesis_Fitzpatrick_hpc_repair`, commit
`a660ecf1539a2477117a850e18d6cc13aaedd534`, rama auditada
`fix/hair-component-scan-performance`.

| Destino V2 | Origen | Uso V2 |
|---|---|---|
| `src/thesis_fitzpatrick/metrics.py` | mismo path | primitivas métricas; el ranking V2 se calcula de nuevo |
| `src/thesis_fitzpatrick/preprocessing.py` | mismo path | `BBox` y detector morfológico; V2 no llama FOV, YOLOv3 ni inpainting |
| `src/thesis_fitzpatrick/masks.py` | mismo path | conversión sRGB→Lab auditada; clean-skin V1 no se usa |
| `src/thesis_fitzpatrick/grabcut.py` | mismo path | backend GrabCut; no se usa posprocesamiento común |
| `scripts/adapters/*.py` | mismos paths | 15 adaptadores con checkpoint nativo |
| `configs/segmentation_models.json` | mismo path | roster D57; prioridades V1 no ordenan resultados V2 |
| `configs/hpc/external_resources.json` | mismo path | URLs/hashes auditados; YOLOv3 e IMA++ quedan ignorados |

No se copiaron `results/`, predicciones, métricas, pesos YOLOv3, identidades
S01–S16 ni el TOP-3 histórico.

