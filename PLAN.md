# Plan de reconstrucción exhaustiva del informe V2

Contrato: guía canónica `10_METHODOLOGY_V2_FROM_ZERO.md` + canon ejecutable `configs/methodology_v2.json`. El código Git y los artefactos de CEDIA son evidencia de ejecución, no fuentes para inventar decisiones.

| ID | Entregable | Estado | Gate |
|---|---|---|---|
| R1 | Inventario completo D01–D60, secciones 2–18 y eventos ejecutados relevantes de la sección 19 | completed | G11 |
| R2 | Matriz de las 37 referencias, releídas en su fuente y vinculadas a las afirmaciones que respaldan | completed | G12 |
| R3 | Inventario verificable de artefactos y resultados reales extraídos de CEDIA | completed | G13 |
| R4 | Metodología exhaustiva con fórmulas, algoritmos, ramas, variables y justificación | completed | G14 |
| R5 | Resultados completos y resúmenes por etapa, subpaso, rama, ablación y variación | completed | G15 |
| R6 | Auditoría final de cobertura, cifras, referencias, Markdown matemático, pruebas, commit, push y sección 19 | completed | G16 |

Reglas: no generar resultados científicos nuevos; no revisar imágenes; no modificar decisiones; usar `NA` cuando el canon o los artefactos no definan una cifra; no ocultar fallos ni intentos históricos.

## Extensión D61 — concordancia ICC en MSKCC

Contrato: conservar intactos D01–D60, TOP-3, scientific freeze y todos los resultados históricos; reutilizar exclusivamente pares MSKCC con procedencia válida; ejecutar el nuevo análisis científico mediante Slurm en `compute-0-2`.

| ID | Entregable | Estado | Gate |
|---|---|---|---|
| R61.1 | Extender al final el canon documental y ejecutable con D61 y referencias [38]–[39] | completed | G61.11–G61.12 |
| R61.2 | Implementar ICC(1,1), ICC(2,1), ICC(3,1), estados de estimabilidad y bootstrap por paciente | in_progress | G61.1, G61.5, G61.6, G61.10 |
| R61.3 | Probar fórmulas, offset, ruido, degeneración y reproducción del ICC(2,1) histórico | pending | G61.2, G61.14 |
| R61.4 | Ejecutar en CEDIA el análisis global y por sitio anatómico sin repetir inferencia | pending | G61.3–G61.9 |
| R61.5 | Generar JSON, CSV, tablas, hashes y provenance D61 sin sobrescribir históricos | pending | G61.13, G61.15 |
| R61.6 | Actualizar el informe completo, auditar invariantes, gates, commit y push | pending | G61.11, G61.16 |
