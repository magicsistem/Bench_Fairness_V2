# Bench_Fairness_V2: reglas del proyecto para agentes

## Planificación y verificación

- El trabajo se organiza con unlazy: entregables en `PLAN.md` y gates en `GATES.md`, con el formato existente (`CHECK:`, `EXPECT:`, `EVIDENCE:`).
- Las decisiones científicas se numeran (D01, D02…) y las referencias son IEEE numéricas, con numeración continua respecto al canon indicado en `PLAN.md`. Las referencias nuevas se añaden al final.
- Invariantes: no modificar las decisiones ya cerradas, el TOP-3, `scientific_freeze.json` ni los resultados históricos. No abrir el Test sellado sin autorización explícita de Miguel.

## Ejecución en CEDIA

- Repositorio en el HPC: `~/Bench_Fairness_V2` (alias SSH `cedia`).
- Slurm: solo el nodo `compute-0-2`; `compute-0-1` siempre excluido.
- Usa los mecanismos existentes: `run.sh`, `scripts/hpc/run_in_container.sh` y los `.slurm` de `.cedia/` como referencia. Léelos antes de lanzar trabajos.
- Registra cada trabajo en `.cedia/jobs.tsv` respetando su formato.
- Protegidos en el HPC (no rastreados por git): `models/` y `vendor/`. Nunca borrarlos ni moverlos.
