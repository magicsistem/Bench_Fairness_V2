# Bench Fairness V2

Implementación reproducible de la metodología D01–D58. El documento canónico
es `../Guia/10_METHODOLOGY_V2_FROM_ZERO.md`; este repositorio contiene código,
configuración y contratos, nunca credenciales ni resultados científicos.

```bash
./run.sh all       # CEDIA: envía la cadena autorizada
./run.sh status    # consulta sin modificar
./run.sh resume    # reanuda fallos compatibles por hash
```

Todo job científico exige `compute-0-2` y excluye `compute-0-1`. ISIC 2018
Test permanece inaccesible hasta un `scientific_freeze.json` válido.

