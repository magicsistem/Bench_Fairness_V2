#!/usr/bin/env bash
#SBATCH --job-name=v2-mskcc-icc-d61
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh --cpu -- python -m unittest discover -s tests -p test_d61_icc.py -v
scripts/hpc/run_in_container.sh --cpu -- python scripts/mskcc_icc_d61.py \
  --census artifacts/mskcc/census.json \
  --color results/mskcc_color/colorimetry.json \
  --historical results/mskcc_analysis.json \
  --invariants configs/d61_historical_invariants.json \
  --output results/mskcc_icc_d61.json \
  --global-csv results/mskcc_icc_d61_global.csv \
  --site-csv results/mskcc_icc_d61_by_anatomical_site.csv \
  --diagnostics-csv results/mskcc_icc_d61_diagnostics.csv \
  --manifest artifacts/mskcc/icc_d61_manifest.json \
  --provenance artifacts/final/provenance_d61.json \
  --repetitions 10000 --seed 20260828
echo D61_MSKCC_ICC_STAGE_COMPLETE
