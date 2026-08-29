#!/usr/bin/env bash
#SBATCH --job-name=v2-analyze-mskcc
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh --cpu -- python scripts/mskcc.py analyze --census artifacts/mskcc/census.json \
  --color results/mskcc_color/colorimetry.json --output results/mskcc_analysis.json --repetitions 10000 --seed 20260828
echo V2_MSKCC_ANALYSIS_COMPLETE
