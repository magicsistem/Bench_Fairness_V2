#!/usr/bin/env bash
#SBATCH --job-name=v2-analyze-mst
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh --cpu -- python scripts/analyze_mst.py --original results/test_original --mst results/test_mst \
  --selection artifacts/selection/top3.json --output results/test_mst/analysis.json --seed 20260828
echo V2_MST_ANALYSIS_COMPLETE
