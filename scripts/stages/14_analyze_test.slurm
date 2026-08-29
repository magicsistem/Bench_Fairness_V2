#!/usr/bin/env bash
#SBATCH --job-name=v2-analyze-test
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
scripts/hpc/run_in_container.sh --cpu -- python scripts/analyze_test.py --rois artifacts/test/rois.json \
  --selection artifacts/selection/top3.json --results results/test_original --margin artifacts/color/lesion_margin.json \
  --output results/test_original/robustness_ita.json --seed 20260828
echo V2_TEST_ANALYSIS_COMPLETE
