#!/usr/bin/env bash
#SBATCH --job-name=v2-color-mskcc
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
scripts/hpc/run_in_container.sh --cpu -- python scripts/colorimetry.py measure --rois artifacts/mskcc/rois.json \
  --selection artifacts/selection/top3.json --results results/mskcc --margin artifacts/color/lesion_margin.json --output results/mskcc_color
echo V2_COLOR_MSKCC_COMPLETE
