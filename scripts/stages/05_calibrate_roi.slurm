#!/usr/bin/env bash
#SBATCH --job-name=v2-roi-q95
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh --cpu -- python scripts/roi.py calibrate \
  --manifest artifacts/development/training_manifest.json --folds artifacts/development/folds.json \
  --labels-root artifacts/yolov7/oof --output artifacts/yolov7/roi_margin.json
echo V2_ROI_Q95_COMPLETE
