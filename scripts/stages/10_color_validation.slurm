#!/usr/bin/env bash
#SBATCH --job-name=v2-color-val
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
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
base=(--rois artifacts/yolov7/validation_rois.json --selection artifacts/selection/top3.json --results results/validation)
scripts/hpc/run_in_container.sh --cpu -- python scripts/colorimetry.py calibrate "${base[@]}" --output artifacts/color/lesion_margin.json
scripts/hpc/run_in_container.sh --cpu -- python scripts/colorimetry.py measure "${base[@]}" --margin artifacts/color/lesion_margin.json --output results/color_validation
scripts/hpc/run_in_container.sh --cpu -- python scripts/mst.py palette --config configs/methodology_v2.json --output artifacts/mst/palette.png
echo V2_COLOR_VALIDATION_COMPLETE
