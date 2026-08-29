#!/usr/bin/env bash
#SBATCH --job-name=v2-generate-mst
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh --cpu -- python scripts/mst.py generate --rois artifacts/test/rois.json \
  --config configs/methodology_v2.json --margin artifacts/color/lesion_margin.json --output artifacts/test/mst
echo V2_MST_GENERATION_COMPLETE
