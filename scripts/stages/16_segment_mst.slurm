#!/usr/bin/env bash
#SBATCH --job-name=v2-seg-mst
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
method=$(scripts/hpc/run_in_container.sh --cpu -- python -c 'import json,sys; print(json.load(open("artifacts/selection/top3.json"))["top3"][int(sys.argv[1])])' "$SLURM_ARRAY_TASK_ID" | tail -1)
scripts/hpc/run_in_container.sh -- python scripts/evaluate_segmenter.py --rois artifacts/test/mst_rois.json \
  --output results/test_mst --method-id "$method" --device cuda
echo V2_SEGMENT_MST_COMPLETE=$method
