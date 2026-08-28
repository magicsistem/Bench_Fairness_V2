#!/usr/bin/env bash
#SBATCH --job-name=v2-yolo-cv
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"
[[ $(hostname -s) == compute-0-2 ]]
fold=${SLURM_ARRAY_TASK_ID:?array task required}
name="fold-$fold"
last="artifacts/yolov7/cv/$name/weights/last.pt"
if [[ -f "$last" ]]; then
    args=(--resume "$last")
else
    args=(--workers 16 --device 0 --batch-size 32 --data "artifacts/development/yolo/$name/dataset.yaml"
          --img-size 640 640 --cfg vendor/yolov7/cfg/training/yolov7.yaml
          --weights models/yolov7/yolov7_training.pt --name "$name"
          --hyp configs/yolov7_hyp_v2.yaml --epochs 300 --project artifacts/yolov7/cv --exist-ok)
fi
export RUNTIME_DIAGNOSTICS_LOG="$PROJECT_ROOT/.cedia/diagnostics/yolov7-cv-${SLURM_ARRAY_JOB_ID}-${fold}.log"
scripts/hpc/run_in_container.sh -- python vendor/yolov7/train.py "${args[@]}"
test -f "artifacts/yolov7/cv/$name/weights/best.pt"
echo V2_YOLO_CV_FOLD_COMPLETE=$fold

