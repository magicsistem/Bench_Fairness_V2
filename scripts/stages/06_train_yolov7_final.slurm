#!/usr/bin/env bash
#SBATCH --job-name=v2-yolo-final
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
last=artifacts/yolov7/final/weights/last.pt
if [[ -f "$last" ]]; then args=(--resume "$last"); else
  args=(--workers 16 --device 0 --batch-size 32 --data artifacts/development/yolo-final/dataset.yaml
        --img-size 640 640 --cfg vendor/yolov7/cfg/training/yolov7.yaml --weights models/yolov7/yolov7_training.pt
        --name final --hyp configs/yolov7_hyp_v2.yaml --epochs 300 --project artifacts/yolov7 --exist-ok)
fi
scripts/hpc/run_in_container.sh -- python vendor/yolov7/train.py "${args[@]}"
test -f artifacts/yolov7/final/weights/best.pt
echo V2_YOLO_FINAL_COMPLETE
