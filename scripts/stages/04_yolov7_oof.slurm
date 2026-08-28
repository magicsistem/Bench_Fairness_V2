#!/usr/bin/env bash
#SBATCH --job-name=v2-yolo-oof
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=12:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
fold=${SLURM_ARRAY_TASK_ID:?array task required}
scripts/hpc/run_in_container.sh -- python vendor/yolov7/detect.py \
  --weights "artifacts/yolov7/cv/fold-$fold/weights/best.pt" \
  --source "artifacts/development/yolo/fold-$fold/images/val" --img-size 640 \
  --conf-thres 0.25 --iou-thres 0.45 --device 0 --save-txt --save-conf --nosave \
  --project artifacts/yolov7/oof --name "fold-$fold" --exist-ok
test -d "artifacts/yolov7/oof/fold-$fold/labels"
echo V2_YOLO_OOF_FOLD_COMPLETE=$fold
