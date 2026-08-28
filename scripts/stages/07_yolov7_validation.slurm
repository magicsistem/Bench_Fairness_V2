#!/usr/bin/env bash
#SBATCH --job-name=v2-yolo-val
#SBATCH --partition=gpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100-sxm4-40gb:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh -- python vendor/yolov7/detect.py \
  --weights artifacts/yolov7/final/weights/best.pt --source data/raw/isic2018_task1/ISIC2018_Task1-2_Validation_Input \
  --img-size 640 --conf-thres 0.25 --iou-thres 0.45 --device 0 --save-txt --save-conf --nosave \
  --project artifacts/yolov7 --name validation --exist-ok
scripts/hpc/run_in_container.sh --cpu -- python scripts/roi.py build \
  --manifest artifacts/development/validation_manifest.json --labels-root artifacts/yolov7/validation/labels \
  --margin artifacts/yolov7/roi_margin.json --output artifacts/yolov7/validation_rois.json
echo V2_YOLO_VALIDATION_COMPLETE
