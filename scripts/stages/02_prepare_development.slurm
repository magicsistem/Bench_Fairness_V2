#!/usr/bin/env bash
#SBATCH --job-name=v2-prepare-dev
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"
[[ $(hostname -s) == compute-0-2 ]]
root=data/raw/isic2018_task1
out=artifacts/development
mkdir -p "$out"
run=(scripts/hpc/run_in_container.sh --cpu -- env PYTHONPATH=src python scripts/prepare_data.py)
"${run[@]}" manifest --split training --images "$root/ISIC2018_Task1-2_Training_Input" --masks "$root/ISIC2018_Task1_Training_GroundTruth" --output "$out/training_manifest.json"
"${run[@]}" manifest --split validation --images "$root/ISIC2018_Task1-2_Validation_Input" --masks "$root/ISIC2018_Task1_Validation_GroundTruth" --output "$out/validation_manifest.json"
"${run[@]}" folds --manifest "$out/training_manifest.json" --metadata "$root/metadata" --output "$out/folds.json" --seed 20260828
"${run[@]}" yolo --manifest "$out/training_manifest.json" --folds "$out/folds.json" --output "$out/yolo"
"${run[@]}" yolo-final --training "$out/training_manifest.json" --validation "$out/validation_manifest.json" --output "$out/yolo-final"
echo V2_DEVELOPMENT_PREPARED
