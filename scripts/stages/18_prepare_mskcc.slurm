#!/usr/bin/env bash
#SBATCH --job-name=v2-mskcc-data
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
root=data/raw/mskcc; mkdir -p "$root" artifacts/mskcc
url=$(scripts/hpc/run_in_container.sh --cpu -- python -c 'import json; print(json.load(open("configs/methodology_v2.json"))["mskcc_bundle_url"])' | tail -1)
test -f "$root/bundle.zip" || curl -fL --retry 3 -o "$root/.bundle.zip.tmp" "$url"
if [[ -f "$root/.bundle.zip.tmp" ]]; then mv "$root/.bundle.zip.tmp" "$root/bundle.zip"; fi
unzip -n "$root/bundle.zip" -d "$root"
scripts/hpc/run_in_container.sh --cpu -- python scripts/mskcc.py prepare --root "$root" --output artifacts/mskcc/census.json
echo V2_MSKCC_CENSUS_COMPLETE
