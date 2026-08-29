#!/usr/bin/env bash
#SBATCH --job-name=v2-finalize
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"; cd "$PROJECT_ROOT"; [[ $(hostname -s) == compute-0-2 ]]
scripts/hpc/run_in_container.sh --cpu -- python scripts/verify_v2.py detector
scripts/hpc/run_in_container.sh --cpu -- python scripts/verify_v2.py selection
scripts/hpc/run_in_container.sh --cpu -- python scripts/verify_v2.py freeze
scripts/hpc/run_in_container.sh --cpu -- python scripts/verify_v2.py mst
scripts/hpc/run_in_container.sh --cpu -- python scripts/verify_v2.py mskcc
scripts/hpc/run_in_container.sh --cpu -- python scripts/finalize.py
scripts/hpc/run_in_container.sh --cpu -- python scripts/verify_v2.py provenance
echo V2_COMPLETE
