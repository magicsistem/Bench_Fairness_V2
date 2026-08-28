#!/usr/bin/env bash
#SBATCH --job-name=v2-preflight
#SBATCH --partition=cpu
#SBATCH --nodelist=compute-0-2
#SBATCH --exclude=compute-0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -Eeuo pipefail
: "${PROJECT_ROOT:?PROJECT_ROOT required}"
cd "$PROJECT_ROOT"
[[ $(hostname -s) == compute-0-2 ]]
test ! -e data/raw/isic2018_task1/ISIC2018_Task1-2_Test_Input
test "$(stat -c %s "$HOME/pytorch_24.01-py3.sif")" = 10135441408
test "$(sha256sum "$HOME/pytorch_24.01-py3.sif" | cut -d' ' -f1)" = b9db68700a47ae0811e8c4758d6effc3221000dac53e5d05ece0a8cafd77e2a3
test -z "$(git status --porcelain --untracked-files=no)"
scripts/hpc/run_in_container.sh --cpu --no-venv -- env PYTHONPATH=src:. python -m unittest discover -s tests -v
scripts/hpc/run_in_container.sh --cpu --no-venv -- env PYTHONPATH=src python scripts/verify_v2.py repository
echo V2_PREFLIGHT_COMPLETE
