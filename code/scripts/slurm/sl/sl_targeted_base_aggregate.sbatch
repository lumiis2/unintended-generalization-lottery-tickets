#!/usr/bin/env bash
# Aggregate Experiment A after every held-out screen array task succeeds.
#SBATCH --job-name=spar-targeted-base-aggregate
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?}"; cd "$root"; source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}" MPLBACKEND=Agg PYTHONUNBUFFERED=1
python code/scripts/sl/plotting/analyze_targeted_base_screen.py
