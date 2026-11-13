#!/usr/bin/env bash
# Freeze preservation-safe targeted base conditions without reading SL outcomes.
#SBATCH --job-name=spar-targeted-base-gate
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?}"; cd "$root"; source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root/code/scripts/sl/masks:$root:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
python code/scripts/sl/masks/analyze_targeted_base_calibration.py
