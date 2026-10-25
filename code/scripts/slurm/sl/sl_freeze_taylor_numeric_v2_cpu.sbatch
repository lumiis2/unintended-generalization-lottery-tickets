#!/usr/bin/env bash
# Freeze remote datasets into the already pre-registered v2 manifest.
#SBATCH --job-name=spar-freeze-taylor-v2
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
python code/scripts/sl/evaluation/freeze_final_evaluation_data.py \
  --output-root experiments/subliminal_learning/frozen_evaluation/taylor_numeric_v2 \
  --seed 20261025
