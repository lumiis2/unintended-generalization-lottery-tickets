#!/usr/bin/env bash
# Descriptive selected-mask overlap extension after owl-48/49 scores complete.
#SBATCH --job-name=spar-owl-overlap
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
python code/scripts/sl/plotting/plot_mask_overlap.py --owl-completion-only
