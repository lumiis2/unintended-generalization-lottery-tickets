#!/usr/bin/env bash
#SBATCH --job-name=spar-taylor-overlap
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/spar-mpl}"
mkdir -p "$MPLCONFIGDIR"
python code/scripts/sl/plotting/plot_taylor_objective_overlap.py
