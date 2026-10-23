#!/usr/bin/env bash
# CPU-only post-hoc plots from existing result artifacts; no model loading.
#SBATCH --job-name=spar-sl-viz
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
echo "date=$(date --iso-8601=seconds) commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
python code/scripts/sl/plotting/plot_additional_sl_analysis.py
