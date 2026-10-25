#!/usr/bin/env bash
#SBATCH --job-name=spar-taylor-numeric-transfer-plot
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}" MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/spar-mpl}"
mkdir -p "$MPLCONFIGDIR"
python code/scripts/sl/plotting/plot_control_mask_transfer.py \
  --prefix transfer-v2-numeric \
  --method taylor_numeric_feature \
  --title Taylor-numeric \
  --output-dir artifacts/sl/mask_analysis/taylor_numeric_transfer_v2
