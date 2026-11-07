#!/usr/bin/env bash
# Aggregate and plot Experiment 1 after the six GPU array tasks finish.
#SBATCH --job-name=spar-sl-act-div-plot
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export MPLBACKEND=Agg
python code/scripts/sl/plotting/plot_activation_divergence_validation.py
