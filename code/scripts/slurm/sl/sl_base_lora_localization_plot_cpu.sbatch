#!/usr/bin/env bash
#SBATCH --job-name=spar-base-lora-plots
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?}"; cd "$root"; source /etc/profile.d/lmod.sh; module purge || true; source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}" MPLBACKEND=Agg
python code/scripts/sl/plotting/plot_base_and_lora_localization.py
