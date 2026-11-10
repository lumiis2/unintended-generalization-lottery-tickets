#!/usr/bin/env bash
# Corrected broad base-weight column screen, six runs, resumable per condition.
#SBATCH --job-name=spar-base-columns-v2
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --array=0-5%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?}"; cd "$root"; source /etc/profile.d/lmod.sh; module purge || true; source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}" HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
configs=(code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml)
python code/scripts/sl/masks/evaluate_base_column_masks.py --config "${configs[$SLURM_ARRAY_TASK_ID]}" --output-name base_column_masking_v2
