#!/usr/bin/env bash
# Held-out targeted base-column SL screen. Submit only after calibration review.
# Six completed SL runs x sixteen frozen candidate scopes.
#SBATCH --job-name=spar-targeted-base-screen
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --array=0-95%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?}"; cd "$root"; source /etc/profile.d/lmod.sh; module purge || true; source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}" HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
configs=(code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml)
scopes=(l3_down_proj l3_gate_proj l3_up_proj l4_down_proj l4_gate_proj l4_up_proj l5_down_proj l5_gate_proj l5_up_proj l3_o_proj l4_o_proj l5_o_proj early_0_2_mlp early_6_8_mlp middle_13_15_mlp late_21_23_mlp)
scope_count=${#scopes[@]}
config_index=$((SLURM_ARRAY_TASK_ID / scope_count))
scope_index=$((SLURM_ARRAY_TASK_ID % scope_count))
python code/scripts/sl/masks/calibrate_targeted_base_masks.py \
  --mode screen --random-replicates 5 --config "${configs[$config_index]}" --scope "${scopes[$scope_index]}"
