#!/usr/bin/env bash
# Experiment 1: independent neutral controls and masked-WS activation paths.
# A task owns one run end-to-end, so WC2 is always complete before its analysis.
# It writes only new WC2 and validation directories and can resume training from
# checkpoints if Slurm requeues the task.
#SBATCH --job-name=spar-sl-act-div
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-5%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "date=$(date --iso-8601=seconds) commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.cuda.get_device_name(0))
PY

configs=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication_seed49.yaml
)
config="${configs[$SLURM_ARRAY_TASK_ID]:?Invalid array index}"

pid=""
on_usr1() {
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap on_usr1 USR1

set +e
bash code/scripts/sl/pipeline/run_activation_divergence_validation.sh "$config" &
pid=$!
wait "$pid"
status=$?
set -e
if (( status == 0 )); then exit 0; fi
if (( ${SLURM_RESTART_COUNT:-0} < ${MAX_REQUEUES:-2} )); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
