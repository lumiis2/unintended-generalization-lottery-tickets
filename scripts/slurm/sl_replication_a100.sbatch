#!/usr/bin/env bash
#SBATCH --job-name=spar-sl-replication
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONFIG="${CONFIG:-configs/sl/qwen25_7b_cat_replication.yaml}"
cd "$PROJECT_ROOT"

source /etc/profile.d/lmod.sh
if [[ -f /etc/profile.d/z01_lmod_cluster.sh ]]; then
    source /etc/profile.d/z01_lmod_cluster.sh
elif [[ -f /etc/profile.d/z00_modulepath.sh ]]; then
    source /etc/profile.d/z00_modulepath.sh
fi
module purge || true

export PYTHONNOUSERSITE=1
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "date=$(date --iso-8601=seconds) commit=$(git rev-parse HEAD) job=${SLURM_JOB_ID} node=${SLURM_JOB_NODELIST}"
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 70 * 2**30
print(torch.__version__, torch.cuda.get_device_name(0))
PY

PID=""
on_usr1() {
    [[ -z "$PID" ]] || kill -TERM "$PID" 2>/dev/null || true
    [[ -z "$PID" ]] || wait "$PID" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap on_usr1 USR1

set +e
bash scripts/sl/run_replication.sh "$CONFIG" &
PID=$!
wait "$PID"
STATUS=$?
set -e
if (( STATUS == 0 )); then exit 0; fi
if (( ${SLURM_RESTART_COUNT:-0} < ${MAX_REQUEUES:-2} )); then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$STATUS"
