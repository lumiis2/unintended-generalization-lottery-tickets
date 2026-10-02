#!/usr/bin/env bash
#SBATCH --job-name=spar-sl-smoke
#SBATCH --partition=new_p_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONFIG="${CONFIG:-configs/sl/qwen25_7b_smoke.yaml}"
cd "$PROJECT_ROOT"

source /etc/profile.d/lmod.sh
source /etc/profile.d/z01_lmod_cluster.sh
module purge
source /etc/profile.d/z01_lmod_cluster.sh
module load pytorch/2.5.1-py3.10-cuda12.8

export PYTHONNOUSERSITE=1
source "${SPAR_VENV:-.venv310}/bin/activate"
VENV_SITE="${VIRTUAL_ENV}/lib/python3.10/site-packages"
export PYTHONPATH="${PROJECT_ROOT}/src:${VENV_SITE}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "date=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "config=${CONFIG}"
echo "hf_home=${HF_HOME}"
echo "restart_count=${SLURM_RESTART_COUNT:-0}"

python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY

MAX_REQUEUES="${MAX_REQUEUES:-2}"
PIPELINE_PID=""

handle_usr1() {
    echo "Received USR1; requesting requeue for job ${SLURM_JOB_ID}" >&2
    if [[ -n "${PIPELINE_PID}" ]]; then
        kill -TERM "${PIPELINE_PID}" 2>/dev/null || true
        wait "${PIPELINE_PID}" 2>/dev/null || true
    fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

handle_term() {
    echo "Received termination; stopping without automatic requeue" >&2
    if [[ -n "${PIPELINE_PID}" ]]; then
        kill -TERM "${PIPELINE_PID}" 2>/dev/null || true
        wait "${PIPELINE_PID}" 2>/dev/null || true
    fi
    exit 143
}

trap handle_usr1 USR1
trap handle_term TERM INT

set +e
bash scripts/sl/run_pilot.sh "$CONFIG" &
PIPELINE_PID=$!
wait "${PIPELINE_PID}"
pipeline_status=$?
set -e
PIPELINE_PID=""

if (( pipeline_status == 0 )); then
    echo "SL END-TO-END RUN COMPLETED"
    exit 0
fi

# Gate failures are scientific outcomes, not transient infrastructure errors.
if (( pipeline_status >= 20 && pipeline_status <= 29 )); then
    echo "Pipeline stopped at an experimental gate (status=${pipeline_status}); not requeueing." >&2
    exit "${pipeline_status}"
fi

restart_count="${SLURM_RESTART_COUNT:-0}"
if (( restart_count < MAX_REQUEUES )); then
    echo "Pipeline failed with status=${pipeline_status}; requeueing (${restart_count}/${MAX_REQUEUES})." >&2
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi

echo "Pipeline failed with status=${pipeline_status}; requeue limit exhausted." >&2
exit "${pipeline_status}"
