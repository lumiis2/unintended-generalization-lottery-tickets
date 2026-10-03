#!/usr/bin/env bash
#SBATCH --job-name=spar-sl-bench
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
CONFIG="${CONFIG:-configs/sl/qwen25_7b_official_benchmark.yaml}"
STAGE="${STAGE:-all}"
cd "$PROJECT_ROOT"

source /etc/profile.d/lmod.sh
source /etc/profile.d/z01_lmod_cluster.sh
module purge
source /etc/profile.d/z01_lmod_cluster.sh
module load pytorch/2.5.1-py3.10-cuda12.8

export PYTHONNOUSERSITE=1
source "${SPAR_VENV:-.venv310}/bin/activate"
VENV_SITE="${VIRTUAL_ENV}/lib/python3.10/site-packages"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${VENV_SITE}:${PYTHONPATH:-}"
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
echo "stage=${STAGE}"

python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("visible_gpu_count=", torch.cuda.device_count())
print("gpu=", torch.cuda.get_device_name(0))
print("gpu_memory_gib=", torch.cuda.get_device_properties(0).total_memory / 2**30)
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 70 * 2**30
PY

MAX_REQUEUES="${MAX_REQUEUES:-2}"
BENCHMARK_PID=""

handle_usr1() {
    echo "Received USR1; requesting requeue for job ${SLURM_JOB_ID}" >&2
    if [[ -n "${BENCHMARK_PID}" ]]; then
        kill -TERM "${BENCHMARK_PID}" 2>/dev/null || true
        wait "${BENCHMARK_PID}" 2>/dev/null || true
    fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

trap handle_usr1 USR1

set +e
STAGE="$STAGE" bash scripts/sl/run_official_benchmark.sh "$CONFIG" &
BENCHMARK_PID=$!
wait "$BENCHMARK_PID"
status=$?
set -e

if (( status == 0 )); then
    echo "SL OFFICIAL-PROTOCOL BENCHMARK COMPLETED"
    exit 0
fi

restart_count="${SLURM_RESTART_COUNT:-0}"
if (( restart_count < MAX_REQUEUES )); then
    echo "Benchmark failed with status=${status}; requeueing." >&2
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
exit "$status"
