#!/usr/bin/env bash
# Generic Slurm wrapper for the preregistered owl-48/49 suite.  The submitter
# overrides wall time per stage and chains stages with afterok dependencies.
#SBATCH --job-name=spar-sl-owl
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-1%2
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
stage="${STAGE:?Set STAGE to replication, scores, masks, activations, or causality}"
cd "$root"

source /etc/profile.d/lmod.sh
if [[ -f /etc/profile.d/z01_lmod_cluster.sh ]]; then
    source /etc/profile.d/z01_lmod_cluster.sh
elif [[ -f /etc/profile.d/z00_modulepath.sh ]]; then
    source /etc/profile.d/z00_modulepath.sh
fi
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "date=$(date --iso-8601=seconds) commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID stage=$stage"
python - <<'PY'
import torch
assert torch.cuda.is_available()
# L40S is marketed as 48 GB decimal, i.e. about 44.7 GiB reported by PyTorch.
# Keep the guard above 24 GB-class GPUs without rejecting a valid L40S.
assert torch.cuda.get_device_properties(0).total_memory >= 44 * 2**30
print(torch.__version__, torch.cuda.get_device_name(0))
PY

pid=""
on_usr1() {
    [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap on_usr1 USR1

set +e
bash code/scripts/sl/pipeline/run_owl_completion_stage.sh "$stage" "$SLURM_ARRAY_TASK_ID" &
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
