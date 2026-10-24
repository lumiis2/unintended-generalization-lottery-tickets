#!/usr/bin/env bash
# One completed SL run per task; all conditions are resumable JSON outputs.
#SBATCH --job-name=spar-mask-transfer
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "date=$(date --iso-8601=seconds) commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_properties(0).total_memory >= 44 * 2**30
print(torch.__version__, torch.cuda.get_device_name(0))
PY
pid=""
on_usr1() { [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true; [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_usr1 USR1
set +e
bash code/scripts/sl/masks/run_control_mask_transfer.sh "$SLURM_ARRAY_TASK_ID" &
pid=$!
wait "$pid"; status=$?
set -e
if (( status == 0 )); then exit 0; fi
if (( ${SLURM_RESTART_COUNT:-0} < ${MAX_REQUEUES:-2} )); then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
