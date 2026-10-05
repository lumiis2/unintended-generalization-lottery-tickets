#!/usr/bin/env bash
# One array task per independently trained SL replication.
#SBATCH --job-name=spar-sl-causal-mask
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
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
export PYTHONPATH="$root/src:$root:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
echo "date=$(date --iso-8601=seconds) commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"

pid=""
trap '[[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true; [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
set +e
bash scripts/sl/run_causal_mask_screen.sh "$SLURM_ARRAY_TASK_ID" &
pid=$!
wait "$pid"
status=$?
set -e
if (( status == 0 )); then exit 0; fi
if (( ${SLURM_RESTART_COUNT:-0} < ${MAX_REQUEUES:-2} )); then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
