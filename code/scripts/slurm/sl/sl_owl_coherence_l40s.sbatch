#!/usr/bin/env bash
# Blind coherence judging for owl seeds 48 and 49 after final masks complete.
#SBATCH --job-name=spar-owl-coherence
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$root"
source /etc/profile.d/lmod.sh
module purge || true
source "${SPAR_VENV:-.venv-a100}/bin/activate"
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/code/src:$root:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/mnt/users/luisa.lopes/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

pid=""
trap '[[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true; [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
set +e
python code/scripts/sl/evaluation/judge_final_coherence.py \
    --batch-size "${JUDGE_BATCH_SIZE:-16}" \
    --output-root artifacts/sl/owl_completion_coherence_judge_v1 \
    --runs \
      owl48=artifacts/sl/qwen25_7b_owl_replication_seed48 \
      owl49=artifacts/sl/qwen25_7b_owl_replication_seed49 &
pid=$!
wait "$pid"; status=$?
set -e
if (( status == 0 )); then exit 0; fi
if (( ${SLURM_RESTART_COUNT:-0} < ${MAX_REQUEUES:-2} )); then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$status"
