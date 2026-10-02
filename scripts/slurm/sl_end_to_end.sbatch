#!/usr/bin/env bash
#SBATCH --job-name=spar-sl-smoke
#SBATCH --partition=new_p_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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

python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
PY

bash scripts/sl/run_pilot.sh "$CONFIG"
echo "SL END-TO-END RUN COMPLETED"
