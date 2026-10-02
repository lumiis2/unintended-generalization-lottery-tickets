#!/usr/bin/env bash
#SBATCH --job-name=spar-sl
#SBATCH --partition=new_p_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
CONFIG="${CONFIG:-configs/sl/qwen25_7b_pilot.yaml}"
STAGE="${STAGE:?Set STAGE to evaluate, generate, train, or summarize}"
CONDITION="${CONDITION:-}"
cd "$PROJECT_ROOT"

# Non-interactive SSH/Slurm shells do not initialize Lmod on this cluster.
source /etc/profile.d/lmod.sh
source /etc/profile.d/z01_lmod_cluster.sh
module purge
source /etc/profile.d/z01_lmod_cluster.sh
module load pytorch/2.5.1-py3.10-cuda12.8
export PYTHONNOUSERSITE=1
source "${SPAR_VENV:-.venv310}/bin/activate"

# Prefer packages installed in the venv, then fall back to the PyTorch module.
VENV_SITE="${VIRTUAL_ENV}/lib/python3.10/site-packages"
export PYTHONPATH="${PROJECT_ROOT}/src:${VENV_SITE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "date=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURM_JOB_NODELIST}"
echo "stage=${STAGE}"
echo "condition=${CONDITION:-none}"
echo "config=${CONFIG}"

python - <<'PY'
import torch
print("torch_version=", torch.__version__)
print("torch_file=", torch.__file__)
print("torch_cuda_build=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("visible_gpu_count=", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed")
x = torch.tensor([[1.0, 2.0]], device="cuda")
y = torch.tensor([[3.0], [4.0]], device="cuda")
assert (x @ y).item() == 11.0
print("gpu_name=", torch.cuda.get_device_name(0))
print("CUDA PREFLIGHT PASSED")
PY

case "$STAGE" in
  evaluate)
    test -n "$CONDITION"
    srun --kill-on-bad-exit=1 python scripts/sl/evaluate_preference.py --config "$CONFIG" --condition "$CONDITION"
    ;;
  generate)
    test -n "$CONDITION"
    srun --kill-on-bad-exit=1 python scripts/sl/generate_numbers.py --config "$CONFIG" --condition "$CONDITION"
    ;;
  train)
    test -n "$CONDITION"
    srun --kill-on-bad-exit=1 python scripts/sl/train_student.py --config "$CONFIG" --condition "$CONDITION"
    ;;
  summarize)
    srun --kill-on-bad-exit=1 python scripts/sl/summarize_run.py --config "$CONFIG"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
