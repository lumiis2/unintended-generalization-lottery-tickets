#!/usr/bin/env bash
# Submit all completed cat/owl runs; each task is independent and resumable.
set -Eeuo pipefail

job=$(sbatch --parsable code/scripts/slurm/sl/sl_base_lora_decomposition_l40s.sbatch)
printf 'base_lora_decomposition=%s\n' "$job"
printf '%s\n' 'Array tasks: cat-47/48/49 and owl-47/48/49; max two concurrent L40S tasks.'
