#!/usr/bin/env bash
# Submit the shared-teacher test and the cat-48/49 LoRA × base factorial.
# All inputs are frozen: the cat-47 base selector and v2 held-out assets.
set -Eeuo pipefail

launcher="code/scripts/slurm/sl/sl_cat_base_lora_factorial_l40s.sbatch"
selection="artifacts/sl/qwen25_7b_cat_replication/base_component_localization_v1/selection.pt"
[[ -f "$selection" ]] || { echo "Missing frozen selector: $selection" >&2; exit 1; }

teacher=$(sbatch --parsable --job-name=spar-base-teacher --time=04:00:00 --array=0-0 --export=ALL,STAGE=shared_teacher "$launcher")
factorial=$(sbatch --parsable --job-name=spar-base-factorial --time=16:00:00 --array=0-3%2 --export=ALL,STAGE=factorial "$launcher")
printf 'shared_teacher=%s\nfactorial=%s\n' "$teacher" "$factorial"
printf '%s\n' 'Both jobs are independent and run immediately: teacher/base sharing plus cat-48/49 trait/control factorial cells.'
