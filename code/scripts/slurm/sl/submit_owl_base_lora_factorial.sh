#!/usr/bin/env bash
# Owl replication of the base × LoRA screen, with selection frozen on owl-47.
set -Eeuo pipefail

launcher="code/scripts/slurm/sl/sl_cat_base_lora_factorial_l40s.sbatch"
pipeline="code/scripts/sl/pipeline/run_owl_base_lora_factorial_stage.sh"
selection_file="artifacts/sl/qwen25_7b_owl_replication/base_component_localization_v1/selection.pt"
dependency=()
selection="already_frozen"
if [[ ! -f "$selection_file" ]]; then
    selection=$(sbatch --parsable --job-name=spar-owl-base-select --time=03:00:00 --array=0-0 --export=ALL,PIPELINE="$pipeline",STAGE=selection "$launcher")
    dependency=(--dependency="afterok:$selection")
fi
teacher=$(sbatch --parsable --job-name=spar-owl-base-teacher --time=04:00:00 --array=0-0 "${dependency[@]}" --export=ALL,PIPELINE="$pipeline",STAGE=shared_teacher "$launcher")
factorial=$(sbatch --parsable --job-name=spar-owl-base-factorial --time=16:00:00 --array=0-3%2 "${dependency[@]}" --export=ALL,PIPELINE="$pipeline",STAGE=factorial "$launcher")
printf 'selection=%s\nshared_teacher=%s\nfactorial=%s\n' "$selection" "$teacher" "$factorial"
printf '%s\n' 'Owl-47 selection is reused when already frozen; owl-48/49 never enter the selection step.'
