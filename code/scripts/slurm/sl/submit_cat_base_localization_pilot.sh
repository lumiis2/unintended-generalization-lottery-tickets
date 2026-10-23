#!/usr/bin/env bash
# Candidate selection is frozen on cat-47; cat-48/49 then evaluate in parallel.
set -Eeuo pipefail
launcher="code/scripts/slurm/sl/sl_cat_base_localization_l40s.sbatch"
selection=$(sbatch --parsable --job-name=spar-cat-base-select --time=02:00:00 --export=ALL,STAGE=selection --array=0-0 "$launcher")
evaluation=$(sbatch --parsable --job-name=spar-cat-base-eval --time=06:00:00 --array=0-1%2 --dependency="afterok:$selection" --export=ALL,STAGE=evaluation "$launcher")
printf 'selection=%s\nevaluation=%s\n' "$selection" "$evaluation"
printf '%s\n' 'Selection is frozen on cat-47; cat-48 and cat-49 are evaluated in parallel after it succeeds.'
