#!/usr/bin/env bash
# Complete the original four-method exploratory mask protocol for owl-48/49.
# The existing Magnitude and Taylor-contrastive scores are reused; Wanda and
# Taylor-trait are added, then the resumable broad causal screen is evaluated.
set -Eeuo pipefail

launcher="code/scripts/slurm/sl/sl_owl_completion_l40s.sbatch"
submit_array() {
    local name="$1" stage="$2" dependency="$3" time="$4"
    local args=(--parsable --job-name="$name" --time="$time" --export="ALL,STAGE=$stage")
    if [[ -n "$dependency" ]]; then
        args+=(--dependency="afterok:$dependency")
    fi
    sbatch "${args[@]}" "$launcher"
}

scores=$(submit_array spar-owl-full-scores full_screen_scores "" 08:00:00)
screen=$(submit_array spar-owl-full-screen full_screen "$scores" 16:00:00)
overlap=$(sbatch --parsable --job-name=spar-owl-full-overlap --dependency="afterok:$scores" \
    code/scripts/slurm/sl/sl_full_mask_overlap_cpu.sbatch)

printf 'full_scores=%s\ncausal_screen=%s\noverlap=%s\n' "$scores" "$screen" "$overlap"
printf '%s\n' 'Each array has owl-48 and owl-49 in parallel. The causal screen starts only after both score tasks finish; plotting starts after scores and does not wait for behavioural evaluations.'
