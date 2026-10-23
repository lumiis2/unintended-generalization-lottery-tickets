#!/usr/bin/env bash
# Submit the complete owl-48/49 suite with safe dependencies.
# Submit this file from the repository root after pulling the committed code.
set -Eeuo pipefail

launcher="code/scripts/slurm/sl/sl_owl_completion_l40s.sbatch"
submit_stage() {
    local stage="$1" dependency="$2" time="$3"
    local args=(--parsable --job-name="spar-owl-${stage}" --time="$time" --export="ALL,STAGE=$stage")
    if [[ -n "$dependency" ]]; then
        args+=(--dependency="afterok:$dependency")
    fi
    sbatch "${args[@]}" "$launcher"
}

train=$(submit_stage replication "" 08:00:00)
scores=$(submit_stage scores "$train" 04:00:00)
activations=$(submit_stage activations "$train" 04:00:00)
masks=$(submit_stage masks "$scores" 16:00:00)
causality=$(submit_stage causality "$activations" 06:00:00)
coherence=$(sbatch --parsable --job-name=spar-owl-coherence --dependency="afterok:$masks" code/scripts/slurm/sl/sl_owl_coherence_l40s.sbatch)
overlap=$(sbatch --parsable --job-name=spar-owl-overlap --dependency="afterok:$scores" code/scripts/slurm/sl/sl_owl_overlap_cpu.sbatch)

printf 'replication=%s\nmask_scores=%s\nactivation_scan=%s\nfinal_masks=%s\nactivation_causality=%s\ncoherence=%s\noverlap=%s\n' \
    "$train" "$scores" "$activations" "$masks" "$causality" "$coherence" "$overlap"
printf '%s\n' 'Each array has owl-48 and owl-49 in parallel; final masks and activation work run in parallel after their own prerequisites.'
