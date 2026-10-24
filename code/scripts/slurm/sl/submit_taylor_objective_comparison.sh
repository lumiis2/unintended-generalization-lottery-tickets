#!/usr/bin/env bash
# First compute all trait-blind scores, then run only CPU overlap plots.
set -Eeuo pipefail
scores=$(sbatch --parsable code/scripts/slurm/sl/sl_taylor_numeric_score_l40s.sbatch)
plots=$(sbatch --parsable --dependency="afterok:$scores" code/scripts/slurm/sl/sl_taylor_objective_overlap_cpu.sbatch)
full_overlap=$(sbatch --parsable code/scripts/slurm/sl/sl_full_mask_overlap_cpu.sbatch)
printf 'taylor_numeric_scores=%s\ntaylor_objective_plots=%s\nrepaired_four_method_overlap=%s\n' "$scores" "$plots" "$full_overlap"
printf '%s\n' 'The score array covers six runs × trait/control adapters; the Taylor plot job starts only after every score succeeds. The repaired existing four-method overlap plot is CPU-only and runs independently.'
