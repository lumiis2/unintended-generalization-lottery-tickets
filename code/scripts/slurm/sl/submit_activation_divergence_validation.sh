#!/usr/bin/env bash
# Submit Experiment 1 after the repository has been pushed and pulled on Slurm.
set -Eeuo pipefail

job=$(sbatch --parsable code/scripts/slurm/sl/sl_activation_divergence_validation_l40s.sbatch)
plot=$(sbatch --parsable --dependency="afterok:$job" code/scripts/slurm/sl/sl_activation_divergence_plot_cpu.sbatch)
printf 'activation_divergence_validation=%s\nactivation_divergence_plot=%s\n' "$job" "$plot"
printf '%s\n' 'Six runs (cat-47/48/49, owl-47/48/49); each independently trains WC2 then validates WS/WC/WC2/masked-WS. Max two L40S tasks run concurrently.'
