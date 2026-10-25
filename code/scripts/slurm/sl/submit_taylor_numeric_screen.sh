#!/usr/bin/env bash
# Submit the exploratory screen now; CPU plotting waits for all six runs.
set -Eeuo pipefail
screen=$(sbatch --parsable code/scripts/slurm/sl/sl_taylor_numeric_screen_l40s.sbatch)
plot=$(sbatch --parsable --dependency="afterok:${screen}" code/scripts/slurm/sl/sl_taylor_numeric_screen_plot_cpu.sbatch)
printf 'taylor_numeric_screen=%s\ntaylor_numeric_screen_plot=%s\n' "$screen" "$plot"
