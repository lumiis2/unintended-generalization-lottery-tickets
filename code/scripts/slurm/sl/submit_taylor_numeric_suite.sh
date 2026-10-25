#!/usr/bin/env bash
# The exploratory screen and the new frozen final evaluation are independent.
# The final array starts only after its external benchmark assets are frozen.
set -Eeuo pipefail
screen=$(sbatch --parsable code/scripts/slurm/sl/sl_taylor_numeric_screen_l40s.sbatch)
screen_plot=$(sbatch --parsable --dependency="afterok:${screen}" code/scripts/slurm/sl/sl_taylor_numeric_screen_plot_cpu.sbatch)
freeze=$(sbatch --parsable code/scripts/slurm/sl/sl_freeze_taylor_numeric_v2_cpu.sbatch)
final=$(sbatch --parsable --dependency="afterok:${freeze}" code/scripts/slurm/sl/sl_taylor_numeric_final_v2_l40s.sbatch)
transfer=$(sbatch --parsable --dependency="afterok:${freeze}" code/scripts/slurm/sl/sl_taylor_numeric_transfer_l40s.sbatch)
transfer_plot=$(sbatch --parsable --dependency="afterok:${transfer}" code/scripts/slurm/sl/sl_taylor_numeric_transfer_plot_cpu.sbatch)
printf 'taylor_numeric_screen=%s\ntaylor_numeric_screen_plot=%s\ntaylor_numeric_v2_freeze=%s\ntaylor_numeric_v2_final=%s\ntaylor_numeric_transfer=%s\ntaylor_numeric_transfer_plot=%s\n' "$screen" "$screen_plot" "$freeze" "$final" "$transfer" "$transfer_plot"
printf '%s\n' 'The screen and data freeze run in parallel. Final v2 and WS-versus-WC transfer begin after the frozen assets are written.'
