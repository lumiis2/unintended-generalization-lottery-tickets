#!/usr/bin/env bash
set -Eeuo pipefail
transfer=$(sbatch --parsable code/scripts/slurm/sl/sl_control_mask_transfer_l40s.sbatch)
plot=$(sbatch --parsable --dependency="afterok:${transfer}" code/scripts/slurm/sl/sl_control_mask_transfer_plot_cpu.sbatch)
printf 'control_mask_transfer=%s\ncontrol_mask_transfer_plot=%s\n' "$transfer" "$plot"
