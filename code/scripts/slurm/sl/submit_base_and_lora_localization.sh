#!/usr/bin/env bash
# Independent Experiment 2 and Experiment 3 arrays; each caps at two L40S GPUs.
set -Eeuo pipefail
base=$(sbatch --parsable code/scripts/slurm/sl/sl_base_column_masking_l40s.sbatch)
lora=$(sbatch --parsable code/scripts/slurm/sl/sl_lora_group_localization_l40s.sbatch)
plot=$(sbatch --parsable --dependency="afterok:${base}:${lora}" code/scripts/slurm/sl/sl_base_lora_localization_plot_cpu.sbatch)
printf 'base_column_masking=%s\nlora_group_localization=%s\naggregate_plots=%s\n' "$base" "$lora" "$plot"
