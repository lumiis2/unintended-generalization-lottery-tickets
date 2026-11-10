#!/usr/bin/env bash
# Submit base preflight, then corrected base screen, alongside independent early-MLP localization.
set -Eeuo pipefail
preflight=$(sbatch --parsable code/scripts/slurm/sl/sl_base_column_preflight_l40s.sbatch)
base=$(sbatch --parsable --dependency="afterok:${preflight}" code/scripts/slurm/sl/sl_base_column_masking_v2_l40s.sbatch)
early_mlp=$(sbatch --parsable code/scripts/slurm/sl/sl_early_mlp_lora_localization_l40s.sbatch)
plots=$(sbatch --parsable --dependency="afterok:${base}:${early_mlp}" code/scripts/slurm/sl/sl_base_v2_early_mlp_plot_cpu.sbatch)
printf 'base_preflight=%s\nbase_column_masking_v2=%s\nearly_mlp_localization=%s\naggregate_plots=%s\n' "$preflight" "$base" "$early_mlp" "$plots"
