#!/usr/bin/env bash
# Stage A: capability-only calibration -> frozen safety gate -> held-out SL
# screen.  The gate exits non-zero when nothing is safe, so the screen then
# remains blocked by afterok and no SL outcomes are inspected.
set -Eeuo pipefail
mkdir -p logs
calibration=$(sbatch --parsable code/scripts/slurm/sl/sl_targeted_base_calibration_l40s.sbatch)
gate=$(sbatch --parsable --dependency="afterok:${calibration}" code/scripts/slurm/sl/sl_targeted_base_calibration_gate.sbatch)
screen=$(sbatch --parsable --dependency="afterok:${gate}" code/scripts/slurm/sl/sl_targeted_base_screen_l40s.sbatch)
aggregate=$(sbatch --parsable --dependency="afterok:${screen}" code/scripts/slurm/sl/sl_targeted_base_aggregate.sbatch)
printf 'targeted_base_calibration=%s\n' "$calibration"
printf 'targeted_base_calibration_gate=%s\n' "$gate"
printf 'targeted_base_screen=%s\n' "$screen"
printf 'targeted_base_aggregate=%s\n' "$aggregate"
