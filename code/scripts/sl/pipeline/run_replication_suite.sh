#!/usr/bin/env bash
# Run the pre-registered follow-up suite serially. Each individual replication
# is resumable through its artifact manifests.
set -Eeuo pipefail

CONFIGS=(
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed48.yaml
    code/configs/sl/reproduction/qwen25_7b_cat_replication_seed49.yaml
    code/configs/sl/reproduction/qwen25_7b_owl_replication.yaml
)

for config in "${CONFIGS[@]}"; do
    echo "===== START replication: ${config} ====="
    bash code/scripts/sl/pipeline/run_replication.sh "$config"
    echo "===== COMPLETE replication: ${config} ====="
done
