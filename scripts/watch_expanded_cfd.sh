#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
log="${project_root}/artifacts/tandem_cylinders/expanded_cfd_batch.log"
while true; do
    completed="$(grep -c '^Completed expanded_' "$log" || true)"
    failed="$(grep -c '^Failed expanded_' "$log" || true)"
    running="$(docker ps -q | wc -l | tr -d ' ')"
    available="$(df -h "$project_root" | awk 'NR==2 {print $4}')"
    printf '%s completed=%s/32 failed=%s containers=%s disk_available=%s\n' \
        "$(date --iso-8601=seconds)" "$completed" "$failed" "$running" "$available"
    if grep -q '^EXPANDED_CFD_BATCH_OK' "$log"; then
        exit 0
    fi
    if (( failed > 0 )); then
        exit 1
    fi
    sleep 30
done
