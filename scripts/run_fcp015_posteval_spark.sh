#!/usr/bin/env bash
# Identity-only FC-P015 profile; no numerical or admission overrides.
set -euo pipefail
root="${FCP008_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
exec env FCP_POSTEVAL_PROFILE=p015 FCP008_REPO_ROOT="$root" \
  bash "$root/scripts/run_fcp008_posteval_spark.sh" "${1:---dry-run}"
