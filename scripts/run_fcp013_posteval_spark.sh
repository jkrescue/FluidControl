#!/usr/bin/env bash
# Identity-only profile wrapper; shared numerical protocol remains unchanged.
set -euo pipefail
root="${FCP008_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
exec env FCP_POSTEVAL_PROFILE=p013 FCP008_REPO_ROOT="$root" \
  bash "$root/scripts/run_fcp008_posteval_spark.sh" "${1:---dry-run}"
