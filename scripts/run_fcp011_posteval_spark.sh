#!/usr/bin/env bash
# Thin identity profile for FC-P011; shared runner owns the unchanged science suite.
set -euo pipefail
root="${FCP008_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
scope="${FCP011_SCOPE:-}"
case "$scope" in
  head_only|decoder_tail) ;;
  *) echo "FCP011_SCOPE must be head_only or decoder_tail" >&2; exit 2;;
esac
exec env FCP_POSTEVAL_PROFILE="p011_${scope}" FCP008_REPO_ROOT="$root" \
  "$root/scripts/run_fcp008_posteval_spark.sh" "${1:---dry-run}"
