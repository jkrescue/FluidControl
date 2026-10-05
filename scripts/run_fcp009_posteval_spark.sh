#!/usr/bin/env bash
# Exact FC-P009 identity profile over the shared unchanged formal suite.
set -euo pipefail
export FCP_POSTEVAL_PROFILE=p009
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/run_fcp008_posteval_spark.sh" "$@"
