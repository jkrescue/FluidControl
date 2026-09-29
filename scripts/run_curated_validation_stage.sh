#!/usr/bin/env bash
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

/usr/bin/time -v \
  .venv-curator/bin/python scripts/validate_tandem_curated.py \
  |& tee artifacts/tandem_cylinders/curated_validation.log
code=${PIPESTATUS[0]}
printf 'curated_validation_exit=%s\n' "$code" \
  | tee -a artifacts/tandem_cylinders/curated_validation.log
printf '%s\n' "$code" > artifacts/tandem_cylinders/curated_validation.exit
exit "$code"
