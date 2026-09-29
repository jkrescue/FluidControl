#!/usr/bin/env bash
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."

while true; do
  clear
  printf 'PhysicsNeMo tandem-cylinder pipeline monitor\n'
  printf 'time: %s\n\n' "$(date --iso-8601=seconds)"

  printf '%s\n' '=== Active pipeline processes ==='
  pgrep -af 'curate_tandem_cfd|validate_tandem_curated|train_tandem_fno|evaluate_tandem_fno' || \
    printf '%s\n' '(none)'

  printf '\n%s\n' '=== GPU 0/1 ==='
  nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader \
    | awk -F, '$1 ~ /^[[:space:]]*[01]$/'

  log=''
  stage='waiting'
  for candidate in \
    artifacts/tandem_fno/train.log \
    artifacts/tandem_fno_smoke/train.log \
    artifacts/tandem_cylinders/curated_validation.log \
    artifacts/tandem_cylinders/curator_full.log
  do
    if [[ -f "$candidate" ]]; then
      log="$candidate"
      stage="$candidate"
      break
    fi
  done

  printf '\n=== Current log: %s ===\n' "$stage"
  if [[ -n "$log" ]]; then
    tail -n 35 "$log"
  else
    printf '%s\n' 'No stage log exists yet.'
  fi

  printf '\n%s\n' 'Refresh: 5 seconds. Leave this window with Ctrl-b then 0.'
  sleep 5
done
