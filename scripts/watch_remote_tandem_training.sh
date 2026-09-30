#!/usr/bin/env bash
set -u

: "${SSH_HOST:?Set SSH_HOST to the remote SSH alias or hostname}"
: "${REMOTE_PROJECT_DIR:?Set REMOTE_PROJECT_DIR to the WSL fluid_control project path}"
WSL_DISTRO="${WSL_DISTRO:-Ubuntu-24.04}"

while true; do
  clear
  ssh "$SSH_HOST" wsl.exe -d "$WSL_DISTRO" -- bash -s -- "$REMOTE_PROJECT_DIR" <<'REMOTE'
cd "$1"
printf 'PhysicsNeMo remote pipeline monitor\n'
printf 'time: %s\n\n' "$(date --iso-8601=seconds)"

printf '%s\n' '=== Active processes ==='
pgrep -af 'curate_tandem_cfd|validate_tandem_curated|train_tandem_fno|evaluate_tandem_fno' || printf '%s\n' '(none)'

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
  tail -n 32 "$log"
else
  printf '%s\n' 'No stage log exists yet.'
fi
REMOTE
  printf '\nRefresh: 5 seconds. Close pane with Ctrl-b then x.\n'
  sleep 5
done
