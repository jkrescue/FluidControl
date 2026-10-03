#!/usr/bin/env bash
# Persistent frozen-blind one-step -> H20 -> validation diagnostic pipeline.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
mode="${1:---dry-run}"
run_id="${2:-}"
token="EXECUTE_REVIEWED_DEV30_QUICKSCREEN_PIPELINE"
data="$root/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
one="$root/artifacts/tandem_fno_full40_dev30_quickscreen_onestep_${run_id}"
h20="$root/artifacts/tandem_fno_full40_dev30_quickscreen_h20_${run_id}"
validation_rel="artifacts/tandem_cylinders/full40_dev30_validation_quickscreen_${run_id}"
validation="$root/$validation_rel"
pipeline="$root/artifacts/tandem_cylinders/full40_dev30_quickscreen_pipeline_${run_id}"

[[ "$run_id" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]] || {
  echo "unique run-id matching ^[a-z0-9][a-z0-9_-]{0,63}$ is required" >&2; exit 2;
}
for output in "$one" "$h20" "$validation" "$pipeline"; do
  [[ ! -e "$output" ]] || {
    echo "refusing existing pipeline output: $output; use a new run-id" >&2; exit 2;
  }
done

plan="$(python3 scripts/plan_full40_dev30_fno_retrain.py)"
python3 -c 'import json,sys; assert json.load(sys.stdin)["status"] == "FULL40_DEV30_FNO_RETRAIN_READY"' <<<"$plan" || {
  echo "$plan" >&2; exit 2;
}
manifest_sha="$(sha256sum "$data/manifest.json" | awk '{print $1}')"
normalization_sha="$(sha256sum "$data/normalization.json" | awk '{print $1}')"

if [[ "$mode" == "--dry-run" ]]; then
  printf 'DEV30_QUICKSCREEN_PIPELINE_READY_STAGE_CANDIDATE_ONLY_NO_FROZEN\n'
  printf 'run_id=%s manifest_sha256=%s normalization_sha256=%s\n' \
    "$run_id" "$manifest_sha" "$normalization_sha"
  printf 'stages=onestep10,h20_5,validation_h1_h10_h50_h100\n'
  printf 'formal_gate=false ppo_authorized=false\n'
  exit 0
fi
[[ "$mode" == "--execute" ]] || { echo "mode must be --dry-run or --execute" >&2; exit 2; }
[[ "${FULL40_DEV30_QUICKSCREEN_PIPELINE_TOKEN:-}" == "$token" ]] || {
  echo "explicit reviewed pipeline token is required" >&2; exit 2;
}

lock="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/fluid-control-dev30-quickscreen-pipeline.lock"
exec 9>"$lock"
flock -n 9 || { echo "another dev30 quick-screen pipeline holds the lock" >&2; exit 1; }
mkdir "$pipeline"
stage="preflight"

write_status() {
  local status="$1"
  local message="$2"
  python3 - "$pipeline/pipeline_status.json" "$status" "$stage" "$message" \
    "$run_id" "$manifest_sha" "$normalization_sha" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "message": sys.argv[4],
    "run_id": sys.argv[5],
    "dev30_manifest_sha256": sys.argv[6],
    "train_only_normalization_sha256": sys.argv[7],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
    "frozen_test_mounted_or_accessed": False,
    "formal_gate_authorized": False,
    "ppo_authorized": False,
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

on_exit() {
  result=$?
  if (( result != 0 )); then
    write_status "DEV30_QUICKSCREEN_PIPELINE_FAILED" "stage failed; downstream stages were not started" || true
  fi
  exit "$result"
}
trap on_exit EXIT

python3 - "$pipeline/pipeline_provenance.json" "$run_id" "$manifest_sha" \
  "$normalization_sha" "$(git rev-parse HEAD)" \
  "$(sha256sum scripts/run_full40_dev30_quickscreen_spark.sh | awk '{print $1}')" \
  "$(sha256sum scripts/run_full40_dev30_validation_diagnostic_spark.sh | awk '{print $1}')" \
  "$(sha256sum scripts/run_full40_dev30_quickscreen_pipeline_spark.sh | awk '{print $1}')" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "DEV30_QUICKSCREEN_PIPELINE_PREDECLARED",
    "run_id": sys.argv[2],
    "dev30_manifest_sha256": sys.argv[3],
    "train_only_normalization_sha256": sys.argv[4],
    "git_commit": sys.argv[5],
    "quickscreen_runner_sha256": sys.argv[6],
    "validation_runner_sha256": sys.argv[7],
    "pipeline_runner_sha256": sys.argv[8],
    "stages": ["onestep_10epoch", "h20_5epoch", "validation_diagnostic"],
    "frozen_test_mounted_or_accessed": False,
    "formal_gate_authorized": False,
    "ppo_authorized": False,
}
with path.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

write_status "DEV30_QUICKSCREEN_PIPELINE_RUNNING" "preflight complete"
stage="onestep_10epoch"
write_status "DEV30_QUICKSCREEN_PIPELINE_RUNNING" "starting official PhysicsNeMo one-step stage"
FULL40_DEV30_QUICKSCREEN_APPROVAL_TOKEN=EXECUTE_REVIEWED_DEV30_QUICKSCREEN \
  bash scripts/run_full40_dev30_quickscreen_spark.sh onestep --execute "$run_id"
mapfile -t one_models < <(find "$one/best" -maxdepth 1 -type f -name 'FNO.*.mdlus' -print)
[[ "${#one_models[@]}" -eq 1 ]] || { echo "one-step best checkpoint is not unique" >&2; exit 1; }
one_sha="$(sha256sum "${one_models[0]}" | awk '{print $1}')"

stage="h20_5epoch"
write_status "DEV30_QUICKSCREEN_PIPELINE_RUNNING" "one-step complete; starting fixed-parent H20"
FULL40_DEV30_QUICKSCREEN_APPROVAL_TOKEN=EXECUTE_REVIEWED_DEV30_QUICKSCREEN \
FULL40_DEV30_QUICKSCREEN_ONESTEP_RUN_ID="$run_id" \
FULL40_DEV30_QUICKSCREEN_PARENT_SHA256="$one_sha" \
  bash scripts/run_full40_dev30_quickscreen_spark.sh h20 --execute "$run_id"
mapfile -t h20_models < <(find "$h20/best" -maxdepth 1 -type f -name 'FNO.*.mdlus' -print)
[[ "${#h20_models[@]}" -eq 1 ]] || { echo "H20 best checkpoint is not unique" >&2; exit 1; }
h20_sha="$(sha256sum "${h20_models[0]}" | awk '{print $1}')"
python3 - "$pipeline/checkpoint_lineage.json" "$one_sha" "$h20_sha" <<'PY'
import json
import sys
from pathlib import Path

with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
    json.dump(
        {"one_step_parent_sha256": sys.argv[2], "h20_candidate_sha256": sys.argv[3]},
        stream,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
PY

stage="validation_diagnostic"
write_status "DEV30_QUICKSCREEN_PIPELINE_RUNNING" "H20 complete; starting validation10 diagnostic"
CHECKPOINT_DIR="artifacts/tandem_fno_full40_dev30_quickscreen_h20_${run_id}/best" \
DEV30_VALIDATION_OUTPUT="$validation_rel" \
DEV30_VALIDATION_APPROVAL_TOKEN=EXECUTE_REVIEWED_DEV30_VALIDATION_DIAGNOSTIC \
  bash scripts/run_full40_dev30_validation_diagnostic_spark.sh --execute "$run_id"
python3 - "$validation/diagnostic.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "DEV30_VALIDATION_DIAGNOSTIC_COMPLETE"
    or report.get("candidate_kind") != "dev30_quickscreen_h20_stage_candidate"
    or report.get("frozen_test_accessed_or_mounted") is not False
    or report.get("formal_gate_authorized") is not False
    or report.get("ppo_authorized") is not False
):
    raise SystemExit("dev30 validation diagnostic contract differs")
PY
stage="complete"
write_status "DEV30_QUICKSCREEN_PIPELINE_COMPLETE_STAGE_DIAGNOSTIC_ONLY" \
  "all three stages completed; formal Gate and PPO remain unauthorized"
trap - EXIT
