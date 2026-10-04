#!/usr/bin/env bash
set -euo pipefail

[[ "${1:-}" == "--dry-run" || "${1:-}" == "--execute" ]] || { echo "usage: $0 --dry-run|--execute" >&2; exit 64; }
mode="$1"
repo="/workspace/fluid_control"
output="$repo/artifacts/fcp003c_mixed_loss_technical_probe_20261005"
preflight="$repo/artifacts/fcp003c_mixed_loss_probe_preflight_20261005"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="5ac306ae64d7fd1b51f3b1c166377d81f953e59a"
base="$repo/data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
train8="$repo/data/curated/tandem_cylinders_dynamic_train8_v1"
train16="$repo/data/curated/tandem_cylinders_directppo_train16_v1"
pair_manifest="$repo/artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
parent="$repo/artifacts/tandem_fno_control_train16_h100_20261004/best"
resolved_config="$preflight/resolved_config.yaml"
cpu_equivalence="$preflight/cpu_equivalence_receipt.json"
regular_identity="$preflight/cpu_regular_loader_identity.json"

probe_sha="887b3610b17ff77bfae7045d1b649923678dfdbaa3ae4f54aa2ad73dd1b5b4e2"
trainer_sha="3c58a8c4d59c41723dcdff7227f766d3cf00b30aad0e5fabfbc7479c53f19d88"
helper_sha="0b1f88509f0da43de0d2f7ce0e517faf258a2e94d17d382f31368f8a23355663"
child_config_sha="17f15f6cb4b4b92a118ea6b4604c5b3b9ce44cb1f50660b8970bad0d7b51c459"
step_test_sha="75207f176dc15431081ce981eb4feba026d6b8f3bfcebd9e43f415dd5c471c4b"
regular_test_sha="34a619223b0283b151e19c19f8eea85b604fa48f238ae4037e24128a9822c4a9"
guard_sha="75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195"
resolved_sha="ebcc839f302cf3eb93eff2a23f9d8ba38f1063e09591668dad3d65d31bc91f4f"
cpu_equivalence_sha="394c904d9593ada6351d206ed61eb302b6b9f3fca588e2853a676b3db191fe09"
regular_identity_sha="df03eb42bce7e9a996378d7e2bd5e46cf15f46d6f8628ec9ed80dd24b694b80a"
model_sha="8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
state_sha="1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"

check_sha() { local p="$1" e="$2"; [[ -f "$p" && "$(sha256sum "$p"|awk '{print $1}')" == "$e" ]] || { echo "SHA mismatch/missing: $p" >&2; exit 65; }; }
[[ "$(git -C "$repo" rev-parse "$source_commit^{commit}")" == "$source_commit" ]] || { echo "source commit missing" >&2; exit 65; }
check_sha "$repo/scripts/probe_fcp003c_mixed_loss.py" "$probe_sha"
check_sha "$repo/scripts/spark_gpu_guard.py" "$guard_sha"
check_sha "$resolved_config" "$resolved_sha"
check_sha "$cpu_equivalence" "$cpu_equivalence_sha"
check_sha "$regular_identity" "$regular_identity_sha"
check_sha "$base/manifest.json" "5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
check_sha "$train8/manifest.json" "a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
check_sha "$train16/manifest.json" "7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"
for norm in "$base/normalization.json" "$train8/normalization.json" "$train16/normalization.json"; do check_sha "$norm" "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"; done
check_sha "$pair_manifest" "b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
check_sha "$train8/train/dynamic_train8_b00_multisine.h5" "400b3b5f8ea381da59c7806aef19f7cd2b50da303f717ad93cebf505c5949e9f"
check_sha "$base/train/matched_start_acquisition_train_b00_zero.h5" "243caa79ac320b421adc1bf0c2cc830a32482dc758c3c0f9ce71d26171a61a01"
check_sha "$base/train/matched_start_acquisition_train_b04_m075.h5" "9fd391090802f7f6a7189d1cbb2ccedf3e667fa4ee78b2f8f1c2c20c1dea0345"
check_sha "$parent/FNO.0.2.mdlus" "$model_sha"; check_sha "$parent/checkpoint.0.2.pt" "$state_sha"; check_sha "$base/train/matched_start_acquisition_train_b04_m075.h5" "9fd391090802f7f6a7189d1cbb2ccedf3e667fa4ee78b2f8f1c2c20c1dea0345"
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "image differs" >&2; exit 65; }

python3 - "$cpu_equivalence" "$regular_identity" <<'PY'
import json,sys
c=json.load(open(sys.argv[1])); r=json.load(open(sys.argv[2]))
expected={"scripts/train_tandem_fno_paired_stats.py":"3c58a8c4d59c41723dcdff7227f766d3cf00b30aad0e5fabfbc7479c53f19d88","src/fluid_control/paired_step_training.py":"0b1f88509f0da43de0d2f7ce0e517faf258a2e94d17d382f31368f8a23355663","conf/tandem_fno_dynamic_paired_true_state_step_h100.yaml":"17f15f6cb4b4b92a118ea6b4604c5b3b9ce44cb1f50660b8970bad0d7b51c459","tests/test_paired_step_training.py":"75207f176dc15431081ce981eb4feba026d6b8f3bfcebd9e43f415dd5c471c4b","tests/test_fcp003c_regular_objective.py":"34a619223b0283b151e19c19f8eea85b604fa48f238ae4037e24128a9822c4a9"}
if c.get("status")!="FC_P003C_CPU_EQUIVALENCE_PASS" or c.get("source_commit")!="5ac306ae64d7fd1b51f3b1c166377d81f953e59a" or c.get("source_sha256")!=expected: raise SystemExit("CPU equivalence binding differs")
x=c.get("exact_contracts",{}); keys=("chunked_vs_monolithic_loss_gradient_and_one_step","regular_extracted_vs_legacy_loss_and_gradient","nonfinite_refuses_step","single_final_gradient_clip")
if not all(x.get(k) is True for k in keys) or x.get("optimizer_step_count")!=1: raise SystemExit("CPU equivalence contract differs")
if r.get("status")!="FC_P003C_REAL_REGULAR_DATALOADER_IDENTITY_PASS" or r.get("first_batch_metadata")!={"case":"matched_start_acquisition_train_b04_m075","step":180,"rollout_steps":100,"split":"train","dataset_index":0} or r.get("validation_or_frozen_mounted") is not False: raise SystemExit("regular identity differs")
PY

if [[ "$mode" == "--dry-run" ]]; then
  printf '%s\n' '{"status":"FC_P003C_MIXED_LOSS_PROBE_DRY_RUN_READY","gpu_execution_enabled":false,"optimizer_steps":1,"pair_id":"b00:multisine","regular_identity":"matched_start_acquisition_train_b04_m075:180","allocator_fraction":0.25,"minimum_mem_available_gib":20,"validation_or_frozen_mounted":false}'
  exit 0
fi
[[ "${FC_P003C_MIXED_PROBE_APPROVAL_TOKEN:-}" == "EXECUTE_REVIEWED_FC_P003C_MIXED_LOSS_PROBE" ]] || { echo "Lead GPU token missing" >&2; exit 77; }
[[ ! -e "$output" ]] || { echo "exclusive output exists" >&2; exit 65; }
mkdir "$output"; mkdir "$output/source_snapshot" "$output/container_output"
git -C "$repo" archive "$source_commit" src conf scripts/train_tandem_fno.py scripts/train_tandem_fno_paired_stats.py tests/test_paired_step_training.py tests/test_fcp003c_regular_objective.py | tar -x -C "$output/source_snapshot"
mkdir -p "$output/source_snapshot/scripts"; cp "$repo/scripts/probe_fcp003c_mixed_loss.py" "$output/source_snapshot/scripts/"
cp "$repo/scripts/spark_gpu_guard.py" "$output/immutable_guard.py"; cp "$0" "$output/immutable_launcher.sh"
cp "$resolved_config" "$output/resolved_config.yaml"; cp "$cpu_equivalence" "$output/cpu_equivalence_receipt.json"; cp "$regular_identity" "$output/cpu_regular_loader_identity.json"
check_sha "$output/source_snapshot/scripts/train_tandem_fno_paired_stats.py" "$trainer_sha"; check_sha "$output/source_snapshot/src/fluid_control/paired_step_training.py" "$helper_sha"
check_sha "$output/source_snapshot/conf/tandem_fno_dynamic_paired_true_state_step_h100.yaml" "$child_config_sha"; check_sha "$output/source_snapshot/tests/test_paired_step_training.py" "$step_test_sha"; check_sha "$output/source_snapshot/tests/test_fcp003c_regular_objective.py" "$regular_test_sha"; check_sha "$output/source_snapshot/scripts/probe_fcp003c_mixed_loss.py" "$probe_sha"
(cd "$output/source_snapshot" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source_snapshot.sha256"
python3 - "$output" "$image_id" "$source_commit" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
r=Path(sys.argv[1]); h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d={"status":"FC_P003C_MIXED_LOSS_PROBE_LAUNCH_BOUND","image_id":sys.argv[2],"source_commit":sys.argv[3],"optimizer_steps_budget":1,"validation_or_frozen_mounted":False,"immutable_launcher_sha256":h(r/"immutable_launcher.sh"),"immutable_guard_sha256":h(r/"immutable_guard.py"),"source_manifest_sha256":h(r/"source_snapshot.sha256"),"resolved_config_sha256":h(r/"resolved_config.yaml"),"cpu_equivalence_receipt_sha256":h(r/"cpu_equivalence_receipt.json"),"regular_identity_receipt_sha256":h(r/"cpu_regular_loader_identity.json")}
t=r/f"launch_receipt.json.tmp.{os.getpid()}"; t.write_text(json.dumps(d,indent=2)+"\n"); os.link(t,r/"launch_receipt.json"); t.unlink()
PY
set +e
/home/USER/env_isaaclab/bin/python "$output/immutable_guard.py" --min-free-gib 20 --allocator-fraction 0.25 --margin-gib 4 --poll-seconds 2 -- docker run --rm --gpus device=0 --network none --read-only --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --pids-limit 512 --shm-size 1g --memory 90g --cpus 8 --tmpfs /tmp:rw,nosuid,size=2g --tmpfs /tmp/home:rw,nosuid,size=512m --tmpfs /tmp/xdg:rw,nosuid,size=512m -e HOME=/tmp/home -e XDG_CACHE_HOME=/tmp/xdg -e PYTHONPATH=/workspace/project/src:/workspace/project/scripts -v "$output/source_snapshot:/workspace/project:ro" -v "$base/train:/workspace/base/train:ro" -v "$base/manifest.json:/workspace/base/manifest.json:ro" -v "$base/normalization.json:/workspace/base/normalization.json:ro" -v "$train8/train:/workspace/train8/train:ro" -v "$train8/manifest.json:/workspace/train8/manifest.json:ro" -v "$train8/normalization.json:/workspace/train8/normalization.json:ro" -v "$train16/train:/workspace/train16/train:ro" -v "$train16/manifest.json:/workspace/train16/manifest.json:ro" -v "$train16/normalization.json:/workspace/train16/normalization.json:ro" -v "$pair_manifest:/workspace/pair_manifest.json:ro" -v "$parent:/workspace/parent:ro" -v "$output/resolved_config.yaml:/workspace/resolved_config.yaml:ro" -v "$output/source_snapshot.sha256:/workspace/source_snapshot.sha256:ro" -v "$output/launch_receipt.json:/workspace/launch_receipt.json:ro" -v "$output/cpu_equivalence_receipt.json:/workspace/cpu_equivalence_receipt.json:ro" -v "$output/container_output:/workspace/output:rw" "$image" python /workspace/project/scripts/probe_fcp003c_mixed_loss.py --config /workspace/resolved_config.yaml --base-root /workspace/base --train8-root /workspace/train8 --train16-root /workspace/train16 --pair-manifest /workspace/pair_manifest.json --parent /workspace/parent --output /workspace/output/result_bundle --source-manifest /workspace/source_snapshot.sha256 --launch-receipt /workspace/launch_receipt.json --cpu-equivalence-receipt /workspace/cpu_equivalence_receipt.json --expected-cpu-equivalence-sha256 "$cpu_equivalence_sha" --image-id "$image_id" --gpu-memory-fraction 0.25 --min-mem-available-gib 20 >"$output/outer.log" 2>&1
rc=$?; set -e; printf '%s\n' "$rc" > "$output/probe_exit_code.txt"; [[ "$rc" -eq 0 ]] || { echo "probe failed; evidence preserved" >&2; exit "$rc"; }
check_sha "$parent/FNO.0.2.mdlus" "$model_sha"; check_sha "$parent/checkpoint.0.2.pt" "$state_sha"
python3 - "$output" "$model_sha" "$state_sha" <<'PY'
import hashlib,json,math,os,sys
from pathlib import Path
r=Path(sys.argv[1]); p=r/"container_output/result_bundle/result.json"; d=json.loads(p.read_text()); h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
if d.get("status")!="FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_PASS" or d.get("optimizer_steps")!=1 or d.get("candidate_saved") is not False or d.get("validation_or_frozen_accessed") is not False or d.get("regular_hdf_sha256")!="9fd391090802f7f6a7189d1cbb2ccedf3e667fa4ee78b2f8f1c2c20c1dea0345": raise SystemExit("result contract differs")
m=d.get("metrics",{}); raw=m.get("paired_step_force_per_channel_mse",[]); weighted=m.get("paired_step_force_per_channel_weighted_contribution",[]); values=[d.get("regular_field_loss"),d.get("regular_force_loss"),m.get("loss"),m.get("base_loss"),m.get("paired_step_force_loss"),m.get("paired_step_force_weighted_loss"),m.get("preclip_gradient_norm"),*raw,*weighted,d.get("cuda_peak_allocated_gib"),d.get("cuda_peak_reserved_gib"),d.get("minimum_mem_available_gib")]
if len(raw)!=4 or len(weighted)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in values): raise SystemExit("finite metric contract differs")
expected={"model":sys.argv[2],"state":sys.argv[3]}
if d.get("parent_file_sha256_before")!=expected or d.get("parent_file_sha256_after")!=expected or d["minimum_mem_available_gib"]<20: raise SystemExit("parent/memory contract differs")
for suffix in ("*.pt","*.mdlus","*.ckpt","*.pth"):
    if list((r/"container_output").rglob(suffix)): raise SystemExit("candidate/checkpoint output forbidden")
v={"status":"FC_P003C_MIXED_LOSS_TECHNICAL_PROBE_COMPLETE","scientific_result":False,"optimizer_steps":1,"candidate_saved":False,"validation_or_frozen_accessed":False,"result_sha256":h(p),"launch_receipt_sha256":h(r/"launch_receipt.json"),"source_manifest_sha256":h(r/"source_snapshot.sha256"),"immutable_launcher_sha256":h(r/"immutable_launcher.sh"),"immutable_guard_sha256":h(r/"immutable_guard.py"),"checkpoint_model_sha256":sys.argv[2],"checkpoint_state_sha256":sys.argv[3]}
t=r/f"completion_receipt.json.tmp.{os.getpid()}"; t.write_text(json.dumps(v,indent=2)+"\n"); os.link(t,r/"completion_receipt.json"); t.unlink()
PY
