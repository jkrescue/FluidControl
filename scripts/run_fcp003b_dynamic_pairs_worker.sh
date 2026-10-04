#!/usr/bin/env bash
# Worker-only immutable launcher for Lead-approved FC-P003B.
set -euo pipefail

mode="${1:---dry-run}"
case "$mode" in --dry-run|--probe|--execute) ;; *) echo "invalid mode" >&2; exit 2;; esac
root="${FCP003B_ROOT:-/home/USER/workspace/fluid_control_fcp003b_dynamic_pairs_20261005}"
source="$root/source_snapshot"
receipt="$root/source_receipt.json"
image="fluid-control-physicsnemo:2.2.2"
image_id="sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
source_commit="2816e86d7224d8fe8e39f7cd2225d86c7d897974"
approval_sha="b042a8570aec4802f1963392bb90368a2f0428c59d8f7afff76bbfa0417611d8"
manifest_sha="b756c6d777b68fe9dc6c5a81e40733e24d2c248b315723ddb40cc3273902d28c"
probe_sha="e549121553e4e1fa0fffe5c0f3080dc1f03db615342530a7a7fcb0a0e45ac7aa"
baseline_order_sha="fab043a70652475e0b03aa869eac3445eaec1ec6c66a74cf976a0342a3dbca88"
real_sampling_sha="da078a1c43f03bf86ee71a5010632c05a4868b169dc8f8c7c341c78134873242"
source_receipt_sha="48df52c75be6a974ba2d1d9827c0897e4e5335981d7fa300d8663dc821723dab"
source_required_sha="ab519370c43c72d1f3247919ca36cd3c6b1c1da681356f89e5bea7da0e22d79b"
validator_sha="4d2cec37659c98d9aac873893af9b15e16a3af94e46cb71d75e2e8e80aaba9d8"
parent_model_sha="8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240"
parent_state_sha="1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0"
normalization_sha="f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
dev30_manifest_sha="5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2"
train8_manifest_sha="a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35"
train16_manifest_sha="7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b"

sha() { sha256sum "$1" | awk '{print $1}'; }
[[ "$(docker image inspect "$image" --format '{{.Id}}')" == "$image_id" ]] || { echo "image differs" >&2; exit 2; }
[[ -d "$source" && -f "$receipt" ]] || { echo "immutable source snapshot missing" >&2; exit 2; }
[[ "$(sha "$root/evidence/FC-P003B_APPROVAL.md")" == "$approval_sha" ]] || { echo "approval differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/dynamic_pair_manifest.json")" == "$manifest_sha" ]] || { echo "dynamic pair manifest differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/cpu_probe.json")" == "$probe_sha" ]] || { echo "CPU probe differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/fc_p003_dataloader_order_result.json")" == "$baseline_order_sha" ]] || { echo "FC-P003 order receipt differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/real_sampling_contract_v2.json")" == "$real_sampling_sha" ]] || { echo "real sampling receipt differs" >&2; exit 2; }
[[ "$(sha "$receipt")" == "$source_receipt_sha" ]] || { echo "source receipt differs" >&2; exit 2; }
[[ "$(sha "$root/evidence/source_required_hashes.json")" == "$source_required_sha" ]] || { echo "source dependency map differs" >&2; exit 2; }
[[ "$(sha "$root/launch/validate_fcp003b_worker_output.py")" == "$validator_sha" ]] || { echo "output validator differs" >&2; exit 2; }
[[ "$(sha "$root/immutable_parent/FNO.0.2.mdlus")" == "$parent_model_sha" ]] || { echo "parent model differs" >&2; exit 2; }
[[ "$(sha "$root/immutable_parent/checkpoint.0.2.pt")" == "$parent_state_sha" ]] || { echo "parent state differs" >&2; exit 2; }
for path in "$root/data/dev30" "$root/data/train8" "$root/data/train16"; do
  [[ "$(sha "$path/normalization.json")" == "$normalization_sha" ]] || { echo "normalization differs: $path" >&2; exit 2; }
done
[[ "$(sha "$root/data/dev30/manifest.json")" == "$dev30_manifest_sha" ]] || { echo "dev30 manifest differs" >&2; exit 2; }
[[ "$(sha "$root/data/train8/manifest.json")" == "$train8_manifest_sha" ]] || { echo "train8 manifest differs" >&2; exit 2; }
[[ "$(sha "$root/data/train16/manifest.json")" == "$train16_manifest_sha" ]] || { echo "train16 manifest differs" >&2; exit 2; }
python3 "$root/launch/validate_fcp003b_worker_output.py" source \
  --source "$source" --receipt "$receipt" --commit "$source_commit" \
  --required-hashes "$root/evidence/source_required_hashes.json"
python3 "$root/launch/validate_fcp003b_worker_output.py" sampling \
  --real "$root/evidence/real_sampling_contract_v2.json" \
  --baseline "$root/evidence/fc_p003_dataloader_order_result.json"
declare -A expected=(
 [scripts/train_tandem_fno_paired_stats.py]=01c94e104541a9e75e8b85a9049264b18f26c9554bb3f0393d892a385119ac44
 [src/fluid_control/paired_training.py]=5d96142f815df0206451f066a47a2170b1e5b197c5a76ec3e7bc6a7044f32ad5
 [conf/tandem_fno_dynamic_paired_interleaved_h100.yaml]=a8e4033240af24ec6ee0c537059244bda765800c239c292b6b18085a42c9ffc9
 [src/fluid_control/dynamic_pair_stat_datapipe.py]=2414b0a2b2635f87e0842ed12d36ad7a3eb6b1b8759cacdb5b8c8fa9fed5af31
 [scripts/spark_gpu_guard.py]=75d14e5fd926f797f3c3ee8b913d07e9d22c7c365344b6de722edb3b40d39195
)
for relative in "${!expected[@]}"; do
  [[ "$(sha "$source/$relative")" == "${expected[$relative]}" ]] || { echo "source differs: $relative" >&2; exit 2; }
done

if [[ "$mode" == --dry-run ]]; then echo "FC_P003B_WORKER_READY_NO_GPU"; exit 0; fi
[[ "${FCP003B_EXECUTION_TOKEN:-}" == EXECUTE_REVIEWED_FC_P003B ]] || { echo "reviewed token required" >&2; exit 2; }
if [[ "$mode" == --probe ]]; then
  output="$root/probe_v2"; container="fcp003b-dynamic-pairs-probe-v2"
  extra=(training.epochs=1 training.max_train_batches=8 training.max_validation_batches=1 training.expected_regular_batches=8 training.paired_dataset_repetitions=1 training.paired_batches_per_epoch=8 training.max_paired_eval_batches=1)
else
  output="$root/output"; container="fcp003b-dynamic-pairs-full"; extra=()
fi
[[ ! -e "$output" ]] || { echo "refusing existing output: $output" >&2; exit 2; }
mkdir -p "$output"
python3 - "$output/launch_receipt.json" "$mode" "$source_commit" "$approval_sha" "$manifest_sha" "$probe_sha" "$baseline_order_sha" "$real_sampling_sha" "$source_receipt_sha" "$validator_sha" "$image_id" "$parent_model_sha" "$parent_state_sha" "$dev30_manifest_sha" "$train8_manifest_sha" "$train16_manifest_sha" <<'PY'
import json,os,pathlib,sys,tempfile
target=pathlib.Path(sys.argv[1]); mode,commit,approval,manifest,probe,baseline,real_sampling,source_receipt,validator,image,parent_model,parent_state,dev30,train8,train16=sys.argv[2:]
updates=8 if mode=="--probe" else 16
payload={"status":"FC_P003B_WORKER_LAUNCH_STAGED","mode":mode,"git_commit":commit,
 "approval_sha256":approval,"dynamic_pair_manifest_sha256":manifest,"cpu_probe_sha256":probe,
 "fc_p003_order_receipt_sha256":baseline,"real_sampling_receipt_sha256":real_sampling,
 "source_receipt_sha256":source_receipt,"validator_sha256":validator,"official_image_id":image,
 "parent_model_sha256":parent_model,"parent_state_sha256":parent_state,
 "dev30_manifest_sha256":dev30,"train8_manifest_sha256":train8,"train16_manifest_sha256":train16,
 "single_factor":"paired_supervision_content_static16_vs_dynamic8_repeated_twice",
 "paired_dataset_kind":"dynamic8","unique_pair_count":8,"paired_updates_per_epoch":updates,
 "paired_dataset_repetitions":2 if mode=="--execute" else 1,
 "validation_or_frozen_training_accessed":False,"ppo_auto_launch":False}
with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False) as f:
 tmp=pathlib.Path(f.name); json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.link(tmp,target); tmp.unlink()
PY
# ``device=0`` is a single Docker --gpus argument, not comma-separated shell items.
# shellcheck disable=SC2054
command=(docker run --rm --name "$container" --gpus device=0 --cpus 8 --memory 90g
 --shm-size 2g --pids-limit 1024 --cap-drop ALL --security-opt no-new-privileges
 --user "$(id -u):$(id -g)" -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e HOME=/tmp
 -e PYTHONPATH=/workspace/src:/workspace/scripts --tmpfs /tmp:rw,nosuid,nodev,size=4g
 -v "$source/src:/workspace/src:ro" -v "$source/scripts:/workspace/scripts:ro"
 -v "$source/conf:/workspace/conf:ro" -v "$root/data/dev30:/workspace/base:ro"
 -v "$root/data/train8:/workspace/train8:ro" -v "$root/data/train16:/workspace/train16:ro"
 -v "$root/evidence/dynamic_pair_manifest.json:/workspace/dynamic_pair_manifest.json:ro"
 -v "$root/immutable_parent:/workspace/parent:ro" -v "$output:/workspace/output:rw"
 -w /workspace "$image" python scripts/train_tandem_fno_paired_stats.py
 --config-name tandem_fno_dynamic_paired_interleaved_h100
 hydra.run.dir=/tmp/hydra hydra.output_subdir=null output_dir=/workspace/output
 "${extra[@]}")
python3 "$source/scripts/spark_gpu_guard.py" --min-free-gib 20 --allocator-fraction 0.45 --margin-gib 2 --poll-seconds 2 -- "${command[@]}" 2>&1 | tee "$output/train.log"
python3 "$root/launch/validate_fcp003b_worker_output.py" output \
  --root "$output" --mode="$mode" --receipt "$output/completion_receipt.json"
