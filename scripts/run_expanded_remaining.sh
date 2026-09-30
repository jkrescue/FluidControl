#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
output="${project_root}/data/curated/tandem_cylinders_expanded_v1"
log="${project_root}/artifacts/tandem_cylinders/expanded_remaining_pipeline.log"
minimum_free_bytes=$((20 * 1024 * 1024 * 1024))
cd "$project_root"

mkdir -p "$(dirname "$log")"
exec > >(tee -a "$log") 2>&1

timestamp() { date '+%F %T %Z'; }

case_h5() {
    local name="$1" split
    case "$name" in
        expanded_train_*) split=train ;;
        expanded_validation_*) split=validation ;;
        expanded_test_*) split=test ;;
        *) return 2 ;;
    esac
    printf '%s/%s/%s.h5\n' "$output" "$split" "$name"
}

all_h5_present() {
    local name
    for name in "$@"; do
        [[ -s "$(case_h5 "$name")" ]] || return 1
    done
}

all_cfd_complete() {
    local name
    for name in "$@"; do
        grep -qx 'End' "cfd/tandem_cylinders/cases/${name}/log.pimpleFoam" 2>/dev/null || return 1
    done
}

windows_d_free_bytes() {
    powershell.exe -NoProfile -Command '(Get-PSDrive -Name D).Free' 2>/dev/null \
        | tr -d '\r[:space:]'
}

check_host_space() {
    local free
    free="$(windows_d_free_bytes || true)"
    if [[ "$free" =~ ^[0-9]+$ ]]; then
        echo "[$(timestamp)] Windows D free bytes: $free"
        if (( free < minimum_free_bytes )); then
            echo "Windows D has less than 20 GiB free; stopping safely before the next chunk" >&2
            exit 75
        fi
    else
        echo "[$(timestamp)] WARNING: could not read Windows D free space" >&2
    fi
}

wait_for_current_chunk() {
    local names=(expanded_train_08 expanded_train_09 expanded_train_10 expanded_train_11)
    if all_h5_present "${names[@]}"; then
        return
    fi
    echo "[$(timestamp)] Waiting for current Curator chunk: ${names[*]}"
    while ! all_h5_present "${names[@]}"; do
        if ! tmux has-session -t expanded_process_08_11 2>/dev/null; then
            echo "Current Curator session ended before all HDF5 files were written" >&2
            exit 1
        fi
        sleep 15
    done
    echo "[$(timestamp)] Current Curator chunk completed"
}

run_chunk() {
    local chunk="$1" names=()
    read -r -a names <<<"$chunk"
    if all_h5_present "${names[@]}"; then
        echo "[$(timestamp)] Skipping completed chunk: ${names[*]}"
        return
    fi

    check_host_space
    if all_cfd_complete "${names[@]}"; then
        echo "[$(timestamp)] Reusing completed CFD chunk: ${names[*]}"
    else
        echo "[$(timestamp)] Starting CFD chunk: ${names[*]}"
        bash cfd/tandem_cylinders/run_expanded_cfd_chunk.sh "${names[@]}"
    fi

    echo "[$(timestamp)] Starting Curator chunk: ${names[*]}"
    bash scripts/process_expanded_chunk.sh "${names[@]}"
    check_host_space
    echo "[$(timestamp)] Completed chunk: ${names[*]}"
}

chunks=(
    'expanded_train_12 expanded_train_13 expanded_train_14 expanded_train_15'
    'expanded_train_16 expanded_train_17 expanded_train_18 expanded_train_19'
    'expanded_train_20 expanded_train_21 expanded_train_22 expanded_train_23'
    'expanded_validation_00 expanded_validation_01 expanded_validation_02 expanded_validation_03'
    'expanded_test_00 expanded_test_01 expanded_test_02 expanded_test_03'
)

echo "[$(timestamp)] Expanded remaining pipeline started"
wait_for_current_chunk
for chunk in "${chunks[@]}"; do
    run_chunk "$chunk"
done
echo "[$(timestamp)] EXPANDED_REMAINING_PIPELINE_OK"
