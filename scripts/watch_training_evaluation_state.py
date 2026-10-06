#!/usr/bin/env python3
"""Persist authoritative train16 training and post-evaluation state.

This sampler classifies current authority separately from superseded failures.
It never starts work; the separate reconciler may execute only reviewed,
idempotent actions from its fixed allow-list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import shlex
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

TRAINING_UNIT = "fluid-control-train16-h100-full-v1-20261004.service"
MAIN_AUTHORITY_UNIT = "fluid-control-train16-posteval-main-v3-20261004.service"
WORKER_AUTHORITY_UNIT = (
    "fluid-control-balanced-posteval-worker-resume-v1-20261004.service"
)
POSTEVAL_UNIT = MAIN_AUTHORITY_UNIT  # Backward-compatible test/import alias.
SUPERSEDED_MAIN_UNITS = (
    "fluid-control-train16-posteval-queue-v1-20261004.service",
    "fluid-control-train16-posteval-main-v2-20261004.service",
)
WORKER_HOST = "USER@WORKER_HOST"
MAIN_RUN = Path("artifacts/tandem_fno_control_train16_h100_20261004")
BALANCED_RUN = Path(
    "artifacts/tandem_fno_control_train16_h100_lift_balanced_worker_20261004"
)
IDLE_ALERT_SECONDS = 300
MAIN_RECEIPT = MAIN_RUN / "posteval_complete_v2/receipt.json"
WORKER_RECEIPT = BALANCED_RUN / "posteval_worker_v1/receipt.json"
PAIRED_DATAPIPE_ROOT = Path("artifacts/train20_paired_stat_datapipe_v1")
PAIRED_DATAPIPE_WORKER_UNIT = "fluid-control-train20-paired-datapipe-probe-20261004.service"
PAIRED_LAMBDA0_UNIT = "fluid-control-paired-stats-lambda0-full-v1-20261004.service"
PAIRED_LAMBDA0_ROOT = Path("artifacts/tandem_fno_paired_stats_lambda0_20261004")
PAIRED_LAMBDA10_ROOT = Path("artifacts/tandem_fno_paired_stats_lambda10_20261004")
PAIRED_TRAINING_STATUS = "PAIRED_STATS_CONTROLLED_TRAINING_COMPLETE"
PAIRED_LAMBDA10_TRANSFER_STATUS = (
    "PAIRED_STATS_LAMBDA10_WORKER_TO_SPARK_TRANSFER_VERIFIED"
)
PAIRED_POSTEVAL_APPROVAL = Path("docs/FC-P001_APPROVAL.md")
FC_P003_APPROVAL = Path("docs/FC-P003_APPROVAL.md")
FC_P003_UNIT = "fluid-control-fcp003-interleaved-lambda10-r2-20261005.service"
FC_P003_PREFIX = "fluid-control-fcp003-interleaved-lambda10-"
FC_P003_PROBE_UNIT = "fluid-control-fcp003-interleaved-probe-20261005.service"
FC_P003_PROBE_PREFIX = "fluid-control-fcp003-interleaved-probe-"
FC_P003_POSTEVAL_UNIT = "fluid-control-fcp003-posteval-queue-v3-20261005.service"
FC_P003_POSTEVAL_PREFIX = "fluid-control-fcp003-posteval-queue-"
FC_P003B_APPROVAL = Path("docs/FC-P003B_APPROVAL.md")
FC_P003C_APPROVAL = Path("docs/FC-P003C_APPROVAL.md")
FC_P003C_FULL_APPROVAL = Path(
    "docs/FC_P003C_FULL_TRAINING_APPROVAL_20261005.json"
)
FC_P003C_UNIT = "fluid-control-fcp003c-true-state-step-20261005.service"
FC_P003C_CONTAINER = "fcp003c-true-state-step-full-20261005"
FC_P003C_POSTEVAL_UNIT = (
    "fluid-control-fcp003c-posteval-wait-fa08ce0-20261005.service"
)
FC_P003C_ROOT = Path("artifacts/tandem_fno_true_state_paired_step_lambda10_20261005")
FC_P003C_TRAINING_RECEIPT = FC_P003C_ROOT / "completion_receipt.json"
FC_P003C_POSTEVAL_RECEIPT = FC_P003C_ROOT / "posteval_fc_p003c/receipt.json"
FC_P003C_DEVELOPMENT_GATE = FC_P003C_ROOT / "posteval_fc_p003c/development_gate.json"
FC_P003C_TRAINING_STATUS = "FC_P003C_TRAINING_COMPLETE"
FC_P003C_POSTEVAL_STATUS = "FC_P003C_POSTEVAL_COMPLETE"
D015_ROOT = Path("artifacts/fcp003c_train_vs_validation_true_state_h1_20261005")
D015_RESULT = D015_ROOT / "result.json"
D015_EXECUTION_RECEIPT = D015_ROOT / "execution_receipt.json"
D015_RESULT_STATUS = "FCP003C_TRAIN_VS_VALIDATION_TRUE_STATE_H1_DIAGNOSTIC_COMPLETE"
D015_TRANSFER_STATUS = "FCP003C_D015_SPARK_TRANSFER_VERIFIED"
D015_RESULT_SHA256 = (
    "b311715724287c34aa496405f0381fb034089123d5dcf3bec51f80633f293b54"
)
D015_MODEL_SHA256 = (
    "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"
)
D015_CALIBRATION_CONTRACT = Path(
    "docs/D015_CONDITIONAL_TRAIN_FIT_CALIBRATION_DRAFT_20261005.md"
)
CALIBRATION_UNIT = "fluid-control-fcp003c-train-fit-calibration-20261005.service"
CALIBRATION_CONTAINER = "fcp003c-train-fit-calibration-20261005"
CALIBRATION_ROOT = Path("artifacts/fcp003c_train_fit_calibration_20261005")
CALIBRATION_LOG = CALIBRATION_ROOT / "run.log"
CALIBRATION_APPROVAL = CALIBRATION_ROOT / "evidence/execution_approval.json"
CALIBRATION_CPU_PREFLIGHT = CALIBRATION_ROOT / "evidence/cpu_preflight.json"
CALIBRATION_COMPLETION_RECEIPT = CALIBRATION_ROOT / "completion_receipt.json"
CALIBRATION_COMPLETION_STATUS = (
    "FCP003C_TRAIN_FIT_CALIBRATION_EXECUTION_COMPLETE_NOT_ADMISSION"
)
CALIBRATION_APPROVAL_SHA256 = (
    "6cd1ffca319ea734ceec3726c6cedb3919e6aff0d45e636ff3b7e6cc4a45d515"
)
CALIBRATION_IMPLEMENTATION_COMMIT = "ff34a1cf402a12832ad8bedc9d4aa9c4971e6758"
CALIBRATION_IMPLEMENTATION_SHA256 = (
    "576790749f731f3cfff06666efe97772ed61c5a11573ea47559aaf00f57675f8"
)
CALIBRATION_LAUNCHER_SHA256 = (
    "fc81504d55b60d7ff6a73ec1e1d7d22a5bb25119212863af34d180104e843bf9"
)
CALIBRATION_PREFLIGHT_SHA256 = (
    "e45b68810382d56b3126193d8ff159f4cca613649ab40a47a8b1e3a1708f753e"
)
CALIBRATION_CPU_PREFLIGHT_SHA256 = (
    "43deb62e24a18d6f99d7379dc0e2483a8fd2595ea4b84b54fa4da879014cd2e8"
)
CALIBRATION_TOTAL_STEPS = 128
CALIBRATION_PAIRED_STEPS = 64
ABSOLUTE_CALIBRATION_UNIT = (
    "fluid-control-fcp003c-train-fit-absolute-calibration-20261005.service"
)
ABSOLUTE_CALIBRATION_CONTAINER = "fcp003c-train-fit-absolute-calibration-20261005"
ABSOLUTE_CALIBRATION_ROOT = Path(
    "artifacts/fcp003c_train_fit_absolute_calibration_20261005"
)
ABSOLUTE_CALIBRATION_LOG = ABSOLUTE_CALIBRATION_ROOT / "run.log"
ABSOLUTE_CALIBRATION_APPROVAL = Path(
    "docs/FCP003C_TRAIN_FIT_ABSOLUTE_CALIBRATION_APPROVAL_20261005.json"
)
ABSOLUTE_CALIBRATION_COMPLETION_RECEIPT = (
    ABSOLUTE_CALIBRATION_ROOT / "completion_receipt.json"
)
ABSOLUTE_CALIBRATION_COMPLETION_STATUS = (
    "FCP003C_TRAIN_FIT_ABSOLUTE_CALIBRATION_EXECUTION_COMPLETE_NOT_ADMISSION"
)
ABSOLUTE_CALIBRATION_APPROVAL_SHA256 = (
    "191e0276e4f51a53d6387c239a295f7648989daaf428d83dba6af3ded26c1201"
)
ABSOLUTE_CALIBRATION_IMPLEMENTATION_COMMIT = (
    "5cb65bb9a02d308b1de5243f0ac14957aa8bb567"
)
ABSOLUTE_CALIBRATION_IMPLEMENTATION_SHA256 = (
    "069ce14157d2bee9b7b3adc9f47125c47247fe0bcd280ef51cf3167da4b3b421"
)
ABSOLUTE_CALIBRATION_LAUNCHER_SHA256 = (
    "a2bd69d367eef8bfc97ff5feba54531aa2f8b9e35d137c70fd85a0a7bbe40523"
)
FIXED_FEATURE_READOUT_UNIT = (
    "fluid-control-fcp003c-fixed-feature-force-readout-20261005.service"
)
FIXED_FEATURE_READOUT_FAILURE = Path(
    "artifacts/fcp003c_fixed_feature_force_readout_20261005/operational_failure.json"
)
FIXED_FEATURE_READOUT_FAILURE_SHA256 = (
    "c2a75643b05d292f7b99c0acb58b73a97c2150ffa4d504547eeb793ee9b76ad9"
)
FC_P003B_UNIT = "fluid-control-fcp003b-dynamic-pairs-20261005.service"
FC_P003B_PROBE_UNIT = "fluid-control-fcp003b-dynamic-pairs-probe-v3-20261005.service"
FC_P003B_POSTEVAL_UNIT = (
    "fluid-control-fcp003b-posteval-resume-2331301-r2-20261005.service"
)
TRUE_STATE_FORCE_PROBE_V2_UNIT = (
    "fluid-control-true-state-paired-force-backward-probe-v2-20261005.service"
)
TRUE_STATE_FORCE_PROBE_V2_ROOT = Path(
    "artifacts/true_state_paired_force_backward_probe_v2_20261005"
)
TRUE_STATE_FORCE_PROBE_V2_STATUS = (
    "TRUE_STATE_PAIRED_FORCE_BACKWARD_TECHNICAL_PROBE_COMPLETE"
)
FC_P003B_CONTAINER = "fcp003b-dynamic-pairs-full"
FC_P003B_PROBE_CONTAINER = "fcp003b-dynamic-pairs-probe-v3"
FC_P003B_ROOT = Path(
    "artifacts/tandem_fno_dynamic_paired_interleaved_lambda10_20261005"
)
FC_P003B_POSTEVAL_RECEIPT = FC_P003B_ROOT / "posteval_fc_p003b/receipt.json"
FC_P003B_DEVELOPMENT_GATE = FC_P003B_ROOT / "posteval_fc_p003b/development_gate.json"
FC_P003B_TRANSFER_RECEIPT = FC_P003B_ROOT / "worker_transfer_complete.json"
FC_P003B_POSTEVAL_STATUS = "FC_P003B_POSTEVAL_COMPLETE"
FC_P003B_TRANSFER_STATUS = "FC_P003B_WORKER_TO_SPARK_TRANSFER_VERIFIED"
FC_P003_ROOT = Path("artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005")
FC_P003_PROBE_ROOT = Path(
    "artifacts/tandem_fno_paired_stats_interleaved_lambda10_probe_20261005"
)
FC_P003_LAUNCH_RECEIPT = FC_P003_ROOT / "launch_receipt.json"
FC_P003_COMPLETION_RECEIPT = FC_P003_ROOT / "completion_receipt.json"
FC_P003_DEVELOPMENT_GATE = FC_P003_ROOT / "posteval_fc_p003/development_gate.json"
FC_P003_LAUNCH_STATUS = "FC_P003_INTERLEAVED_LAUNCH_STAGED"
FC_P003_COMPLETION_STATUS = "FC_P003_INTERLEAVED_TRAINING_COMPLETE"
FC_P003_PROBE_STATUS = "FC_P003_INTERLEAVED_PROBE_PASS"
PAIRED_POSTEVAL_STATUS = "PAIRED_STATS_FC_P001_POSTEVAL_COMPLETE"
PAIRED_LAMBDA0_POSTEVAL_UNIT = (
    "fluid-control-paired-lambda0-posteval-fcp001-v2-20261004.service"
)
PAIRED_LAMBDA0_POSTEVAL_PREFIX = "fluid-control-paired-lambda0-posteval-fcp001-"
PAIRED_LAMBDA0_POSTEVAL_RECEIPT = (
    PAIRED_LAMBDA0_ROOT / "posteval_fc_p001/receipt.json"
)
PAIRED_LAMBDA10_POSTEVAL_UNIT = (
    "fluid-control-paired-stats-lambda10-posteval-worker-20261004.service"
)
PAIRED_LAMBDA10_POSTEVAL_RECEIPT = (
    PAIRED_LAMBDA10_ROOT / "posteval_fc_p001/receipt.json"
)
PAIRED_LAMBDA0_DEVELOPMENT_GATE = (
    PAIRED_LAMBDA0_ROOT / "posteval_fc_p001/development_gate.json"
)
PAIRED_LAMBDA10_DEVELOPMENT_GATE = (
    PAIRED_LAMBDA10_ROOT / "posteval_fc_p001/development_gate.json"
)
PAIRED_DEVELOPMENT_FAIL_STATUS = "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"
PAIRED_PROTOCOL_SHA256 = {
    "evaluator": "eb93ec1e95ee5ac491bcf929e9e00bc9da83d2c97240c0d9c87d517faffe9d5f",
    "force_window": "4ef0a878ac6f3ab3a8e0a957b7b16882731b2aff8093d0efa46cb4e29a954df8",
    "dynamic6_audit": "4e78d8473d1d0f93b25031a3bf9dcc0582f43604f7b1f67c65e6754f032af100",
    "endpoint_audit": "eedc114549eacd787951ef8d8531db6a1ef40c088ce8bff6fcf10d351fa09269",
    "development_gate": "ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412",
    "step_validator": "c2588d1ad4e7fbea0840fe4d97ccf780865cf912cf25b0ef86d2f4e2689af3c0",
}
REVIEWED_MAIN_RESUME_ACTION = "main-posteval-resume-78d827f"
PRODUCTION_AUTO_RECOVERY_ENABLED = True
P013_UNIT = "fluid-control-fcp013-posteval-r2-20261005.service"
P013_INVOCATION = "7235b2f06282435a89b84964e384c60f"
P013_APPROVAL = Path("docs/FC_P013_FORMAL_EVALUATION_APPROVAL_20261005.json")
P013_APPROVAL_SHA256 = "7216f471ebfd205dfcdff51a393a97f48726a25a20135da9bd9875cc6511d120"
P013_LAUNCHER = "artifacts/p013_posteval_chain_f95048c3a786_immutable/scripts/run_fcp008_posteval_spark.sh"
P015_UNIT = "fluid-control-fcp015-window-accumulation-20261005.service"
P015_INVOCATION = "7842742926284d0c94b0383163d5dc0b"
P015_APPROVAL = Path("docs/FC_P015_EXECUTION_APPROVAL_20261005.json")
P015_APPROVAL_SHA256 = "5f42527e4b032b0c3aff4180aa34a6570ec1fcb3945ec7302934d8386c4702d6"
P015_LAUNCHER = "artifacts/fcp015_window_accumulation_source_20261005_immutable/scripts/run_fcp015_window_accumulation_spark.sh"
P018_UNIT = "fluid-control-fcp018-reduced-rate-20261005.service"
P018_INVOCATION = "1ca4654aab074278bb2efdfff8dbc1eb"
P018_APPROVAL = Path("docs/FC_P018_EXECUTION_APPROVAL_20261005.json")
P018_APPROVAL_SHA256 = "eea5bd5c6a3fe585ae1104600421e50b335f61299c4014c5e3136722af3d4d39"
P018_OBSERVATION = Path("docs/FC_P018_RUNNING_EXECUTION_20261005.json")
P018_OBSERVATION_SHA256 = "c040eae25fa31a98164e08e10dc4f007eb6a3ef38329ad0dcfaddd734c7eb53c"
P018_PROTOCOL_SHA256 = "310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d"
P018_LAUNCHER = "artifacts/fcp018_reduced_rate_source_20261005_immutable/scripts/run_fcp018_reduced_rate_spark.sh"
P018_LAUNCHER_SHA256 = "102b8c6efe54b2e868e4973315260b6ed73b8e24e770af4fa2f04876af36c731"
P018_ROOT = Path("artifacts/fcp018_reduced_rate_training_20261005")
P018_FORMAL_UNIT = "fluid-control-fcp018-posteval-20261005.service"
P018_FORMAL_INVOCATION = "ef589f7dbeff4fa0ab064309409971ad"
P018_FORMAL_APPROVAL = Path("docs/FC_P018_FORMAL_EVALUATION_APPROVAL_20261005.json")
P018_FORMAL_APPROVAL_SHA = "7b137d4600aaa38458dccb8a1db4f13bd6b9e0140e23dcb9f409702fc2ed275b"
P018_SUPERVISOR = Path("artifacts/fcp018_formal_supervisor_5f956e4_immutable.py")
P018_SUPERVISOR_SHA = "4bb55e02a1e14bef92e69ac881603608233b3b1c996c277df77a8287e7124a0c"
P018_CHAIN = Path("artifacts/p018_posteval_chain_d46622b7cb51_immutable")
P018_CHAIN_SHA = "6a97e0b4792f2e15156694b91e148267f577ac6e4eaf0f134caad1dae50efe3f"
P018_RUNNER_SHA = "918ab117b780d6c24c177e3036e58cf7260ab0647bf832466540f7305df5c71e"
P018_FORMAL_MONITOR = P018_ROOT / "formal_supervision_r1"


def p018_formal_argv(repo):
    return ["/usr/bin/python3", str(repo/P018_SUPERVISOR),
        "--runner-sha256", P018_RUNNER_SHA, "--chain-receipt-sha256", P018_CHAIN_SHA,
        "--approval-sha256", P018_FORMAL_APPROVAL_SHA,
        "--training-approval-sha256", P018_APPROVAL_SHA256,
        "--observation-sha256", P018_OBSERVATION_SHA256,
        "--approval", str(repo/P018_FORMAL_APPROVAL),
        "--monitor-dir", str(repo/P018_FORMAL_MONITOR), "--execute"]


def p018_formal_authority(repo, state, now=None):
    """Observe resources at read time; explicit ``now`` is for deterministic replay.

    Callers doing live sampling must omit ``now``: legacy/unit queries can take
    seconds, and a poll-start timestamp predates newly appended memory rows.
    """
    issues=[]
    for path, expected in {
        P018_SUPERVISOR:P018_SUPERVISOR_SHA, P018_FORMAL_APPROVAL:P018_FORMAL_APPROVAL_SHA,
        P018_CHAIN/'receipt.json':P018_CHAIN_SHA,
        P018_CHAIN/'scripts/run_fcp008_posteval_spark.sh':P018_RUNNER_SHA,
        P018_APPROVAL:P018_APPROVAL_SHA256, P018_OBSERVATION:P018_OBSERVATION_SHA256,
    }.items():
        try:
            if file_sha256(repo/path)!=expected: issues.append('Formal identity SHA differs: '+str(path))
        except OSError: issues.append('Formal identity missing: '+str(path))
    if state.get('load_state')!='loaded' or state.get('invocation_id')!=P018_FORMAL_INVOCATION:
        issues.append('Formal unit/invocation differs')
    match=re.search(r"argv\[\]=(.*?) ;",state.get('exec_start',''))
    try: argv=shlex.split(match.group(1)) if match else []
    except ValueError: argv=[]
    if argv!=p018_formal_argv(repo): issues.append('Formal immutable command differs')
    bound=not issues
    if state.get('result')!='success' or state.get('exec_main_status')!='0':
        issues.append('Formal service reports unsuccessful result/exit status')
    running=(state.get('active_state'),state.get('sub_state')) in [('activating','start'),('active','running')]
    running=bool(running and state.get('main_pid',0)>0 and state.get('main_pid_alive') is True)
    terminal=(state.get('active_state')=='active' and state.get('sub_state')=='exited'
              and state.get('main_pid')==0 and state.get('result')=='success'
              and state.get('exec_main_code')=='1' and state.get('exec_main_status')=='0')
    monitor=repo/P018_FORMAL_MONITOR
    progress={'completed_steps':[], 'runner_log_age_seconds':None, 'memory':None}
    approval=read_json(repo/P018_FORMAL_APPROVAL,{})
    for name in ('validation10','dynamic6','force_window'):
        path=repo/P018_ROOT/'posteval_fc_p018/step_receipts'/f'{name}.json'
        if not path.exists(): continue
        row=read_json(path,{})
        required={'status':'FC_P018_POSTEVAL_STEP_COMPLETE','step':name,
            'candidate_kind':'fcp018_reduced_rate_dual_fno',
            'formal_evaluation_approval_sha256':P018_FORMAL_APPROVAL_SHA,
            'posteval_chain_receipt_sha256':P018_CHAIN_SHA,
            'checkpoint_sha256':approval.get('candidate_model_sha256'),
            'checkpoint_state_sha256':approval.get('candidate_state_sha256'),
            'dual_manifest_sha256':approval.get('dual_manifest_sha256')}
        if any(row.get(k)!=v for k,v in required.items()): issues.append('Formal step identity differs: '+name)
        else: progress['completed_steps'].append(name)
    try:
        with (monitor/'memory.jsonl').open('rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-16384))
            rows=stream.read().decode().splitlines()
        if now is None: now=utc_now()
        memory=json.loads(rows[-1]); age=(now-datetime.fromisoformat(memory['timestamp_utc'])).total_seconds()
        progress['memory']=memory; progress['memory_age_seconds']=age
        if running and (age<0 or age>30): issues.append('Formal resource samples stale')
        if any(type(memory.get(k)) not in (int,float) or not math.isfinite(memory[k]) or memory[k]<20*1024**2 for k in ('mem_available_kib','mem_free_kib')):
            issues.append('Formal resource below 20 GiB or unavailable')
    except (OSError,ValueError,KeyError,IndexError,TypeError):
        if running: issues.append('Formal resource samples unavailable')
    if now is None: now=utc_now()
    try: progress['runner_log_age_seconds']=max(0,now.timestamp()-(monitor/'runner.log').stat().st_mtime)
    except OSError: pass
    classification='NEEDS_AGENT_ANALYSIS'
    action='核查正式评估身份、资源或失败原因；保留证据，不自动重跑。'
    if not running and not terminal:
        issues.append('Formal unit is neither verified running nor successful retained terminal')
    if bound and running:
        classification='RUNNING' if not issues else 'RUNNING_REQUIRES_REVIEW'
        action='继续已批准的原协议正式评估；日志静默不代表结束。完成后独立核对原始完整门槛，不自动启动PPO。'
    elif not issues and terminal:
        classification='TERMINAL_AUDIT_PENDING'
        action='正式评估进程已成功退出；核对守卫收据、完整评估收据及原始数值门槛，交Lead科学判断，不自动认定准入或启动PPO。'
    if terminal:
        guard=read_json(monitor/'receipt.json',{})
        expected={'status':'FC_P018_FORMAL_RESOURCE_GUARD_COMPLETE_NOT_ADMISSION',
            'supervisor_sha256':P018_SUPERVISOR_SHA,'runner_sha256':P018_RUNNER_SHA,
            'approval_sha256':P018_FORMAL_APPROVAL_SHA,'chain_receipt_sha256':P018_CHAIN_SHA,
            'training_approval_sha256':P018_APPROVAL_SHA256,
            'observation_sha256':P018_OBSERVATION_SHA256,
            'command':['bash',str(repo/P018_CHAIN/'scripts/run_fcp008_posteval_spark.sh'),'--execute'],
            'runner_exit_code':0}
        if any(guard.get(k)!=v for k,v in expected.items()):
            issues.append('Formal terminal guard proof absent or differs');classification='NEEDS_AGENT_ANALYSIS'
    return {'authority_unit':P018_FORMAL_UNIT,'invocation_id':P018_FORMAL_INVOCATION,
        'observed_utc':now.isoformat(),
        'state':classification,'running':bound and running,'stage_complete':False,
        'approved_action_id':None,'identity_issues':issues,'progress':progress,
        'next_action':action,'scientific_admission':False,'automatic_recovery_eligible':False}


def classify_authority_task(state: dict, complete: bool, *, allow_resume: bool) -> tuple[str, str | None]:
    if complete:
        return "STAGE_COMPLETE", None
    if state.get("active_state") == "active":
        return "RUNNING", None
    error = (state.get("last_error_line") or "").lower()
    must_stop = (
        "lineage",
        "sha mismatch",
        "hash mismatch",
        "schema",
        "nonfinite",
        "out of memory",
        "oom",
        "20 gib",
    )
    if any(marker in error for marker in must_stop):
        return "NEEDS_AGENT_ANALYSIS", None
    known_transient = (
        "invalid choice: 'control_train16_development'",
        "container startup",
        "gpu_guard",
    )
    if PRODUCTION_AUTO_RECOVERY_ENABLED and allow_resume and (
        state.get("result") == "success"
        or any(marker in error for marker in known_transient)
    ):
        return "RETRY_ELIGIBLE", REVIEWED_MAIN_RESUME_ACTION
    return "NEEDS_AGENT_ANALYSIS", None


def classify_approved_evaluation(state: dict, complete: bool) -> str:
    """Classify FC-P001 without treating approved preflight as a failure."""
    if complete:
        return "STAGE_COMPLETE"
    if state.get("active_state") == "active":
        return "RUNNING"
    if state.get("active_state") == "failed" or state.get("result") == "exit-code":
        return "NEEDS_AGENT_ANALYSIS"
    return "LEAD_APPROVED_PREFLIGHT"


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback


def paired_scientific_verdict(repo: Path, paired_complete: bool) -> dict:
    """Read receipt-bound development gates without turning a science fail into a retry."""
    if not paired_complete:
        return {"status": "PENDING", "lambda0": None, "lambda10": None}
    lambda0 = read_json(repo / PAIRED_LAMBDA0_DEVELOPMENT_GATE, {}).get("status")
    lambda10 = read_json(repo / PAIRED_LAMBDA10_DEVELOPMENT_GATE, {}).get("status")
    status = (
        "FC_P001_SCIENTIFIC_REJECTED"
        if lambda0 == lambda10 == PAIRED_DEVELOPMENT_FAIL_STATUS
        else "FC_P001_COMPLETE_REQUIRES_AGENT_REVIEW"
    )
    return {"status": status, "lambda0": lambda0, "lambda10": lambda10}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_payload_sha256(path: Path, receipt_path: Path) -> str:
    """Monitor optimization only: prior byte SHA plus unchanged FD identity.

    This is not a new candidate/scientific audit. Only completed-receipt model
    payloads use this cache; authority, receipts and live logs remain uncached.
    """
    absolute = path.absolute()
    artifacts = next((p for p in receipt_path.absolute().parents if p.name == "artifacts"), None)
    if artifacts is None:
        return file_sha256(path)
    cache = artifacts / "monitor/historical_payload_sha_cache.json"
    entries = {}
    try:
        if any(p.is_symlink() for p in (cache, *cache.parents)):
            raise ValueError("unsafe cache path")
        info = cache.stat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o022 or info.st_size > 2_000_000):
            raise ValueError("unsafe cache file")
        document = json.loads(cache.read_text())
        if document.get("schema") != 1 or not isinstance(document.get("entries"), dict):
            raise ValueError("cache schema")
        if len(document["entries"]) > 1024:
            raise ValueError("cache size")
        entries = document["entries"]
    except (OSError, ValueError, TypeError, AttributeError):
        entries = {}

    def persist():
        # Cache failure must never substitute for a successful byte verification.
        try:
            if any(p.is_symlink() for p in (cache, *cache.parents)):
                return
            while len(entries) > 1024:
                entries.pop(next(iter(entries)))
            atomic_json(cache, {"schema": 1, "scope": "monitor_only_not_scientific_audit", "entries": entries})
        except OSError:
            pass

    key = str(absolute)
    fd = None
    try:
        # Open every ancestor without following links, not resolve-then-open.
        directory = os.open("/", os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in absolute.parts[1:-1]:
                child = os.open(part, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            os.close(directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("nonregular historical payload")
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        identity = [getattr(before, name) for name in fields]
        entry = entries.get(key, {})
        cached = (isinstance(entry, dict) and entry.get("identity") == identity
                  and isinstance(entry.get("sha256"), str)
                  and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None)
        if cached:
            digest = entry["sha256"]
        else:
            hasher = hashlib.sha256()
            while chunk := os.read(fd, 1024 * 1024):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        after = os.fstat(fd)
        current = absolute.stat(follow_symlinks=False)
        if (identity != [getattr(after, name) for name in fields]
                or identity != [getattr(current, name) for name in fields]):
            raise OSError("historical payload changed during verification")
        entries[key] = {"identity": identity, "sha256": digest,
                        "last_verification": "cached_prior_sha_stat_unchanged" if cached else "new_same_fd_sha256"}
        persist()
        return digest
    except OSError:
        entries.pop(key, None)
        persist()
        raise
    finally:
        if fd is not None:
            os.close(fd)


def verify_receipt(path: Path, expected_status: str) -> tuple[bool, list[str]]:
    receipt = read_json(path, None)
    if not isinstance(receipt, dict):
        return False, ["receipt missing or invalid JSON"]
    issues = []
    if receipt.get("status") != expected_status:
        issues.append("receipt status differs")
    hashes = receipt.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        issues.append("receipt sha256 table missing")
        return False, issues
    root = path.parent.resolve()
    for relative, expected in hashes.items():
        unresolved = root / relative
        target = unresolved.resolve()
        if root not in target.parents or not target.is_file() or any(p.is_symlink() for p in (unresolved, *unresolved.parents)):
            issues.append(f"missing/unsafe payload: {relative}")
        else:
            completed = receipt.get("status") == expected_status and any(
                token in expected_status for token in ("COMPLETE", "VERIFIED", "PASS"))
            try:
                digest = (historical_payload_sha256(target, path)
                          if completed and target.suffix in (".pt", ".mdlus")
                          else file_sha256(target))
                if digest != expected:
                    issues.append(f"payload SHA differs: {relative}")
            except OSError as error:
                issues.append(f"payload unavailable/changed: {relative}: {error}")
    return not issues, issues


def verify_true_state_force_probe(root: Path) -> tuple[bool, list[str]]:
    """Verify the non-scientific v2 technical probe without a generic SHA table."""
    receipt_path = root / "completion_receipt.json"
    result_path = root / "results/result.json"
    launch_path = root / "launch_receipt.json"
    launcher_path = root / "immutable_launcher.sh"
    receipt = read_json(receipt_path, None)
    result = read_json(result_path, None)
    if not isinstance(receipt, dict) or not isinstance(result, dict):
        return False, ["technical probe receipt/result missing or invalid"]
    issues = []
    expected = {
        "status": TRUE_STATE_FORCE_PROBE_V2_STATUS,
        "scientific_result": False,
        "optimizer_steps": 0,
        "validation_or_frozen_accessed": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            issues.append(f"technical probe receipt {key} differs")
    if (
        result.get("status")
        != "TRUE_STATE_PAIRED_FORCE_BACKWARD_TECHNICAL_PROBE_PASS"
        or result.get("training_executed") is not False
        or result.get("optimizer_constructed") is not False
        or result.get("optimizer_steps") != 0
        or result.get("candidate_weights_saved") is not False
        or result.get("validation_or_frozen_accessed") is not False
    ):
        issues.append("technical probe result scope/status differs")
    for path, key in (
        (result_path, "result_sha256"),
        (launch_path, "launch_receipt_sha256"),
        (launcher_path, "immutable_launcher_sha256"),
    ):
        if not path.is_file() or receipt.get(key) != file_sha256(path):
            issues.append(f"technical probe {key} differs")
    before = result.get("input_sha256", {}).get(
        "model_parameters_and_buffers_before"
    )
    after = result.get("input_sha256", {}).get(
        "model_parameters_and_buffers_after"
    )
    if not before or before != after:
        issues.append("technical probe model parameter/buffer SHA differs")
    return not issues, issues


def verify_d015_terminal(repo: Path) -> tuple[bool, list[str]]:
    """Verify the immutable D015 result and its Spark transfer receipt."""
    result_path = repo / D015_RESULT
    receipt_path = repo / D015_EXECUTION_RECEIPT
    result = read_json(result_path, None)
    receipt = read_json(receipt_path, None)
    issues = []
    if not isinstance(result, dict):
        return False, ["D015 result missing or invalid JSON"]
    if not isinstance(receipt, dict):
        return False, ["D015 execution receipt missing or invalid JSON"]
    if not result_path.is_file() or file_sha256(result_path) != D015_RESULT_SHA256:
        issues.append("D015 result SHA differs")
    if result.get("status") != D015_RESULT_STATUS:
        issues.append("D015 result status differs")
    if result.get("model_sha256") != D015_MODEL_SHA256:
        issues.append("D015 result model SHA differs")
    expected_receipt = {
        "status": D015_TRANSFER_STATUS,
        "result_sha256": D015_RESULT_SHA256,
        "model_sha256": D015_MODEL_SHA256,
        "optimizer_steps": 0,
        "candidate_saved": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            issues.append(f"D015 execution receipt {key} differs")
    return not issues, issues


def verify_d015_calibration_authorization(repo: Path) -> tuple[bool, list[str]]:
    """Verify immutable calibration approval and executing code, not a draft doc."""
    approval_path = repo / CALIBRATION_APPROVAL
    approval = read_json(approval_path, None)
    if not isinstance(approval, dict):
        return False, ["D015 calibration execution approval missing or invalid"]
    issues = []
    if file_sha256(approval_path) != CALIBRATION_APPROVAL_SHA256:
        issues.append("D015 calibration approval SHA differs")
    expected = {
        "status": "FCP003C_TRAIN_FIT_CALIBRATION_EXECUTION_APPROVED",
        "gpu_execution_authorized": True,
        "scope": "single_bounded_train_only_calibration",
        "optimizer_steps": CALIBRATION_TOTAL_STEPS,
        "paired_steps": CALIBRATION_PAIRED_STEPS,
        "implementation_commit": CALIBRATION_IMPLEMENTATION_COMMIT,
        "implementation_sha256": CALIBRATION_IMPLEMENTATION_SHA256,
        "launcher_sha256": CALIBRATION_LAUNCHER_SHA256,
        "preflight_script_sha256": CALIBRATION_PREFLIGHT_SHA256,
        "cpu_preflight_sha256": CALIBRATION_CPU_PREFLIGHT_SHA256,
        "minimum_mem_available_gib": 20,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "scientific_admission": False,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            issues.append(f"D015 calibration approval {key} differs")
    files = {
        CALIBRATION_ROOT / "source_snapshot/scripts/run_fcp003c_train_fit_calibration.py": CALIBRATION_IMPLEMENTATION_SHA256,
        Path("scripts/run_fcp003c_train_fit_calibration_spark_2aaec85_immutable.sh"): CALIBRATION_LAUNCHER_SHA256,
        Path("scripts/preflight_fcp003c_train_fit_calibration.py"): CALIBRATION_PREFLIGHT_SHA256,
        CALIBRATION_CPU_PREFLIGHT: CALIBRATION_CPU_PREFLIGHT_SHA256,
    }
    for relative, expected_sha in files.items():
        path = repo / relative
        if not path.is_file() or file_sha256(path) != expected_sha:
            issues.append(f"D015 calibration input SHA differs: {relative}")
    return not issues, issues


def verify_absolute_calibration_authorization(repo: Path) -> tuple[bool, list[str]]:
    """Verify the immutable absolute-force approval and executing code."""
    approval_path = repo / ABSOLUTE_CALIBRATION_APPROVAL
    approval = read_json(approval_path, None)
    if not isinstance(approval, dict):
        return False, ["absolute calibration approval missing or invalid"]
    issues = []
    if file_sha256(approval_path) != ABSOLUTE_CALIBRATION_APPROVAL_SHA256:
        issues.append("absolute calibration approval SHA differs")
    expected = {
        "status": "FCP003C_TRAIN_FIT_ABSOLUTE_CALIBRATION_EXECUTION_APPROVED",
        "gpu_execution_authorized": True,
        "scope": "single_bounded_train_only_absolute_force_calibration",
        "optimizer_steps": CALIBRATION_TOTAL_STEPS,
        "paired_steps": CALIBRATION_PAIRED_STEPS,
        "paired_force_objective": "absolute",
        "implementation_commit": ABSOLUTE_CALIBRATION_IMPLEMENTATION_COMMIT,
        "implementation_sha256": ABSOLUTE_CALIBRATION_IMPLEMENTATION_SHA256,
        "launcher_sha256": ABSOLUTE_CALIBRATION_LAUNCHER_SHA256,
        "cpu_preflight_sha256": CALIBRATION_CPU_PREFLIGHT_SHA256,
        "minimum_mem_available_gib": 20,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "scientific_admission": False,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            issues.append(f"absolute calibration approval {key} differs")
    files = {
        Path("scripts/run_fcp003c_train_fit_calibration.py"): ABSOLUTE_CALIBRATION_IMPLEMENTATION_SHA256,
        Path("scripts/run_fcp003c_train_fit_absolute_calibration_spark.sh"): ABSOLUTE_CALIBRATION_LAUNCHER_SHA256,
        Path("scripts/preflight_fcp003c_train_fit_calibration.py"): CALIBRATION_PREFLIGHT_SHA256,
        CALIBRATION_CPU_PREFLIGHT: CALIBRATION_CPU_PREFLIGHT_SHA256,
    }
    for relative, expected_sha in files.items():
        path = repo / relative
        if not path.is_file() or file_sha256(path) != expected_sha:
            issues.append(f"absolute calibration input SHA differs: {relative}")
    return not issues, issues


def verify_fixed_feature_operational_failure(repo: Path) -> tuple[bool, list[str]]:
    """Verify the bounded v1 engineering failure without making a science claim."""
    path = repo / FIXED_FEATURE_READOUT_FAILURE
    record = read_json(path, None)
    if not isinstance(record, dict):
        return False, ["fixed-feature operational failure record missing or invalid"]
    issues = []
    if file_sha256(path) != FIXED_FEATURE_READOUT_FAILURE_SHA256:
        issues.append("fixed-feature operational failure record SHA differs")
    expected = {
        "status": "FCP003C_FIXED_FEATURE_FORCE_READOUT_OPERATIONAL_FAILURE",
        "classification": "ENGINEERING_NUMERICAL_WIRING_CHECK_FAILURE_NOT_SCIENTIFIC_RESULT",
        "authority_unit": FIXED_FEATURE_READOUT_UNIT,
        "exec_main_status": 1,
        "optimizer_steps": 0,
        "candidate_saved": False,
        "result_json_created": False,
        "cache_created": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
        "automatic_retry": False,
        "scientific_gate_changed": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            issues.append(f"fixed-feature operational failure {key} differs")
    return not issues, issues


def calibration_log_progress(
    repo: Path,
    log: Path = CALIBRATION_LOG,
    paired_force_objective: str | None = None,
) -> dict:
    """Read only complete JSON events already flushed by the guarded run."""
    path = repo / log
    progress = []
    gpu_preflight = None
    guard_exit = None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if event.get("event") == "fcp003c_train_fit_progress":
            completed = event.get("completed_steps")
            paired = event.get("paired_steps_completed")
            total = event.get("total_steps")
            if (
                isinstance(completed, int)
                and isinstance(paired, int)
                and total == CALIBRATION_TOTAL_STEPS
                and 0 <= completed <= CALIBRATION_TOTAL_STEPS
                and 0 <= paired <= CALIBRATION_PAIRED_STEPS
                and (
                    paired_force_objective is None
                    or event.get("paired_force_objective")
                    == paired_force_objective
                )
            ):
                progress.append(event)
        elif event.get("event") == "gpu_preflight":
            gpu_preflight = event
        elif event.get("event") in {"gpu_guard_exit", "gpu_guard_complete"}:
            guard_exit = event
    latest = progress[-1] if progress else {}
    observed = guard_exit or gpu_preflight or {}
    return {
        "log": str(log),
        "completed_steps": latest.get("completed_steps", 0),
        "total_steps": CALIBRATION_TOTAL_STEPS,
        "paired_steps_completed": latest.get("paired_steps_completed", 0),
        "paired_steps_total": CALIBRATION_PAIRED_STEPS,
        "latest_kind": latest.get("latest_kind"),
        "latest_loss": latest.get("latest_loss"),
        "progress_event_count": len(progress),
        "minimum_required_mem_available_gib": 20.0,
        "latest_logged_mem_available_gib": observed.get(
            "min_observed_mem_available_gib",
            observed.get("mem_available_gib"),
        ),
        "latest_logged_cuda_free_gib": observed.get(
            "min_observed_cuda_free_gib", observed.get("cuda_free_gib")
        ),
        "guard_exit_code": guard_exit.get("exit_code") if guard_exit else None,
        "paired_force_objective": paired_force_objective,
    }


def verify_fc_p003b_terminal(repo: Path) -> tuple[bool, list[str]]:
    """Verify the transferred FC-P003B result, not merely an inactive unit."""
    root = repo / FC_P003B_ROOT
    posteval_path = repo / FC_P003B_POSTEVAL_RECEIPT
    transfer_path = repo / FC_P003B_TRANSFER_RECEIPT
    gate_path = repo / FC_P003B_DEVELOPMENT_GATE
    posteval_ok, issues = verify_receipt(posteval_path, FC_P003B_POSTEVAL_STATUS)
    transfer = read_json(transfer_path, None)
    posteval = read_json(posteval_path, None)
    gate = read_json(gate_path, None)
    issues = list(issues)
    if not posteval_ok:
        return False, issues
    if not isinstance(transfer, dict):
        return False, issues + ["FC-P003B transfer receipt missing or invalid"]
    if transfer.get("status") != FC_P003B_TRANSFER_STATUS:
        issues.append("FC-P003B transfer status differs")
    if transfer.get("file_count") != 36:
        issues.append("FC-P003B transfer file count differs")
    if transfer.get("frozen_test_accessed") is not False:
        issues.append("FC-P003B transfer says frozen data were accessed")
    if transfer.get("ppo_auto_launched") is not False:
        issues.append("FC-P003B transfer says PPO was launched")
    if transfer.get("posteval_receipt_sha256") != file_sha256(posteval_path):
        issues.append("FC-P003B transferred posteval receipt SHA differs")
    if transfer.get("checkpoint_sha256") != posteval.get("checkpoint_sha256"):
        issues.append("FC-P003B transfer/checkpoint identity differs")
    if not isinstance(gate, dict) or gate.get("status") != PAIRED_DEVELOPMENT_FAIL_STATUS:
        issues.append("FC-P003B development gate is absent or not the reviewed FAIL")
    expected_gate_sha = posteval.get("sha256", {}).get("development_gate.json")
    if expected_gate_sha != file_sha256(gate_path):
        issues.append("FC-P003B development gate SHA differs from receipt")
    if root.resolve() not in posteval_path.resolve().parents:
        issues.append("FC-P003B posteval receipt escapes canonical root")
    return not issues, issues


def select_main_authority(units: dict[str, dict]) -> str:
    candidates = [
        name
        for name in units
        if name.startswith("fluid-control-train16-posteval-main-")
    ]
    active = [name for name in candidates if units[name].get("active_state") == "active"]
    if active:
        return max(active, key=lambda name: units[name].get("started_at") or "")
    versioned = []
    for name in candidates:
        match = re.search(r"-v(\d+)-", name)
        versioned.append((int(match.group(1)) if match else 0, name))
    return max(versioned, default=(0, MAIN_AUTHORITY_UNIT))[1]


def select_versioned_authority(
    units: dict[str, dict], prefix: str, default: str
) -> str:
    candidates = [name for name in units if name.startswith(prefix)]
    active = [name for name in candidates if units[name].get("active_state") == "active"]
    if active:
        return max(active, key=lambda name: units[name].get("started_at") or "")
    versioned = []
    for name in candidates:
        match = re.search(r"-[vr](\d+)-", name)
        versioned.append((int(match.group(1)) if match else 1, name))
    return max(versioned, default=(0, default))[1]


def _command(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, text=True, capture_output=True, check=False, timeout=15
    )


def most_specific_error(lines: list[str]) -> str | None:
    """Prefer the runtime exception over systemd's later generic failure line."""
    specific = re.compile(
        r"ValueError:|RuntimeError:|Traceback \(most recent call last\)|error: argument",
        re.I,
    )
    generic = re.compile(r"Error:|FAILED|failed", re.I)
    for pattern in (specific, generic):
        for line in reversed(lines):
            if pattern.search(line):
                return line[-500:]
    return None


def unit_state(unit: str) -> dict:
    properties = (
        "ActiveState,SubState,Result,ExecMainStatus,ExecMainStartTimestamp,"
        "ExecMainExitTimestamp,MainPID,InvocationID,ExecStart,LoadState,ExecMainCode"
    )
    result = _command(
        ["systemctl", "--user", "show", unit, f"--property={properties}"]
    )
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    journal = _command(
        ["journalctl", "--user", "-u", unit, "--no-pager", "-n", "40"]
    )
    error = most_specific_error(journal.stdout.splitlines())
    main_pid = int(values.get("MainPID") or 0)
    required_training_args = (
        ("--config-name", "tandem_fno_dynamic_paired_true_state_step_h100")
        if unit == FC_P003C_UNIT
        else ("--output", "/workspace/output/calibration")
        if unit in (CALIBRATION_UNIT, ABSOLUTE_CALIBRATION_UNIT)
        else ()
    )
    training_needle = (
        "run_fcp003c_train_fit_calibration.py"
        if unit in (CALIBRATION_UNIT, ABSOLUTE_CALIBRATION_UNIT)
        else "train_tandem_fno_paired_stats.py"
    )
    training_process_pids = descendant_processes_matching(
        main_pid,
        training_needle,
        required_args=required_training_args,
    )
    if unit == FC_P003C_UNIT:
        training_process_pids = local_container_python_pids(
            FC_P003C_CONTAINER,
            "train_tandem_fno_paired_stats.py",
            required_args=required_training_args,
        )
    elif unit in (CALIBRATION_UNIT, ABSOLUTE_CALIBRATION_UNIT):
        container = (
            ABSOLUTE_CALIBRATION_CONTAINER
            if unit == ABSOLUTE_CALIBRATION_UNIT
            else CALIBRATION_CONTAINER
        )
        training_process_pids = local_container_python_pids(
            container,
            training_needle,
            required_args=required_training_args,
        )
    posteval_process_pids = sorted(
        set(
            descendant_processes_matching(main_pid, "evaluate_tandem_fno.py")
            + descendant_processes_matching(main_pid, "diagnose_fno_force_window.py")
        )
    )
    return {
        "unit": unit,
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "result": values.get("Result", "unknown"),
        "exec_main_status": values.get("ExecMainStatus", "unknown"),
        "exec_main_code": values.get("ExecMainCode", "unknown"),
        "load_state": values.get("LoadState", "unknown"),
        "main_pid": main_pid,
        "main_pid_alive": main_pid > 0 and Path(f"/proc/{main_pid}").is_dir(),
        "invocation_id": values.get("InvocationID", ""),
        "exec_start": values.get("ExecStart", ""),
        "training_process_pids": training_process_pids,
        "posteval_process_pids": posteval_process_pids,
        "started_at": values.get("ExecMainStartTimestamp") or None,
        "exited_at": values.get("ExecMainExitTimestamp") or None,
        "last_error_line": error,
    }


def is_python_payload(
    command: str, needle: str, required_args: tuple[str, ...] = ()
) -> bool:
    """Reject Docker/guard/shell wrappers that merely quote a Python payload."""
    tokens = command.split()
    if not tokens or not Path(tokens[0]).name.startswith("python"):
        return False
    if "spark_gpu_guard.py" in command or needle not in command:
        return False
    return all(value in tokens for value in required_args)


def parse_container_python_pids(
    output: str, needle: str, required_args: tuple[str, ...] = ()
) -> list[int]:
    pids = []
    for line in output.splitlines()[1:]:
        columns = line.split(maxsplit=1)
        if len(columns) != 2 or not columns[0].isdigit():
            continue
        if is_python_payload(columns[1], needle, required_args):
            pids.append(int(columns[0]))
    return sorted(set(pids))


def local_container_python_pids(
    container: str, needle: str, *, required_args: tuple[str, ...] = ()
) -> list[int]:
    result = _command(["docker", "top", container, "-eo", "pid,args"])
    if result.returncode != 0:
        return []
    return parse_container_python_pids(result.stdout, needle, required_args)


def descendant_processes_matching(
    root_pid: int, needle: str, *, required_args: tuple[str, ...] = ()
) -> list[int]:
    """Return descendants whose command line proves the actual training payload."""
    if root_pid <= 0:
        return []
    children: dict[int, list[int]] = {}
    commands: dict[int, str] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            close = stat.rfind(")")
            parent = int(stat[close + 2 :].split()[1])
            pid = int(entry.name)
            children.setdefault(parent, []).append(pid)
            commands[pid] = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (OSError, ValueError, IndexError):
            continue
    descendants = []
    queue = [root_pid]
    while queue:
        parent = queue.pop()
        for pid in children.get(parent, []):
            queue.append(pid)
            if is_python_payload(commands.get(pid, ""), needle, required_args):
                descendants.append(pid)
    return sorted(descendants)


def worker_unit_state(unit: str, *, user_scope: bool = True) -> dict:
    """Read Worker systemd state through the existing key-only SSH path."""
    command = (
        "systemctl " + ("--user " if user_scope else "") + "show "
        f"{unit} --property=ActiveState,SubState,Result,ExecMainStatus,"
        "MainPID,ExecMainStartTimestamp,ExecMainExitTimestamp --no-pager"
    )
    result = _command(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            WORKER_HOST,
            command,
        ]
    )
    if result.returncode != 0:
        return {
            "unit": unit,
            "node": "worker78",
            "active_state": "unknown",
            "sub_state": "unknown",
            "result": "unknown",
            "exec_main_status": "unknown",
            "started_at": None,
            "exited_at": None,
            "last_error_line": result.stderr.strip()[-500:] or "Worker SSH unavailable",
        }
    values = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    return {
        "unit": unit,
        "node": "worker78",
        "active_state": values.get("ActiveState", "unknown"),
        "sub_state": values.get("SubState", "unknown"),
        "result": values.get("Result", "unknown"),
        "exec_main_status": values.get("ExecMainStatus", "unknown"),
        "main_pid": int(values.get("MainPID", "0") or 0),
        "started_at": values.get("ExecMainStartTimestamp") or None,
        "exited_at": values.get("ExecMainExitTimestamp") or None,
        "last_error_line": None,
    }


def worker_container_state(container: str, process_needle: str) -> dict:
    """Use the exact owned container to prove a Worker compute payload is live."""
    base = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", WORKER_HOST,
    ]
    inspect = _command(
        base
        + [
            # SSH concatenates command arguments for a remote shell.  Keep the
            # template whitespace-free so it remains one docker --format value.
            "docker", "inspect", "--format={{.State.Running}},{{.State.Pid}}",
            container,
        ]
    )
    if inspect.returncode != 0:
        return {
            "container": container,
            "container_running": False,
            "container_pid": 0,
            "training_process_pids": [],
        }
    fields = inspect.stdout.strip().split(",")
    running = len(fields) == 2 and fields[0].lower() == "true"
    container_pid = int(fields[1]) if running and fields[1].isdigit() else 0
    top = _command(base + ["docker", "top", container, "-eo", "pid,ppid,args"])
    pids = []
    if top.returncode == 0:
        for line in top.stdout.splitlines()[1:]:
            columns = line.strip().split(maxsplit=2)
            if (
                len(columns) == 3
                and columns[0].isdigit()
                and process_needle in columns[2]
                # Wrapper and guard argv repeat the child command after ``--``.
                # They are not evidence that the trainer itself is still alive.
                and "spark_gpu_guard.py" not in columns[2]
            ):
                pids.append(int(columns[0]))
    return {
        "container": container,
        "container_running": running,
        "container_pid": container_pid,
        "training_process_pids": sorted(set(pids)),
    }


def discover_related_units() -> list[str]:
    result = _command(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--all",
            "--plain",
            "--no-legend",
        ]
    )
    names = []
    for line in result.stdout.splitlines():
        columns = line.lstrip("● ").split()
        if columns and (
            columns[0].startswith("fluid-control-train16-")
            or columns[0].startswith(PAIRED_LAMBDA0_POSTEVAL_PREFIX)
            or columns[0].startswith(FC_P003_PREFIX)
            or columns[0].startswith(FC_P003_PROBE_PREFIX)
            or columns[0].startswith(FC_P003_POSTEVAL_PREFIX)
            or columns[0].startswith("fluid-control-fcp003c-")
        ):
            names.append(columns[0])
    return sorted(
        set(
            (
                TRAINING_UNIT,
                MAIN_AUTHORITY_UNIT,
                PAIRED_LAMBDA0_UNIT,
                PAIRED_LAMBDA0_POSTEVAL_UNIT,
                FC_P003_UNIT,
                FC_P003_PROBE_UNIT,
                FC_P003_POSTEVAL_UNIT,
                TRUE_STATE_FORCE_PROBE_V2_UNIT,
                FC_P003C_UNIT,
                FC_P003C_POSTEVAL_UNIT,
                CALIBRATION_UNIT,
                ABSOLUTE_CALIBRATION_UNIT,
                FIXED_FEATURE_READOUT_UNIT,
                *SUPERSEDED_MAIN_UNITS,
                *names,
            )
        )
    )


def _cpu_counters() -> list[int]:
    line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
    return [int(value) for value in line.split()[1:]]


def _cpu_usage(current: list[int], previous: list[int] | None):
    if not previous or len(previous) != len(current):
        return None
    total = sum(current) - sum(previous)
    idle = (current[3] + current[4]) - (previous[3] + previous[4])
    if total <= 0 or idle < 0:
        return None
    return 100.0 * (total - idle) / total


def resource_state(previous: dict | None) -> dict:
    counters = _cpu_counters()
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(("MemAvailable:", "MemTotal:", "MemFree:")):
            key, value, _unit = line.split()
            memory[key.rstrip(":")] = int(value)
    gpu = _command(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [part.strip() for part in gpu.stdout.splitlines()[0].split(",")] if gpu.stdout.strip() else []
    return {
        "cpu_utilization_pct": _cpu_usage(
            counters, previous.get("cpu_counters") if previous else None
        ),
        "cpu_counters": counters,
        "gpu_utilization_pct": float(values[0]) if len(values) == 3 else None,
        "gpu_temperature_c": float(values[1]) if len(values) == 3 else None,
        "gpu_power_w": float(values[2]) if len(values) == 3 else None,
        "mem_available_gib": memory.get("MemAvailable", 0) / 1024**2,
        "mem_free_gib": memory.get("MemFree", 0) / 1024**2,
        "mem_total_gib": memory.get("MemTotal", 0) / 1024**2,
    }


def latest_relevant_file(repo: Path) -> dict | None:
    candidates = []
    for root in (
        repo / MAIN_RUN,
        repo / BALANCED_RUN,
        repo / PAIRED_LAMBDA0_ROOT,
        repo / PAIRED_LAMBDA10_ROOT,
    ):
        if not root.is_dir():
            continue
        for pattern in (
            "train.log",
            "physicsnemo.log",
            "training_history.json",
            "worker_transfer_complete.json",
            "posteval_complete/**/*.log",
            "posteval_complete/**/*.json",
            "posteval_complete_v2/**/*.log",
            "posteval_complete_v2/**/*.json",
            "posteval_worker_v1/**/*.log",
            "posteval_worker_v1/**/*.json",
            "posteval_fc_p001/**/*.log",
            "posteval_fc_p001/**/*.json",
            "completion_receipt.json",
            "worker_transfer_complete.json",
            "training_history.json",
            "train.log",
        ):
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    return {
        "path": str(path.relative_to(repo)),
        "modified_at_utc": modified.isoformat(timespec="seconds"),
    }


def workflow_progress(repo: Path) -> dict:
    main_history = read_json(repo / MAIN_RUN / "training_history.json", [])
    balanced_history = read_json(repo / BALANCED_RUN / "training_history.json", [])
    main_complete, main_receipt_issues = verify_receipt(
        repo / MAIN_RECEIPT, "CONTROL_TRAIN16_POSTEVAL_COMPLETE"
    )
    balanced_complete, balanced_receipt_issues = verify_receipt(
        repo / WORKER_RECEIPT,
        "CONTROL_TRAIN16_H100_LIFT_BALANCED_WORKER_POSTEVAL_COMPLETE",
    )
    return {
        "main_training_epochs": len(main_history) if isinstance(main_history, list) else 0,
        "balanced_training_epochs": (
            len(balanced_history) if isinstance(balanced_history, list) else 0
        ),
        "main_training_complete": (
            isinstance(main_history, list)
            and [row.get("epoch") for row in main_history] == [1, 2]
        ),
        "balanced_training_complete": (
            isinstance(balanced_history, list)
            and [row.get("epoch") for row in balanced_history] == [1, 2]
        ),
        "main_posteval_complete": main_complete,
        "balanced_posteval_complete": balanced_complete,
        "main_receipt_issues": main_receipt_issues,
        "balanced_receipt_issues": balanced_receipt_issues,
        "main_receipt_path": str(MAIN_RECEIPT),
        "balanced_receipt_path": str(WORKER_RECEIPT),
        "scientific_gate_bypassed": False,
    }


def paired_training_progress(repo: Path) -> dict:
    lambda0_complete, lambda0_issues = verify_receipt(
        repo / PAIRED_LAMBDA0_ROOT / "completion_receipt.json",
        PAIRED_TRAINING_STATUS,
    )
    lambda10_complete, lambda10_issues = verify_receipt(
        repo / PAIRED_LAMBDA10_ROOT / "completion_receipt.json",
        PAIRED_TRAINING_STATUS,
    )
    lambda10_transfer_complete, lambda10_transfer_issues = verify_receipt(
        repo / PAIRED_LAMBDA10_ROOT / "worker_transfer_complete.json",
        PAIRED_LAMBDA10_TRANSFER_STATUS,
    )
    lambda0_posteval_complete, lambda0_posteval_issues = verify_receipt(
        repo / PAIRED_LAMBDA0_POSTEVAL_RECEIPT,
        PAIRED_POSTEVAL_STATUS,
    )
    lambda10_posteval_complete, lambda10_posteval_issues = verify_receipt(
        repo / PAIRED_LAMBDA10_POSTEVAL_RECEIPT,
        PAIRED_POSTEVAL_STATUS,
    )
    return {
        "lambda0_training_complete": lambda0_complete,
        "lambda0_receipt_issues": lambda0_issues,
        "lambda10_training_complete": lambda10_complete,
        "lambda10_receipt_issues": lambda10_issues,
        "lambda10_transfer_complete": lambda10_transfer_complete,
        "lambda10_transfer_issues": lambda10_transfer_issues,
        "paired_training_complete": (
            lambda0_complete and lambda10_complete and lambda10_transfer_complete
        ),
        "lambda0_posteval_complete": lambda0_posteval_complete,
        "lambda0_posteval_issues": lambda0_posteval_issues,
        "lambda0_posteval_receipt": str(PAIRED_LAMBDA0_POSTEVAL_RECEIPT),
        "lambda10_posteval_complete": lambda10_posteval_complete,
        "lambda10_posteval_issues": lambda10_posteval_issues,
        "lambda10_posteval_receipt": str(PAIRED_LAMBDA10_POSTEVAL_RECEIPT),
        "paired_posteval_complete": (
            lambda0_posteval_complete and lambda10_posteval_complete
        ),
        "paired_posteval_status": "PAIRED_POSTEVAL_APPROVED_PREFLIGHT",
        "approval_state": "LEAD_APPROVED",
        "approval_reference": str(PAIRED_POSTEVAL_APPROVAL),
        "next_owner": "Surrogate + Physics/Data",
        "approval_required": None,
    }


def _build_legacy_sample(
    repo: Path,
    previous: dict | None,
    units: dict[str, dict],
    resources: dict,
    now: datetime,
) -> dict:
    progress = workflow_progress(repo)
    prior_posteval_stage_complete = (
        progress["main_posteval_complete"]
        and progress["balanced_posteval_complete"]
    )
    paired = paired_training_progress(repo)
    paired_verdict = paired_scientific_verdict(
        repo, paired["paired_posteval_complete"]
    )
    fc_p003_approved = (repo / FC_P003_APPROVAL).is_file()
    fc_p003b_approved = (repo / FC_P003B_APPROVAL).is_file()
    fc_p003c_approved = (repo / FC_P003C_APPROVAL).is_file()
    fc_p003c_full_approved = (repo / FC_P003C_FULL_APPROVAL).is_file()
    fc_p003c_unit = units.get(FC_P003C_UNIT, {})
    fc_p003c_posteval_unit = units.get(FC_P003C_POSTEVAL_UNIT, {})
    fc_p003c_running = (
        fc_p003c_unit.get("active_state") == "active"
        and bool(fc_p003c_unit.get("training_process_pids"))
    )
    fc_p003c_posteval_running = (
        fc_p003c_posteval_unit.get("active_state") == "active"
    )
    fc_p003c_training_complete, fc_p003c_training_issues = verify_receipt(
        repo / FC_P003C_TRAINING_RECEIPT, FC_P003C_TRAINING_STATUS
    )
    fc_p003c_posteval_complete, fc_p003c_posteval_issues = verify_receipt(
        repo / FC_P003C_POSTEVAL_RECEIPT, FC_P003C_POSTEVAL_STATUS
    )
    fc_p003c_gate_status = read_json(
        repo / FC_P003C_DEVELOPMENT_GATE, {}
    ).get("status")
    fc_p003c_rejected = (
        fc_p003c_posteval_complete
        and fc_p003c_gate_status == PAIRED_DEVELOPMENT_FAIL_STATUS
    )
    d015_verified, d015_issues = verify_d015_terminal(repo)
    calibration_contract_verified, calibration_contract_issues = (
        verify_d015_calibration_authorization(repo)
    )
    calibration_unit = units.get(CALIBRATION_UNIT, {})
    calibration_progress = calibration_log_progress(repo)
    calibration_complete, calibration_completion_issues = verify_receipt(
        repo / CALIBRATION_COMPLETION_RECEIPT, CALIBRATION_COMPLETION_STATUS
    )
    calibration_running = (
        calibration_unit.get("active_state") == "active"
        and bool(calibration_unit.get("training_process_pids"))
        and calibration_contract_verified
    )
    absolute_contract_verified, absolute_contract_issues = (
        verify_absolute_calibration_authorization(repo)
    )
    absolute_unit = units.get(ABSOLUTE_CALIBRATION_UNIT, {})
    absolute_progress = calibration_log_progress(
        repo, ABSOLUTE_CALIBRATION_LOG, paired_force_objective="absolute"
    )
    absolute_complete, absolute_completion_issues = verify_receipt(
        repo / ABSOLUTE_CALIBRATION_COMPLETION_RECEIPT,
        ABSOLUTE_CALIBRATION_COMPLETION_STATUS,
    )
    absolute_running = (
        absolute_unit.get("active_state") == "active"
        and bool(absolute_unit.get("training_process_pids"))
        and absolute_contract_verified
    )
    fixed_feature_unit = units.get(FIXED_FEATURE_READOUT_UNIT, {})
    fixed_feature_failure_verified, fixed_feature_failure_issues = (
        verify_fixed_feature_operational_failure(repo)
    )
    fc_p003b_unit = units.get(FC_P003B_UNIT, {})
    fc_p003b_probe_unit = units.get(FC_P003B_PROBE_UNIT, {})
    fc_p003b_posteval_unit = units.get(FC_P003B_POSTEVAL_UNIT, {})
    true_state_probe_verified, true_state_probe_issues = (
        verify_true_state_force_probe(repo / TRUE_STATE_FORCE_PROBE_V2_ROOT)
    )
    fc_p003b_running = (
        fc_p003b_unit.get("active_state") == "active"
        and bool(fc_p003b_unit.get("training_process_pids"))
    )
    fc_p003b_probe_running = (
        fc_p003b_probe_unit.get("active_state") == "active"
        and bool(fc_p003b_probe_unit.get("training_process_pids"))
    )
    fc_p003b_terminal_verified, fc_p003b_terminal_issues = verify_fc_p003b_terminal(
        repo
    )
    fc_p003_authority = select_versioned_authority(
        units, FC_P003_PREFIX, FC_P003_UNIT
    )
    fc_p003_unit = units.get(fc_p003_authority, {})
    fc_p003_probe_authority = select_versioned_authority(
        units, FC_P003_PROBE_PREFIX, FC_P003_PROBE_UNIT
    )
    fc_p003_probe_unit = units.get(fc_p003_probe_authority, {})
    fc_p003_posteval_authority = select_versioned_authority(
        units, FC_P003_POSTEVAL_PREFIX, FC_P003_POSTEVAL_UNIT
    )
    fc_p003_posteval_unit = units.get(fc_p003_posteval_authority, {})
    fc_p003_posteval_pids = fc_p003_posteval_unit.get(
        "posteval_process_pids", []
    )
    fc_p003_running = (
        fc_p003_unit.get("active_state") == "active"
        and bool(fc_p003_unit.get("training_process_pids"))
    )
    fc_p003_probe_running = (
        fc_p003_probe_unit.get("active_state") == "active"
        and bool(fc_p003_probe_unit.get("training_process_pids"))
    )
    fc_p003_launch_present = (repo / FC_P003_LAUNCH_RECEIPT).is_file()
    fc_p003_completion_present = (repo / FC_P003_COMPLETION_RECEIPT).is_file()
    fc_p003_launch_verified, fc_p003_launch_issues = verify_receipt(
        repo / FC_P003_LAUNCH_RECEIPT, FC_P003_LAUNCH_STATUS
    )
    fc_p003_completion_verified, fc_p003_completion_issues = verify_receipt(
        repo / FC_P003_COMPLETION_RECEIPT, FC_P003_COMPLETION_STATUS
    )
    fc_p003_probe_verified, fc_p003_probe_issues = verify_receipt(
        repo / FC_P003_PROBE_ROOT / "completion_receipt.json", FC_P003_PROBE_STATUS
    )
    fc_p003_probe_launch_verified, fc_p003_probe_launch_issues = verify_receipt(
        repo / FC_P003_PROBE_ROOT / "launch_receipt.json", FC_P003_LAUNCH_STATUS
    )
    fc_p003_gate_status = read_json(
        repo / FC_P003_DEVELOPMENT_GATE, {}
    ).get("status")
    if fc_p003_gate_status == PAIRED_DEVELOPMENT_FAIL_STATUS:
        fc_p003_state = "SCIENTIFIC_FAIL_NEEDS_LEAD_NEXT_HYPOTHESIS"
    elif fc_p003_gate_status:
        fc_p003_state = "SCIENTIFIC_RESULT_REQUIRES_AGENT_REVIEW"
    elif fc_p003_running and fc_p003_launch_verified:
        fc_p003_state = "RUNNING"
    elif fc_p003_unit.get("active_state") == "failed":
        fc_p003_state = "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS"
    elif fc_p003_completion_verified and fc_p003_posteval_pids:
        fc_p003_state = "POSTEVAL_RUNNING"
    elif (
        fc_p003_completion_verified
        and fc_p003_posteval_unit.get("active_state") == "active"
    ):
        fc_p003_state = "POSTEVAL_PREFLIGHT_OR_CPU_AUDIT"
    elif fc_p003_completion_verified:
        fc_p003_state = "TRAINING_COMPLETE_POSTEVAL_PENDING"
    elif fc_p003_completion_present or (
        fc_p003_launch_present and not fc_p003_launch_verified
    ):
        fc_p003_state = "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS"
    elif fc_p003_launch_verified:
        fc_p003_state = "PREFLIGHT_COMPLETE_WAITING_RUN"
    elif fc_p003_probe_running and fc_p003_probe_launch_verified:
        fc_p003_state = "RESOURCE_PROBE_RUNNING"
    elif fc_p003_probe_verified:
        fc_p003_state = "RESOURCE_PROBE_PASS_FULL_RUN_PENDING"
    elif fc_p003_probe_unit.get("active_state") == "failed":
        fc_p003_state = "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS"
    elif fc_p003_approved:
        fc_p003_state = "PREFLIGHT_IMPLEMENTATION"
    else:
        fc_p003_state = "PLANNED_NOT_APPROVED"
    progress.update(paired)
    fc_p001_stage_complete = (
        prior_posteval_stage_complete
        and paired["paired_training_complete"]
        and paired["paired_posteval_complete"]
    )
    stage_complete = (
        fc_p003c_posteval_complete
        if fc_p003c_full_approved
        else fc_p003_gate_status is not None
        if fc_p003_approved
        else fc_p001_stage_complete
    )
    pending = True  # The accepted surrogate/controller/real-CFD project goal is unmet.
    main_authority = select_main_authority(units)
    lambda0_posteval_authority = select_versioned_authority(
        units, PAIRED_LAMBDA0_POSTEVAL_PREFIX, PAIRED_LAMBDA0_POSTEVAL_UNIT
    )
    authority_names = (
        main_authority,
        WORKER_AUTHORITY_UNIT,
        lambda0_posteval_authority,
        PAIRED_LAMBDA10_POSTEVAL_UNIT,
        fc_p003_authority,
        fc_p003_probe_authority,
        fc_p003_posteval_authority,
        FC_P003B_POSTEVAL_UNIT,
        TRUE_STATE_FORCE_PROBE_V2_UNIT,
        FC_P003C_UNIT,
        FC_P003C_POSTEVAL_UNIT,
        CALIBRATION_UNIT,
        ABSOLUTE_CALIBRATION_UNIT,
    )
    active_units = sorted(
        name
        for name in authority_names
        if (state := units.get(name, {})).get("active_state") == "active"
        and (
            name != TRUE_STATE_FORCE_PROBE_V2_UNIT
            or bool(state.get("main_pid"))
            or bool(state.get("training_process_pids"))
            or bool(state.get("posteval_process_pids"))
        )
        and (name != CALIBRATION_UNIT or calibration_running)
        and (name != ABSOLUTE_CALIBRATION_UNIT or absolute_running)
    )
    paired_evaluation_active = any(
        name in active_units
        for name in (lambda0_posteval_authority, PAIRED_LAMBDA10_POSTEVAL_UNIT)
    )
    if paired["paired_posteval_complete"]:
        paired["paired_posteval_status"] = (
            "PAIRED_POSTEVAL_COMPLETE_SCIENTIFIC_REJECTED"
            if paired_verdict["status"] == "FC_P001_SCIENTIFIC_REJECTED"
            else "PAIRED_POSTEVAL_COMPLETE_REQUIRES_AGENT_REVIEW"
        )
    elif paired_evaluation_active:
        paired["paired_posteval_status"] = "PAIRED_POSTEVAL_RUNNING"
    progress["paired_posteval_status"] = paired["paired_posteval_status"]
    progress["train_fit_calibration"] = {
        **calibration_progress,
        "state": (
            "RUNNING"
            if calibration_running
            else "EXECUTION_COMPLETE_NOT_ADMISSION"
            if calibration_complete
            else "APPROVED_NOT_RUNNING"
            if calibration_contract_verified
            else "APPROVAL_OR_CODE_BINDING_INVALID"
        ),
        "authority_unit": CALIBRATION_UNIT,
        "active_state": calibration_unit.get("active_state", "unknown"),
        "main_pid": calibration_unit.get("main_pid", 0),
        "training_process_pids": calibration_unit.get("training_process_pids", []),
        "approval_verified": calibration_contract_verified,
        "completion_verified": calibration_complete,
        "completion_issues": calibration_completion_issues,
        "current_mem_available_gib": resources.get("mem_available_gib"),
        "guard_active": calibration_running,
    }
    progress["train_fit_absolute_calibration"] = {
        **absolute_progress,
        "state": (
            "RUNNING"
            if absolute_running
            else "EXECUTION_COMPLETE_NOT_ADMISSION"
            if absolute_complete
            else "APPROVED_NOT_RUNNING"
            if absolute_contract_verified
            else "APPROVAL_OR_CODE_BINDING_INVALID"
        ),
        "authority_unit": ABSOLUTE_CALIBRATION_UNIT,
        "active_state": absolute_unit.get("active_state", "unknown"),
        "main_pid": absolute_unit.get("main_pid", 0),
        "training_process_pids": absolute_unit.get("training_process_pids", []),
        "approval_verified": absolute_contract_verified,
        "approval_issues": absolute_contract_issues,
        "completion_verified": absolute_complete,
        "completion_issues": absolute_completion_issues,
        "current_mem_available_gib": resources.get("mem_available_gib"),
        "guard_active": absolute_running,
    }
    if absolute_contract_verified or absolute_running or absolute_complete:
        # Keep the established dashboard contract; the objective tag distinguishes
        # this approved successor from the completed delta-objective calibration.
        progress["train_fit_calibration"] = progress[
            "train_fit_absolute_calibration"
        ]
    progress["fixed_feature_force_readout"] = {
        "state": (
            "ENGINEERING_FAILURE_NEEDS_FIX"
            if fixed_feature_failure_verified
            else "FAILURE_EVIDENCE_INVALID_OR_PENDING"
            if fixed_feature_unit.get("active_state") == "failed"
            else "NOT_RUNNING"
        ),
        "authority_unit": FIXED_FEATURE_READOUT_UNIT,
        "active_state": fixed_feature_unit.get("active_state", "unknown"),
        "main_pid": fixed_feature_unit.get("main_pid", 0),
        "operational_failure_verified": fixed_feature_failure_verified,
        "operational_failure_issues": fixed_feature_failure_issues,
        "scientific_result_available": False,
        "automatic_retry": False,
    }
    if not pending or active_units:
        idle_since = None
        idle_seconds = 0
    else:
        idle_since = previous.get("no_running_since_utc") if previous else None
        if not idle_since:
            idle_since = now.isoformat(timespec="seconds")
        idle_seconds = max(
            0, int((now - datetime.fromisoformat(idle_since)).total_seconds())
        )
    authority_failures = sorted(
        name
        for name in authority_names
        if (state := units.get(name, {}))
        if state["active_state"] == "failed" or state["result"] == "exit-code"
    )
    failed_units = authority_failures if not prior_posteval_stage_complete else []
    post_completion_incidents = (
        authority_failures if prior_posteval_stage_complete else []
    )
    superseded_failures = sorted(
        name
        for name, state in units.items()
        if name != main_authority
        and (
            name in SUPERSEDED_MAIN_UNITS
            or name.startswith("fluid-control-train16-posteval-main-")
        )
        and (state.get("active_state") == "failed" or state.get("result") == "exit-code")
    )
    tasks = {
        "main_posteval": {
            "authority_unit": main_authority,
            "node": "spark",
            "receipt_path": str(MAIN_RECEIPT),
            "stage_complete": progress["main_posteval_complete"],
        },
        "balanced_posteval": {
            "authority_unit": WORKER_AUTHORITY_UNIT,
            "node": "worker78",
            "receipt_path": str(WORKER_RECEIPT),
            "stage_complete": progress["balanced_posteval_complete"],
        },
        "paired_lambda0_posteval": {
            "authority_unit": lambda0_posteval_authority,
            "node": "spark",
            "receipt_path": str(PAIRED_LAMBDA0_POSTEVAL_RECEIPT),
            "stage_complete": paired["lambda0_posteval_complete"],
        },
        "paired_lambda10_posteval": {
            "authority_unit": PAIRED_LAMBDA10_POSTEVAL_UNIT,
            "node": "worker78",
            "receipt_path": str(PAIRED_LAMBDA10_POSTEVAL_RECEIPT),
            "stage_complete": paired["lambda10_posteval_complete"],
        },
    }
    if (
        fc_p003c_full_approved
        or fc_p003c_running
        or fc_p003c_posteval_running
        or fc_p003c_training_complete
        or fc_p003c_posteval_complete
    ):
        tasks.update(
            {
                "fc_p003c_training": {
                    "authority_unit": FC_P003C_UNIT,
                    "node": "spark",
                    "receipt_path": str(FC_P003C_TRAINING_RECEIPT),
                    "stage_complete": fc_p003c_training_complete,
                },
                "fc_p003c_posteval": {
                    "authority_unit": FC_P003C_POSTEVAL_UNIT,
                    "node": "spark",
                    "receipt_path": str(FC_P003C_POSTEVAL_RECEIPT),
                    "stage_complete": fc_p003c_posteval_complete,
                },
            }
        )
    if calibration_contract_verified or calibration_running or calibration_complete:
        tasks["train_fit_calibration"] = {
            "authority_unit": CALIBRATION_UNIT,
            "node": "spark",
            "receipt_path": str(CALIBRATION_COMPLETION_RECEIPT),
            "stage_complete": calibration_complete,
        }
    if absolute_contract_verified or absolute_running or absolute_complete:
        tasks["train_fit_absolute_calibration"] = {
            "authority_unit": ABSOLUTE_CALIBRATION_UNIT,
            "node": "spark",
            "receipt_path": str(ABSOLUTE_CALIBRATION_COMPLETION_RECEIPT),
            "stage_complete": absolute_complete,
        }
    for task_name, task in tasks.items():
        state = units.get(task["authority_unit"], {})
        if task_name in ("paired_lambda0_posteval", "paired_lambda10_posteval"):
            task["state"] = classify_approved_evaluation(
                state, task["stage_complete"]
            )
            task["automatic_recovery_eligible"] = False
            continue
        task["state"], action_id = classify_authority_task(
            state,
            task["stage_complete"],
            allow_resume=task_name == "main_posteval",
        )
        task["automatic_recovery_eligible"] = action_id is not None
        if action_id:
            task["approved_action_id"] = action_id
    alerts = []
    blocker_reasons = []
    if fc_p003_state == "OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS":
        alerts.append("FC_P003_OPERATIONAL_FAILURE_NEEDS_AGENT_ANALYSIS")
        blocker_reasons.append(
            f"{fc_p003_authority}: approved FC-P003 has no scientific gate; "
            f"result={fc_p003_unit.get('result')} "
            f"status={fc_p003_unit.get('exec_main_status')}; blind restart forbidden"
        )
    needs_analysis = [name for name, task in tasks.items() if task["state"] == "NEEDS_AGENT_ANALYSIS"]
    retryable = [name for name, task in tasks.items() if task["state"] == "RETRY_ELIGIBLE"]
    if pending and needs_analysis:
        alerts.append(
            "PAIRED_POSTEVAL_NEEDS_AGENT_ANALYSIS"
            if any(name.startswith("paired_") for name in needs_analysis)
            else "TRAIN16_POSTEVAL_NEEDS_AGENT_ANALYSIS"
        )
        for task_name in needs_analysis:
            name = tasks[task_name]["authority_unit"]
            state = units.get(
                name,
                {
                    "result": "unknown",
                    "exec_main_status": "unknown",
                    "last_error_line": "authority state missing",
                },
            )
            blocker_reasons.append(
                f"{task_name}/{name}: missing authoritative completion receipt; "
                f"result={state['result']} status={state['exec_main_status']}"
                + (
                    f"; {state['last_error_line']}"
                    if state.get("last_error_line")
                    else ""
                )
            )
    if pending and retryable and not active_units:
        alerts.append("TRAIN16_REVIEWED_RECOVERY_PENDING")
        blocker_reasons.append(
            "Reviewed idempotent Main --resume is eligible for one guarded attempt: "
            + ", ".join(retryable)
        )
    if pending and not active_units and idle_seconds >= IDLE_ALERT_SECONDS:
        paired_wait = (
            prior_posteval_stage_complete
            and paired["paired_training_complete"]
            and not paired["paired_posteval_complete"]
        )
        alerts.append(
            "PAIRED_POSTEVAL_APPROVED_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS"
            if paired_wait
            else "D015_COMPLETE_CALIBRATION_ENGINEERING_WITH_NO_RUNNING_TASK_FOR_300_SECONDS"
            if fc_p003c_rejected
            and d015_verified
            and calibration_contract_verified
            else "D015_COMPLETE_AWAITING_CALIBRATION_CONTRACT_WITH_NO_RUNNING_TASK_FOR_300_SECONDS"
            if fc_p003c_rejected and d015_verified
            else "FC_P003C_REJECTED_D015_IMPLEMENTATION_WITH_NO_RUNNING_TASK_FOR_300_SECONDS"
            if fc_p003c_rejected
            else "TRAIN16_PENDING_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS"
        )
        if paired_wait:
            blocker_reasons.append(
                "FC-P001 paired λ0/λ10 post-evaluation is Lead-approved; "
                "implementation/lineage preflight has no active evaluation unit"
            )
    if resources["mem_available_gib"] < 20.0:
        alerts.append("SPARK_MEMORY_BELOW_20_GIB")
    lambda0_receipt_path = PAIRED_LAMBDA0_ROOT / "completion_receipt.json"
    lambda0_complete = paired["lambda0_training_complete"]
    lambda0_unit = units.get(PAIRED_LAMBDA0_UNIT, {})
    if paired["lambda0_posteval_complete"]:
        lambda0_state = "TRAINING_AND_POSTEVAL_STAGE_COMPLETE"
    elif lambda0_complete:
        lambda0_state = "TRAINING_STAGE_COMPLETE_POSTEVAL_PENDING"
    elif lambda0_unit.get("active_state") == "active":
        lambda0_state = "RUNNING"
    elif lambda0_unit.get("active_state") == "failed":
        lambda0_state = "NEEDS_AGENT_ANALYSIS"
        alerts.append("PAIRED_LAMBDA0_TRAINING_FAILED_NEEDS_AGENT_ANALYSIS")
        blocker_reasons.append(
            f"{PAIRED_LAMBDA0_UNIT}: result={lambda0_unit.get('result')} "
            f"status={lambda0_unit.get('exec_main_status')}; automatic restart forbidden"
        )
    else:
        lambda0_state = "PENDING"
    fc_p003b_posteval_running = (
        fc_p003b_posteval_unit.get("active_state") == "active"
    )
    scientific_status = (
        "D015_TRAIN_FIT_ABSOLUTE_CALIBRATION_RUNNING"
        if absolute_running
        else "D015_TRAIN_FIT_ABSOLUTE_CALIBRATION_COMPLETE_NOT_ADMISSION"
        if absolute_complete
        else "D015_TRAIN_FIT_CALIBRATION_RUNNING"
        if calibration_running
        else "D015_TRAIN_FIT_CALIBRATION_COMPLETE_NOT_ADMISSION"
        if calibration_complete
        else "D015_COMPLETE_TRAIN_FIT_CALIBRATION_ENGINEERING"
        if fc_p003c_rejected and d015_verified and calibration_contract_verified
        else "D015_COMPLETE_AWAITING_CALIBRATION_CONTRACT"
        if fc_p003c_rejected and d015_verified
        else "C_POSTEVAL_COMPLETE_REJECTED_D015_IMPLEMENTATION"
        if fc_p003c_rejected
        else "FC_P003C_TRAINING_AND_IMMUTABLE_POSTEVAL_WAIT_RUNNING"
        if fc_p003c_running and fc_p003c_posteval_running
        else "FC_P003C_TRAINING_RUNNING"
        if fc_p003c_running
        else "FC_P003C_POSTEVAL_RUNNING"
        if fc_p003c_posteval_running
        else
        "FC_P003B_POSTEVAL_RUNNING_FC_P003_REJECTED"
        if fc_p003b_posteval_running
        else "FC_P003B_SCIENTIFIC_FAIL_FC_P003C_IMPLEMENTATION_APPROVED"
        if fc_p003b_terminal_verified and fc_p003c_approved
        else "FC_P003B_SCIENTIFIC_FAIL_NEEDS_MODEL_IMPROVEMENT"
        if fc_p003b_terminal_verified
        else f"FC_P003_{fc_p003_state}"
        if paired_verdict["status"] == "FC_P001_SCIENTIFIC_REJECTED"
        and fc_p003_approved
        else paired["paired_posteval_status"]
        if paired["paired_posteval_complete"]
        else (
            "PAIRED_POSTEVAL_RUNNING"
            if paired_evaluation_active
            else (
                "PAIRED_STATS_CONTROLLED_TRAINING_RUNNING"
                if lambda0_state == "RUNNING"
                else (
                    "PAIRED_POSTEVAL_APPROVED_PREFLIGHT"
                    if paired["paired_training_complete"]
                    else "PAIRED_STATS_CONTROLLED_TRAINING_AND_POSTEVAL_PENDING"
                )
            )
        )
    )
    return {
        "status": "ALERT" if alerts else "MONITORING",
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "policy": {
            "sample_interval_seconds": 60,
            "no_running_alert_seconds": IDLE_ALERT_SECONDS,
            "automatic_restart_or_repair": False,
            "auto_recovery_enabled": PRODUCTION_AUTO_RECOVERY_ENABLED,
            "auto_recovery_reason": (
                "Main strict content-verified --resume at commit 78d827f, at most once; "
                "Worker remains external takeover"
            ),
            "scientific_failure_is_never_bypassed": True,
        },
        "stage_complete": stage_complete,
        "fc_p001_stage_complete": fc_p001_stage_complete,
        "prior_posteval_stage_complete": prior_posteval_stage_complete,
        "project_goal_complete": False,
        "project_status": (
            "NEEDS_MODEL_IMPROVEMENT"
            if prior_posteval_stage_complete
            else "POSTEVAL_INCOMPLETE"
        ),
        "scientific_next_stage": {
            "status": scientific_status,
            "active_work": (
                "d015_train_fit_absolute_calibration_bounded_execution"
                if absolute_running
                else "d015_train_fit_absolute_calibration_complete_awaiting_scientific_review"
                if absolute_complete
                else "d015_train_fit_calibration_bounded_execution"
                if calibration_running
                else "d015_train_fit_calibration_complete_awaiting_scientific_review"
                if calibration_complete
                else "d015_complete_train_fit_calibration_implementation_and_cpu_validation"
                if fc_p003c_rejected
                and d015_verified
                and calibration_contract_verified
                else "d015_complete_calibration_contract_review"
                if fc_p003c_rejected and d015_verified
                else "fc_p003c_rejected_d015_implementation_and_cpu_validation"
                if fc_p003c_rejected
                else "fc_p003c_true_state_force_training_and_immutable_posteval"
                if fc_p003c_running or fc_p003c_posteval_running
                else
                "fc_p003b_unchanged_formal_posteval"
                if fc_p003b_posteval_running
                else "fc_p003c_true_state_force_implementation_and_cpu_tests"
                if fc_p003b_terminal_verified and fc_p003c_approved
                else "fc_p003b_scientific_fail_awaiting_lead_next_hypothesis"
                if fc_p003b_terminal_verified
                else (
                    "fc_p003_interleaved_paired_supervision_training"
                    if fc_p003_state == "RUNNING"
                    else "fc_p003_bounded_resource_probe"
                    if fc_p003_state == "RESOURCE_PROBE_RUNNING"
                    else "fc_p003_unchanged_formal_posteval"
                    if fc_p003_state
                    in {
                        "TRAINING_COMPLETE_POSTEVAL_PENDING",
                        "POSTEVAL_RUNNING",
                        "POSTEVAL_PREFLIGHT_OR_CPU_AUDIT",
                    }
                    else "fc_p003_scientific_fail_awaiting_lead_next_hypothesis"
                    if fc_p003_state == "SCIENTIFIC_FAIL_NEEDS_LEAD_NEXT_HYPOTHESIS"
                    else "fc_p003_interleaved_paired_supervision_preflight"
                )
                if paired_verdict["status"] == "FC_P001_SCIENTIFIC_REJECTED"
                and fc_p003_approved
                else "fc_p002_failure_map"
                if paired["paired_posteval_complete"]
                else (
                    "paired_posteval_running"
                    if paired_evaluation_active
                    else "paired_posteval_approved_preflight"
                    if paired["paired_training_complete"]
                    else "paired_stats_controlled_training"
                )
            ),
            "purpose": (
                "The separately approved absolute-force train-only calibration is "
                "running under the 128/64-step contract and 20 GiB memory guard. "
                "Progress requires flushed events tagged paired_force_objective=absolute; "
                "this run is not scientific admission and does not authorize PPO."
                if absolute_running
                else "The bounded absolute-force train-only calibration execution is "
                "complete but is not scientific admission; its result still requires review."
                if absolute_complete
                else "The separately approved bounded train-only calibration is running "
                "under the 128/64-step contract and 20 GiB memory guard. Progress is "
                "reported only from flushed run.log events; this run is not scientific "
                "admission and does not authorize PPO."
                if calibration_running
                else "The bounded train-only calibration execution is complete but is "
                "not scientific admission; its train-only result still requires review."
                if calibration_complete
                else "D015 is complete and SHA-verified: the existing FC-P003C checkpoint "
                "also misses rear-Cl on its train paired window. The current stage is "
                "CPU implementation and validation of the bounded train-only "
                "calibration contract; no calibration run, scientific improvement, "
                "PPO authorization, or project completion is claimed."
                if fc_p003c_rejected
                and d015_verified
                and calibration_contract_verified
                else "D015 is complete and SHA-verified, but the calibration contract "
                "is absent or differs; calibration work remains fail-closed."
                if fc_p003c_rejected and d015_verified
                else "FC-P003C completed its immutable post-evaluation but failed the "
                "unchanged force-window development gate. PPO remains blocked; "
                "only the approved D015 implementation and CPU validation may proceed."
                if fc_p003c_rejected
                else "FC-P003C is executing the separately approved fixed two-epoch "
                "single-factor training and immutable post-evaluation chain. The "
                "completed one-step mixed-loss probe established only engineering "
                "feasibility; no scientific PASS or PPO authorization exists."
                if fc_p003c_running or fc_p003c_posteval_running
                else
                "FC-P003 is a retained scientific rejection; independently approved "
                "FC-P003B is now completing its unchanged formal post-evaluation and "
                "cannot be admitted before the dynamic and force-window gates finish"
                if fc_p003b_posteval_running
                else "FC-P003 and FC-P003B are retained scientific rejections. "
                "FC-P003C is approved only for implementation and CPU tests of one "
                "true-state per-endpoint paired-force objective; GPU training and PPO "
                "remain unauthorized"
                if fc_p003b_terminal_verified and fc_p003c_approved
                else "FC-P003B completed the reviewed protocol but failed the force-window "
                "development gate; PPO remains blocked and the project remains incomplete"
                if fc_p003b_terminal_verified
                else "Lead-approved FC-P003 changes only paired-update timing from "
                "frontloaded to uniformly interleaved; training completion is not "
                "scientific admission, and a failed gate returns control to Lead for "
                "the next hypothesis"
            ),
            "fc_p001_verdict": paired_verdict,
            "fc_p003": {
                "approval_state": "LEAD_APPROVED" if fc_p003_approved else "PLANNED",
                "approval_reference": str(FC_P003_APPROVAL),
                "state": (
                    fc_p003_state
                ),
                "authority_unit": (
                    fc_p003_authority
                    if fc_p003_running
                    else fc_p003_probe_authority
                    if fc_p003_probe_running
                    else fc_p003_authority
                ),
                "main_pid": (
                    fc_p003_unit.get("main_pid", 0)
                    or fc_p003_probe_unit.get("main_pid", 0)
                ),
                "output_root": str(FC_P003_ROOT),
                "launch_receipt": str(FC_P003_LAUNCH_RECEIPT),
                "launch_receipt_present": fc_p003_launch_present,
                "launch_receipt_verified": fc_p003_launch_verified,
                "launch_receipt_issues": fc_p003_launch_issues,
                "completion_receipt": str(FC_P003_COMPLETION_RECEIPT),
                "completion_receipt_present": fc_p003_completion_present,
                "completion_receipt_verified": fc_p003_completion_verified,
                "completion_receipt_issues": fc_p003_completion_issues,
                "probe_unit": fc_p003_probe_authority,
                "probe_receipt": str(FC_P003_PROBE_ROOT / "completion_receipt.json"),
                "probe_receipt_verified": fc_p003_probe_verified,
                "probe_receipt_issues": fc_p003_probe_issues,
                "probe_launch_receipt_verified": fc_p003_probe_launch_verified,
                "probe_launch_receipt_issues": fc_p003_probe_launch_issues,
                "development_gate": str(FC_P003_DEVELOPMENT_GATE),
                "development_gate_status": fc_p003_gate_status,
                "posteval_unit": fc_p003_posteval_authority,
                "posteval_main_pid": fc_p003_posteval_unit.get("main_pid", 0),
                "posteval_gpu_process_pids": fc_p003_posteval_pids,
                "posteval_receipt": str(FC_P003_ROOT / "posteval_fc_p003/receipt.json"),
                "single_factor": "paired_update_schedule_frontloaded_to_interleaved",
                "automatic_recovery_eligible": False,
            },
            "parallel_cpu_work": {
                "owner": "Physics/Data",
                "status": read_json(
                    repo / "artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json",
                    {},
                ).get("status", "DYNAMIC8_EXISTING_RESTART_PAIR_QC"),
                "scope": "same-restart train-only dynamic8 action/zero pairing audit",
                "training_authorized": False,
                "authorization_scope": (
                    "historical candidate-QC artifact only; FC-P003B training is "
                    "separately Lead-approved below"
                ),
                "reference": "docs/FC-P003_DYNAMIC8_PAIR_CANDIDATE.md",
                "artifact": (
                    "artifacts/fc_p003_dynamic8_pair_candidate_20261005/manifest.json"
                ),
            },
            "fc_p003b": {
                "approval_state": "LEAD_APPROVED" if fc_p003b_approved else "PLANNED",
                "approval_reference": str(FC_P003B_APPROVAL),
                "state": (
                    "TRAINING_RUNNING"
                    if fc_p003b_running
                    else "POSTEVAL_RUNNING"
                    if fc_p003b_posteval_unit.get("active_state") == "active"
                    else "SCIENTIFIC_FAIL_NEEDS_MODEL_IMPROVEMENT"
                    if fc_p003b_terminal_verified
                    else "TECHNICAL_PROBE_RUNNING"
                    if fc_p003b_probe_running
                    else "PROBE_UNIT_ACTIVE_AWAITING_PROCESS_EVIDENCE"
                    if fc_p003b_probe_unit.get("active_state") == "active"
                    else "UNIT_ACTIVE_AWAITING_PROCESS_EVIDENCE"
                    if fc_p003b_unit.get("active_state") == "active"
                    else "PRECHECK"
                    if fc_p003b_approved
                    else "PLANNED_NOT_APPROVED"
                ),
                "authority_unit": (
                    FC_P003B_UNIT
                    if fc_p003b_running
                    else FC_P003B_POSTEVAL_UNIT
                    if fc_p003b_posteval_unit.get("active_state") == "active"
                    or fc_p003b_terminal_verified
                    else FC_P003B_PROBE_UNIT
                    if fc_p003b_probe_unit.get("active_state") == "active"
                    else FC_P003B_UNIT
                ),
                "unit_scope": "system",
                "main_pid": (
                    fc_p003b_unit.get("main_pid", 0)
                    if fc_p003b_running
                    else fc_p003b_posteval_unit.get("main_pid", 0)
                    or fc_p003b_probe_unit.get("main_pid", 0)
                    or fc_p003b_unit.get("main_pid", 0)
                ),
                "container": (
                    fc_p003b_probe_unit.get("container")
                    if fc_p003b_probe_unit.get("active_state") == "active"
                    else fc_p003b_unit.get("container")
                ),
                "container_pid": (
                    fc_p003b_probe_unit.get("container_pid", 0)
                    or fc_p003b_unit.get("container_pid", 0)
                ),
                "training_process_pids": (
                    fc_p003b_probe_unit.get("training_process_pids", [])
                    or fc_p003b_unit.get("training_process_pids", [])
                ),
                "worker_root": (
                    "/home/USER/workspace/fluid_control_fcp003b_dynamic_pairs_20261005"
                ),
                "canonical_spark_root": str(FC_P003B_ROOT),
                "single_factor": "paired_supervision_content_static16_to_dynamic8x2",
                "unique_pair_count": 8,
                "updates_per_epoch": 16,
                "training_authorized": fc_p003b_approved,
                "scientific_result_available": fc_p003b_terminal_verified,
                "terminal_receipt": str(FC_P003B_POSTEVAL_RECEIPT),
                "terminal_receipt_verified": fc_p003b_terminal_verified,
                "terminal_receipt_issues": fc_p003b_terminal_issues,
                "development_gate": str(FC_P003B_DEVELOPMENT_GATE),
                "development_gate_status": read_json(
                    repo / FC_P003B_DEVELOPMENT_GATE, {}
                ).get("status"),
                "transfer_receipt": str(FC_P003B_TRANSFER_RECEIPT),
                "posteval": {
                    "authority_unit": FC_P003B_POSTEVAL_UNIT,
                    "unit_scope": "system",
                    "main_pid": fc_p003b_posteval_unit.get("main_pid", 0),
                    "state": (
                        "WAITING_FOR_TRAINING_COMPLETION"
                        if fc_p003b_posteval_unit.get("active_state") == "active"
                        and fc_p003b_running
                        else "POSTEVAL_RUNNING"
                        if fc_p003b_posteval_unit.get("active_state") == "active"
                        else "SCIENTIFIC_FAIL"
                        if fc_p003b_terminal_verified
                        else "POSTEVAL_NOT_ACTIVE"
                    ),
                    "scientific_result_available": fc_p003b_terminal_verified,
                    "automatic_recovery_eligible": False,
                },
            },
            "fc_p003c": {
                "approval_state": "LEAD_APPROVED_FULL_TRAINING"
                if fc_p003c_full_approved
                else "LEAD_APPROVED_ENGINEERING_ONLY"
                if fc_p003c_approved
                else "NOT_APPROVED",
                "approval_reference": str(FC_P003C_APPROVAL),
                "full_training_approval_reference": str(FC_P003C_FULL_APPROVAL),
                "state": "TRAINING_AND_POSTEVAL_WAIT_RUNNING"
                if fc_p003c_running and fc_p003c_posteval_running
                else "D015_TRAIN_FIT_CALIBRATION_RUNNING"
                if calibration_running
                else "D015_TRAIN_FIT_CALIBRATION_COMPLETE_NOT_ADMISSION"
                if calibration_complete
                else "POSTEVAL_REJECTED_D015_COMPLETE_CALIBRATION_ENGINEERING"
                if fc_p003c_rejected
                and d015_verified
                and calibration_contract_verified
                else "POSTEVAL_REJECTED_D015_COMPLETE_AWAITING_CALIBRATION_CONTRACT"
                if fc_p003c_rejected and d015_verified
                else "POSTEVAL_COMPLETE_REJECTED_D015_IMPLEMENTATION"
                if fc_p003c_rejected
                else "TRAINING_RUNNING"
                if fc_p003c_running
                else "POSTEVAL_RUNNING"
                if fc_p003c_posteval_running
                else "FULL_TRAINING_APPROVED_PREFLIGHT"
                if fc_p003c_full_approved
                else "IMPLEMENTATION_AND_CPU_TESTS"
                if fc_p003c_approved and fc_p003b_terminal_verified
                else "WAITING_FOR_FC_P003B_REVIEW",
                "authority_unit": FC_P003C_UNIT,
                "posteval_authority_unit": FC_P003C_POSTEVAL_UNIT,
                "unit_scope": "user",
                "main_pid": fc_p003c_unit.get("main_pid", 0),
                "training_process_pids": fc_p003c_unit.get(
                    "training_process_pids", []
                ),
                "posteval_main_pid": fc_p003c_posteval_unit.get("main_pid", 0),
                "output_root": str(FC_P003C_ROOT),
                "training_receipt": str(FC_P003C_TRAINING_RECEIPT),
                "training_receipt_verified": fc_p003c_training_complete,
                "training_receipt_issues": fc_p003c_training_issues,
                "posteval_receipt": str(FC_P003C_POSTEVAL_RECEIPT),
                "posteval_receipt_verified": fc_p003c_posteval_complete,
                "posteval_receipt_issues": fc_p003c_posteval_issues,
                "gpu_training_authorized": fc_p003c_full_approved,
                "training_completion_is_scientific_pass": False,
                "mixed_probe_is_scientific_pass": False,
                "ppo_authorized": False,
                "single_factor": "paired_statistic_to_true_state_endpoint_force_loss",
            },
            "d015": {
                "state": (
                    "TRAIN_FIT_CALIBRATION_RUNNING"
                    if calibration_running
                    else "TRAIN_FIT_CALIBRATION_COMPLETE_NOT_ADMISSION"
                    if calibration_complete
                    else "COMPLETE_CALIBRATION_ENGINEERING"
                    if d015_verified and calibration_contract_verified
                    else "COMPLETE_AWAITING_CALIBRATION_CONTRACT"
                    if d015_verified
                    else "IMPLEMENTATION_OR_EXECUTION_PENDING"
                ),
                "result": str(D015_RESULT),
                "result_sha256": D015_RESULT_SHA256,
                "execution_receipt": str(D015_EXECUTION_RECEIPT),
                "completion_verified": d015_verified,
                "completion_issues": d015_issues,
                "calibration_approval": str(CALIBRATION_APPROVAL),
                "calibration_approval_sha256": CALIBRATION_APPROVAL_SHA256,
                "calibration_code_binding_verified": calibration_contract_verified,
                "calibration_binding_issues": calibration_contract_issues,
                "design_reference": str(D015_CALIBRATION_CONTRACT),
                "calibration_execution_authorized": calibration_contract_verified,
                "gpu_running": calibration_running,
                "ppo_authorized": False,
                "project_goal_complete": False,
            },
            "true_state_force_technical_probe": {
                "state": (
                    "TECHNICAL_PROBE_COMPLETE"
                    if true_state_probe_verified
                    else "TECHNICAL_PROBE_RUNNING"
                    if units.get(TRUE_STATE_FORCE_PROBE_V2_UNIT, {}).get(
                        "active_state"
                    )
                    == "active"
                    else "TECHNICAL_PROBE_NEEDS_REVIEW"
                ),
                "authority_unit": TRUE_STATE_FORCE_PROBE_V2_UNIT,
                "unit_scope": "user",
                "receipt": str(
                    TRUE_STATE_FORCE_PROBE_V2_ROOT / "completion_receipt.json"
                ),
                "receipt_verified": true_state_probe_verified,
                "receipt_issues": true_state_probe_issues,
                "scientific_result": False,
                "training_authorized": False,
                "ppo_authorized": False,
            },
            "planned_spark_root": str(PAIRED_DATAPIPE_ROOT),
            "planned_worker_unit": PAIRED_DATAPIPE_WORKER_UNIT,
            "lambda0": {
                "unit": PAIRED_LAMBDA0_UNIT,
                "output_root": str(PAIRED_LAMBDA0_ROOT),
                "source_commit": "b6aada926942161da4430d52664c92db1a2269c7",
                "state": lambda0_state,
                "completion_receipt": str(lambda0_receipt_path),
                "posteval_unit": lambda0_posteval_authority,
                "posteval_state": tasks["paired_lambda0_posteval"]["state"],
                "posteval_receipt": str(PAIRED_LAMBDA0_POSTEVAL_RECEIPT),
            },
            "lambda10": {
                "state": (
                    "TRAINING_AND_POSTEVAL_STAGE_COMPLETE"
                    if paired["lambda10_posteval_complete"]
                    else (
                        "TRAINING_STAGE_COMPLETE_TRANSFER_VERIFIED"
                        if paired["lambda10_transfer_complete"]
                        else "WORKER_TRANSFER_PENDING"
                    )
                ),
                "output_root": str(PAIRED_LAMBDA10_ROOT),
                "posteval_unit": PAIRED_LAMBDA10_POSTEVAL_UNIT,
                "posteval_state": tasks["paired_lambda10_posteval"]["state"],
                "posteval_receipt": str(PAIRED_LAMBDA10_POSTEVAL_RECEIPT),
            },
            "paired_training_complete": paired["paired_training_complete"],
            "paired_posteval_complete": paired["paired_posteval_complete"],
            "paired_posteval_status": paired["paired_posteval_status"],
            "next_owner": paired["next_owner"],
            "approval_required": paired["approval_required"],
            "approval_state": paired["approval_state"],
            "approval_reference": (
                str(FC_P003_APPROVAL) if fc_p003_approved else paired["approval_reference"]
            ),
            "protocol_review": {
                "status": "LAMBDA0_LAMBDA10_PROTOCOL_MATCH_VERIFIED",
                "physicsnemo_image_id": (
                    "sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e"
                ),
                "normalization_sha256": (
                    "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1"
                ),
                "implementation_sha256": PAIRED_PROTOCOL_SHA256,
                "validation10_expected_segments": {
                    "H1": 320,
                    "H10": 320,
                    "H50": 310,
                    "H100": 290,
                },
                "dynamic6_expected_segments": {
                    "H1": 1200,
                    "H10": 1146,
                    "H50": 906,
                    "H100": 606,
                    "all": 3858,
                },
                "training_epoch_metrics_used_for_verdict": False,
                "frozen_test_accessed": False,
            },
            "automatic_restart_allowed": False,
            "formal_training_units_assigned": True,
        },
        "workflow_pending": pending,
        "no_running_since_utc": idle_since,
        "no_running_duration_seconds": idle_seconds,
        "active_units": active_units,
        "failed_units": failed_units,
        "post_completion_incidents": post_completion_incidents,
        "superseded_failures": superseded_failures,
        "authority_tasks": tasks,
        "alerts": alerts,
        "blocker_reasons": blocker_reasons,
        "progress": progress,
        "units": units,
        "resources": resources,
        "latest_relevant_file": latest_relevant_file(repo),
    }


def p013_authority(repo: Path, state: dict) -> dict:
    """Observe only the approved invocation; never authorize an automatic retry."""
    issues = []
    approval = repo / P013_APPROVAL
    if not approval.is_file() or file_sha256(approval) != P013_APPROVAL_SHA256:
        issues.append("formal approval SHA differs or is missing")
    if state.get("invocation_id") != P013_INVOCATION:
        issues.append("formal invocation differs or is missing")
    match = re.search(r"argv\[\]=(.*?) ;", state.get("exec_start", ""))
    try:
        argv = shlex.split(match.group(1)) if match else []
    except ValueError:
        argv = []
    if ("FCP_POSTEVAL_PROFILE=p013" not in argv or
            argv[-2:] != [str(repo / P013_LAUNCHER), "--execute"]):
        issues.append("formal ExecStart profile/immutable launcher/execute differs")
    active, sub = state.get("active_state"), state.get("sub_state")
    if issues or active == "failed" or state.get("result") not in ("success", None):
        classification = "NEEDS_AGENT_ANALYSIS"
    elif active == "active" and sub == "running" and state.get("main_pid", 0) > 0:
        classification = "RUNNING"
    else:
        classification = "TERMINAL_REQUIRES_INDEPENDENT_REVIEW"
    return {"authority_unit": P013_UNIT, "invocation_id": P013_INVOCATION,
            "state": classification, "stage_complete": False,
            "approved_action_id": None, "identity_issues": issues,
            "scientific_admission": False}


def p015_authority(repo: Path, state: dict) -> dict:
    """P015 is an explicitly bound training observation, never a retry authority."""
    issues = []
    approval = repo / P015_APPROVAL
    if not approval.is_file() or file_sha256(approval) != P015_APPROVAL_SHA256:
        issues.append("P015 training approval SHA differs or is missing")
    if state.get("invocation_id") != P015_INVOCATION:
        issues.append("P015 training invocation differs or is missing")
    match = re.search(r"argv\[\]=(.*?) ;", state.get("exec_start", ""))
    try:
        argv = shlex.split(match.group(1)) if match else []
    except ValueError:
        argv = []
    expected = ["/usr/bin/env", "FCP015_APPROVAL_SHA256=" + P015_APPROVAL_SHA256,
                "/bin/bash", str(repo / P015_LAUNCHER), "--execute"]
    if argv != expected:
        issues.append("P015 ExecStart approval/immutable launcher/execute differs")
    active, sub = state.get("active_state"), state.get("sub_state")
    if (issues or active == "failed" or state.get("result") not in ("success", None)
            or str(state.get("exec_main_status", "0")) != "0"):
        classification = "NEEDS_AGENT_ANALYSIS"
    elif active == "active" and sub == "running" and state.get("main_pid", 0) > 0:
        classification = "RUNNING"
    else:
        classification = "TERMINAL_REQUIRES_INDEPENDENT_REVIEW"
    return {"authority_unit": P015_UNIT, "invocation_id": P015_INVOCATION,
            "state": classification, "stage_complete": False,
            "approved_action_id": None, "identity_issues": issues,
            "scientific_admission": False}


def p018_present(repo: Path) -> bool:
    return any((repo / p).exists() for p in (P018_APPROVAL, P018_OBSERVATION, P018_ROOT))


def p018_progress(path: Path, now: datetime) -> dict:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell()-262144))
            log = stream.read().decode(errors="replace")
        updates = None
        for line in log.splitlines():
            try:
                row = json.loads(line)
                if (row.get("event") == "accumulation_update" and type(row.get("update")) is int
                        and 1 <= row["update"] <= 171 and row.get("consumed_windows") == 8*row["update"]
                        and row.get("actual_learning_rate") == 1.5625e-7
                        and row.get("training_protocol_sha256") == P018_PROTOCOL_SHA256):
                    updates = max(updates or 0, row["update"])
            except (ValueError, TypeError, AttributeError):
                continue
        return {"completed_updates": updates, "consumed_windows": None if updates is None else 8*updates,
                "log_age_seconds": max(0.0, now.timestamp()-path.stat().st_mtime)}
    except OSError:
        return {"completed_updates": None, "consumed_windows": None, "log_age_seconds": None}


def p018_authority(repo: Path, state: dict, now: datetime) -> dict:
    issues = []
    base = repo / P018_ROOT
    for relative, expected in {
        P018_APPROVAL: P018_APPROVAL_SHA256, P018_OBSERVATION: P018_OBSERVATION_SHA256,
        P018_ROOT / "execution_approval.json": P018_APPROVAL_SHA256,
        P018_ROOT / "training_protocol.json": P018_PROTOCOL_SHA256,
        Path("docs/FC_P018_TRAINING_PROTOCOL_20261005.json"): P018_PROTOCOL_SHA256,
        P018_ROOT / "immutable_launcher.sh": P018_LAUNCHER_SHA256,
        Path(P018_LAUNCHER): P018_LAUNCHER_SHA256,
    }.items():
        try:
            if file_sha256(repo / relative) != expected: issues.append(f"P018 identity SHA differs: {relative}")
        except OSError:
            issues.append(f"P018 identity evidence missing: {relative}")
    if state.get("invocation_id") != P018_INVOCATION or state.get("load_state") != "loaded":
        issues.append("P018 exact retained unit/invocation missing or changed")
    match = re.search(r"argv\[\]=(.*?) ;", state.get("exec_start", ""))
    try:
        argv = shlex.split(match.group(1)) if match else []
    except ValueError:
        argv = []
    if argv != ["/usr/bin/env", "FCP018_APPROVAL_SHA256="+P018_APPROVAL_SHA256,
                "/bin/bash", str(repo / P018_LAUNCHER), "--execute"]:
        issues.append("P018 ExecStart approval/immutable launcher differs")
    identity_bound = not issues
    for name in ("resource_or_deadline_violation", "resource_watcher_unexpected_exit"):
        if (base / name).exists(): issues.append("P018 resource watchdog failure: "+name)
    if state.get("result") != "success" or state.get("exec_main_status") != "0":
        issues.append("P018 service result/exit status not successful")
    progress = p018_progress(base / "run.log", now)
    active = state.get("active_state") == "active"
    running = active and state.get("sub_state") == "running" and state.get("main_pid", 0)>0 and state.get("main_pid_alive") is True
    terminal = (active and state.get("sub_state") == "exited" and state.get("main_pid") == 0
                and state.get("exec_main_status") == "0" and state.get("exec_main_code") == "1"
                and state.get("result") == "success")
    classification, action = "NEEDS_AGENT_ANALYSIS", "核查执行身份或失败原因，保留日志；不得盲目重启训练。"
    if not issues and running:
        age = progress["log_age_seconds"]
        classification = "RUNNING_PROGRESS_UNAVAILABLE" if age is None else "RUNNING_PROGRESS_STALE" if age>300 else "RUNNING"
        action = "继续观察已批准训练；并行准备终态审计，不执行训练文件扫描。" if classification == "RUNNING" else "进程仍存活；检查最新日志与资源、守卫状态，不能把停滞当作训练结束或自动重启。"
    elif not issues and terminal:
        classification, action = "TERMINAL_AUDIT_PENDING", "成功退出不等于模型合格：由Lead确认并执行P018终态审计和finalizer；本计时器不启动昂贵审计。"
        completion = base / "completion_receipt.json"
        if completion.exists():
            receipt = read_json(completion, {})
            audit_path = base / "candidate_audit.json"
            try:
                audit = read_json(audit_path, {})
                valid = (receipt.get("status") == "FC_P018_TRAINING_COMPLETE_NOT_ADMISSION"
                         and receipt.get("terminal_unit", {}).get("InvocationID") == P018_INVOCATION
                         and receipt.get("candidate_audit_sha256") == file_sha256(audit_path)
                         and audit.get("status") == "FC_P018_DUAL_CANDIDATE_LINEAGE_PASS_NOT_ADMISSION"
                         and audit.get("invocation_id") == P018_INVOCATION
                         and receipt.get("training_approval_sha256") == P018_APPROVAL_SHA256
                         and receipt.get("execution_observation_sha256") == P018_OBSERVATION_SHA256
                         and audit.get("training_protocol_sha256") == P018_PROTOCOL_SHA256
                         and audit.get("candidate_result_sha256") == file_sha256(base / "candidate/result.json")
                         and audit.get("dual_manifest_sha256") == file_sha256(base / "candidate/dual_model_manifest.json"))
                if not valid: raise ValueError("terminal receipt identity differs")
                classification, action = "TERMINAL_AUDITED_RELOAD_REVIEW_PENDING", "终态审计收据已绑定；由Lead核对并批准官方CPU双模型重载，再按原门槛准备正式评估。禁止自动启动评估或PPO。"
            except (OSError, ValueError, AttributeError):
                classification, action = "NEEDS_AGENT_ANALYSIS", "终态收据缺失、损坏或身份冲突；保留现有文件并由Lead核查，不覆盖或重跑训练。"
                issues.append("P018 completion/audit receipt binding invalid")
    elif not issues:
        issues.append("P018 unit neither verified running nor successful retained terminal")
    return {"authority_unit":P018_UNIT,"invocation_id":P018_INVOCATION,"state":classification,
            "running":bool(running and identity_bound),"stage_complete":False,"approved_action_id":None,
            "identity_issues":issues,"progress":progress,"next_action":action,"scientific_admission":False,
            "automatic_recovery_eligible":False,"lead_review_required":classification != "RUNNING"}


def build_sample(repo, previous, units, resources, now):
    legacy = _build_legacy_sample(repo, previous, units, resources, now)
    if (repo/P018_FORMAL_APPROVAL).exists() or (repo/P018_FORMAL_MONITOR).exists():
        task = p018_formal_authority(repo, units.get(P018_FORMAL_UNIT, {}))
        authority, prefix = "p018_formal", "FC_P018_FORMAL_"
    elif p018_present(repo):
        task = p018_authority(repo, units.get(P018_UNIT, {}), now)
        authority, prefix = "p018_training", "FC_P018_TRAINING_"
    elif (repo / P015_APPROVAL).exists():
        task = p015_authority(repo, units.get(P015_UNIT, {}))
        authority, prefix = "p015_training", "FC_P015_TRAINING_"
    elif (repo / P013_APPROVAL).exists():
        task = p013_authority(repo, units.get(P013_UNIT, {}))
        authority, prefix = "p013_formal", "FC_P013_FORMAL_"
    else:
        return legacy
    sample = dict(legacy)
    # Snapshot this sample only: never recursively embed previous samples.
    sample["historical_legacy_sample"] = dict(legacy)
    sample.update(status=prefix + task["state"], stage_complete=False,
                  workflow_pending=True, authority_tasks={authority: task},
                  active_units=[task["authority_unit"]] if task.get("running", task["state"] == "RUNNING") else [],
                  current_authority=authority)
    if authority == "p018_formal":
        sample["timestamp_utc"] = task["observed_utc"]
    if authority == "p015_training":
        sample["historical_p013_authority"] = p013_authority(repo, units.get(P013_UNIT, {}))
    historical_idle_alerts = {
        "PAIRED_POSTEVAL_APPROVED_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS",
        "D015_COMPLETE_CALIBRATION_ENGINEERING_WITH_NO_RUNNING_TASK_FOR_300_SECONDS",
        "D015_COMPLETE_AWAITING_CALIBRATION_CONTRACT_WITH_NO_RUNNING_TASK_FOR_300_SECONDS",
        "FC_P003C_REJECTED_D015_IMPLEMENTATION_WITH_NO_RUNNING_TASK_FOR_300_SECONDS",
        "TRAIN16_PENDING_WITH_NO_RUNNING_UNIT_FOR_300_SECONDS",
        "TRAIN16_REVIEWED_RECOVERY_PENDING",
    }
    sample["alerts"] = [a for a in legacy["alerts"] if a not in historical_idle_alerts]
    if task["state"] != "RUNNING":
        sample["alerts"].append(prefix + task["state"])
    sample["blocker_reasons"] = list(legacy["blocker_reasons"]) + task["identity_issues"]
    if authority in ("p018_training", "p018_formal"):
        sample["next_action"] = task["next_action"]
        sample["blocker_reasons"] = list(task["identity_issues"])
        sample["historical_p015_authority"] = p015_authority(repo, units.get(P015_UNIT, {}))
        for metric in ("mem_available_gib", "mem_free_gib"):
            value = resources.get(metric)
            if value is None or value < 20:
                sample["alerts"].append("FC_P018_RESOURCE_"+metric.upper()+"_MISSING_OR_BELOW_20_GIB")
    if task.get("running", task["state"] == "RUNNING"):
        sample["no_running_since_utc"] = None
        sample["no_running_duration_seconds"] = 0
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/monitor/training_evaluation_watchdog"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = repo / output
    previous = read_json(output / "latest.json", None)
    units = {name: unit_state(name) for name in discover_related_units()}
    if (repo / P013_APPROVAL).exists():
        units[P013_UNIT] = unit_state(P013_UNIT)
    if (repo / P015_APPROVAL).exists():
        units[P015_UNIT] = unit_state(P015_UNIT)
    if p018_present(repo):
        units[P018_UNIT] = unit_state(P018_UNIT)
    if (repo/P018_FORMAL_APPROVAL).exists() or (repo/P018_FORMAL_MONITOR).exists():
        units[P018_FORMAL_UNIT] = unit_state(P018_FORMAL_UNIT)
    units[WORKER_AUTHORITY_UNIT] = worker_unit_state(WORKER_AUTHORITY_UNIT)
    units[PAIRED_LAMBDA10_POSTEVAL_UNIT] = worker_unit_state(
        PAIRED_LAMBDA10_POSTEVAL_UNIT, user_scope=False
    )
    units[FC_P003B_UNIT] = worker_unit_state(FC_P003B_UNIT, user_scope=False)
    units[FC_P003B_UNIT].update(
        worker_container_state(FC_P003B_CONTAINER, "train_tandem_fno_paired_stats.py")
    )
    units[FC_P003B_PROBE_UNIT] = worker_unit_state(
        FC_P003B_PROBE_UNIT, user_scope=False
    )
    units[FC_P003B_PROBE_UNIT].update(
        worker_container_state(
            FC_P003B_PROBE_CONTAINER, "train_tandem_fno_paired_stats.py"
        )
    )
    units[FC_P003B_POSTEVAL_UNIT] = worker_unit_state(
        FC_P003B_POSTEVAL_UNIT, user_scope=False
    )
    resources = resource_state(previous.get("resources") if previous else None)
    sample = build_sample(repo, previous, units, resources, utc_now())
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")
    old_alerts = set(previous.get("alerts", [])) if previous else set()
    new_alerts = set(sample["alerts"]) - old_alerts
    if new_alerts:
        with (output / "alerts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp_utc": sample["timestamp_utc"],
                        "new_alerts": sorted(new_alerts),
                        "blocker_reasons": sample["blocker_reasons"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    atomic_json(output / "latest.json", sample)
    print(json.dumps({key: sample[key] for key in ("status", "alerts", "blocker_reasons")}))


if __name__ == "__main__":
    main()
