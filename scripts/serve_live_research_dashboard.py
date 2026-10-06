#!/usr/bin/env python3
"""Read-only live dashboard for the tandem-cylinder research run.

Run on the primary Spark and access through an SSH local port forward. The
sampler records host CPU, GPU activity, and unified-memory headroom every 10 s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import threading
import time
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

C_EPOCH1_PREVIEW = Path("artifacts/fcp003c_epoch1_flow_visualization_preview_20261005")
C_EPOCH1_SHA = "fac949916859211b24553c410005aaff8ace057ce2bac5e08ac4dec971ecefba"
C_FINAL_PREVIEW = Path("artifacts/fcp003c_final_flow_visualization_20261005")
C_FINAL_SHA = "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"
FIXED_READOUT = Path("artifacts/fcp003c_fixed_feature_force_readout_v2_20261005")
FCP008 = Path("artifacts/fcp008_force_readout_candidate_20261005")
FCP008_RESULT_SHA = "cbfbf7c395e788a3eb78100d064024031b9bcd4cc8bafefdc007c198be769409"

RUN = Path("artifacts/distributed_runs/gateb_multistep_20261002/formal/tandem_fno_total_drag_rollout_seed20261003")
SECOND_RUN = Path("artifacts/distributed_runs/gateb_multistep_seed20261004_20261002/formal/tandem_fno_total_drag_rollout_seed20261004")
V3_WORKER_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_seed20261005_30epoch")
V3_ROLLOUT_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_rollout_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_seed20261005_10epoch")
V3_PRIMARY_ROLLOUT_RUN = Path("artifacts/tandem_fno_gate_b_aug_v3_rollout_seed20261002_10epoch")
V3_REAR_WEIGHTED_RUN = Path("artifacts/tandem_fno_gate_b_aug_v3_rear_drag_seed20261007_10epoch")
V3_H20_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261005_10epoch")
V3_PRIMARY_SEED_H20_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261002_dense_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261002_dense_10epoch")
V3_H20_REAR_DRAG_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch")
V4_H20_DEVELOPMENT_RUN = Path("artifacts/tandem_fno_control_gap_v4_h20_rear_drag_seed20261003_5epoch")
V4_VALIDATION_DECISION = Path(
    "artifacts/tandem_cylinders/control_gap_v4_validation_only_20261003/decision.json"
)
V4_WINDOW_MEAN_CD = Path(
    "artifacts/tandem_cylinders/control_gap_v4_window_mean_cd_20261003.json"
)
V3_PARENT_WINDOW_MEAN_CD = Path(
    "artifacts/tandem_cylinders/control_gap_v4_window_mean_cd_v3_parent_20261003.json"
)
TWO_PHASE_ALTERNATING = Path(
    "artifacts/tandem_cylinders/two_phase_alternating_result_20261003/result.json"
)
LONG_DWELL075 = Path(
    "artifacts/tandem_cylinders/longdwell075_result_20261003/result.json"
)
LOW_ACTION_PHASE94_CANONICAL = Path(
    "artifacts/distributed_runs/control_gap_low_action_phase94_validation_v1_worker78/"
    "low_action_phase94_physical_audit_v3_canonical.json"
)
LOW_ACTION_FNO_ROOT = Path(
    "artifacts/distributed_runs/control_gap_low_action_phase94_validation_v1_worker78/"
    "evaluations"
)
LOW_ACTION_FNO_PAIRWISE = Path(
    "artifacts/tandem_cylinders/low_action_phase94_fno_pairwise_h100_20261003.json"
)
OLD = Path("artifacts/tandem_fno_total_drag_spark_30epoch")
SAMPLE_FILE = Path("artifacts/monitor/live_resource_samples.jsonl")
DUAL_NODE_WATCHDOG = Path(
    "artifacts/monitor/dual_node_watchdog_20261003/latest.json"
)
TRAINING_EVALUATION_WATCHDOG = Path(
    "artifacts/monitor/training_evaluation_watchdog/latest.json"
)
MATCHED_START_DT = 0.005
MATCHED_START_ITERATIONS = 16_000
MATCHED_START_PHASE_MANIFEST_SHA256 = (
    "6279492bd3a79eff868a4333e1f642e3be4a4d58a78e46dc47dde040e4e39603"
)
MATCHED_START_RECEIPTS = Path("artifacts/matched_start_acquisition/transfer_verified")
MATCHED_START_VTK_READY = Path("artifacts/matched_start_acquisition/vtk_ready")
MATCHED_START_CURATOR_LOGS = Path("artifacts/matched_start_acquisition/curator_logs")
MATCHED_START_STAGING = (
    Path("data/curated/.staging") / "matched_start_commissioning_train9_v1"
)
MATCHED_START_PHYSICS_SUMMARY = Path(
    "artifacts/matched_start_acquisition/physics_summary/result.json"
)
FULL40_EXTENSION_ROOT = Path("artifacts/matched_start_full40_extension")
FULL40_SCHEDULER_STATE = FULL40_EXTENSION_ROOT / "scheduler_state.json"
FULL40_RECEIPTS = FULL40_EXTENSION_ROOT / "transfer_verified"
FULL40_TRAIN20_PHYSICS = FULL40_EXTENSION_ROOT / "train20_physics_summary.json"
MATCHED9_FINAL = Path(
    "data/curated/tandem_cylinders_matched_start_commissioning_train9_v1"
)
FULL40_STAGING = Path("data/curated/.staging/matched_start_full40_v1")
DEV30_RELEASE = Path(
    "data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
)
DEV30_QUICKSCREEN_PREFIX = "tandem_fno_full40_dev30_quickscreen_"
DEV30_DIAGNOSTIC_PREFIX = "full40_dev30_validation_"
H50_DYNAMIC_RUN = Path(
    "artifacts/tandem_fno_dynamic_train8_h50_spark_h50_e5_dynamic4_20261004"
)
H50_FIXED_FIGURES = (
    H50_DYNAMIC_RUN / "fixed_field_visualizations_b01_start0/png"
)
DIRECTPPO_TRAIN16_MANIFEST = Path(
    "data/curated/tandem_cylinders_directppo_train16_v1/manifest.json"
)
CANONICAL_PPO_ROOT = Path("artifacts/hydrogym/full40_canonical_joint_v1")
FULL40_PREDECLARATION_SHA256 = (
    "d7ff174ef10194a8739357376335ca13ff9b45c8079970846bb71f15a715d24b"
)
FULL40_AUTHORIZATION_SHA256 = (
    "da8bccaf18a86666ac78804e775c1608d8fc390bfca1484cbd1eaabe93c17151"
)
MATCHED_START_CASES = {
    "matched_start_acquisition_train_b00_zero": ("b00 · zero", 148.0),
    "matched_start_acquisition_train_b00_p075": ("b00 · +0.75", 148.0),
    "matched_start_acquisition_train_b00_m075": ("b00 · −0.75", 148.0),
    "matched_start_acquisition_train_b02_zero": ("b02 · zero", 106.0),
    "matched_start_acquisition_train_b02_p075": ("b02 · +0.75", 106.0),
    "matched_start_acquisition_train_b02_m075": ("b02 · −0.75", 106.0),
    "matched_start_acquisition_train_b04_zero": ("b04 · zero", 120.0),
    "matched_start_acquisition_train_b04_p075": ("b04 · +0.75", 120.0),
    "matched_start_acquisition_train_b04_m075": ("b04 · −0.75", 120.0),
}
FULL40_ACTIONS = {
    "m075": -0.75,
    "m0375": -0.375,
    "zero": 0.0,
    "p0375": 0.375,
    "p075": 0.75,
}
FULL40_CASES = {
    **{
        f"matched_start_acquisition_train_b{phase:02d}_{action}": (
            "train",
            phase,
            target,
        )
        for phase, actions in ((0, ("m0375", "p0375")), (2, ("m0375", "p0375")), (4, ("m0375", "p0375")), (6, tuple(FULL40_ACTIONS)))
        for action in actions
        for target in (FULL40_ACTIONS[action],)
    },
    **{
        f"matched_start_acquisition_{split}_b{phase:02d}_{action}": (
            split,
            phase,
            target,
        )
        for split, phases in (("validation", (1, 5)), ("frozen_test", (3, 7)))
        for phase in phases
        for action, target in FULL40_ACTIONS.items()
    },
}
HOST_COMMAND = (
    "awk '/^cpu /{print}' /proc/stat; "
    "awk '/^MemTotal:|^MemAvailable:/{print}' /proc/meminfo; "
    "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw "
    "--format=csv,noheader,nounits; "
    "printf '__TASKS__\\n'; ps -eo comm=,args=; "
    "printf '__EPOCH__\\n'; "
    "grep -F '{\"epoch\":' /tmp/fluid_control_multistep_seed20261004.log 2>/dev/null | tail -n 1 || true; "
    "printf '__V3_WORKER_EPOCH__\\n'; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/tandem_fno_gate_b_aug_v3_seed20261005_30epoch/training_history.json 2>/dev/null || true; "
    "printf '__V3_ROLLOUT_EPOCH__\\n'; "
    "grep -F '{\"epoch\":' /tmp/fluid_control_v3_rollout_seed20261005.log 2>/dev/null | tail -n 1 || true; "
    "printf '__V3_H20_EPOCH__\\n'; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261005_10epoch/training_history.json 2>/dev/null || true; "
    "printf '__V3_PRIMARY_SEED_H20_EPOCH__\\n'; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261002_dense_10epoch/training_history.json 2>/dev/null || true; "
    "printf '__V3_H20_REAR_DRAG_EPOCH__\\n'; "
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/training_history.json 2>/dev/null || true; "
    "printf '__V4_EPOCH__\\n'; "
    "jq -r 'length' /home/USER/workspace/fluid_control_v4_compute_415a50f/project/artifacts/tandem_fno_control_gap_v4_10epoch_seed20261002/training_history.json 2>/dev/null || true; "
    "printf '__V4_SINGLE_VALIDATION__\\n'; "
    "jq -c '[.cases[].horizons[\"100\"] | {segments,total_drag_rmse,total_drag_target_rms}]' /home/USER/workspace/fluid_control_v4_compute_415a50f/project/artifacts/tandem_fno_control_gap_v4_10epoch_seed20261002/validation_evaluation.json 2>/dev/null || true; "
    "printf '__MATCHED_START__\n'; "
    "matched_root=/home/USER/workspace/fluid_control_v4_compute_415a50f/project/cfd/tandem_cylinders/cases; "
    "for matched_case in matched_start_acquisition_train_b00_zero matched_start_acquisition_train_b00_p075 matched_start_acquisition_train_b00_m075 matched_start_acquisition_train_b02_zero matched_start_acquisition_train_b02_p075 matched_start_acquisition_train_b02_m075 matched_start_acquisition_train_b04_zero matched_start_acquisition_train_b04_p075 matched_start_acquisition_train_b04_m075; do "
    "matched_log=\"$matched_root/$matched_case/log.pimpleFoam.matched_start_acquisition\"; matched_time=; matched_active=0; matched_end=0; "
    "if [ -f \"$matched_log\" ]; then matched_time=$(tail -c 262144 \"$matched_log\" | awk '/^Time = /{t=$3} END{print t}'); tail -c 262144 \"$matched_log\" | grep -q '^End$' && matched_end=1; fi; "
    "pgrep -f \"pimpleFoam -case /case/cases/$matched_case$\" >/dev/null && matched_active=1; "
    "printf '%s|%s|%s|%s\n' \"$matched_case\" \"$matched_time\" \"$matched_active\" \"$matched_end\"; done"
)
PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>fluid_control · 实时科研监控</title>
<style>
:root{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:#0c1422;color:#e5eff9}
*{box-sizing:border-box}body{margin:0}main{max-width:1250px;margin:auto;padding:25px 20px 70px}
h1{font-size:25px;margin:0}h2{font-size:19px;margin:30px 0 11px}h3{font-size:15px;margin:0 0 10px}
.muted{color:#9aafc4}.top{display:flex;justify-content:space-between;align-items:start;gap:20px;flex-wrap:wrap}
.stamp{color:#9aafc4;font-size:13px}.banner{padding:14px 16px;margin:20px 0;border-left:4px solid #edae61;background:#172438;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.card{background:#152237;border:1px solid #2a3d53;border-radius:8px;padding:16px;min-width:0}
.resources{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.number{font-size:25px;font-variant-numeric:tabular-nums;font-weight:650}
.label{font-size:12px;color:#9aafc4}.small{font-size:12px;color:#a9bdd0}.good{color:#79d5a3}.bad{color:#f69d97}.task{margin:11px 0;color:#e5eff9;font-weight:600}
svg{width:100%;height:180px;background:#101b2b;border:1px solid #25374b;border-radius:5px}.tall{height:235px}
.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:#bed0df;margin:6px 0}.sw{display:inline-block;width:11px;height:3px;vertical-align:middle;margin-right:5px}
select{background:#152237;color:#e5eff9;border:1px solid #42617f;padding:7px;border-radius:5px}
img{width:100%;height:auto;background:white;border-radius:4px}.row{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}
canvas.actual-series{width:100%;height:180px;background:#101b2b;border:1px solid #25374b;border-radius:5px;margin-top:8px}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:13px}.summary .card{padding:12px}
.summary b{font-size:19px;display:block;margin:3px 0}.foot{margin-top:35px;border-top:1px solid #2a3d53;padding-top:13px;font-size:12px;color:#9aafc4}
.casegrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin:10px 0}.caseitem{background:#101b2b;border:1px solid #25374b;border-radius:5px;padding:8px;font-size:12px}.caseitem b{display:block;margin-bottom:3px}.caseitem .bar{height:4px;background:#26394e;border-radius:3px;margin-top:5px;overflow:hidden}.caseitem .bar i{display:block;height:100%;background:#60c9fb}
details.archive{margin:18px 0;border:1px solid #2a3d53;border-radius:8px;background:#101b2b;padding:10px 14px}details.archive>summary{cursor:pointer;color:#bed0df;font-weight:650}details.archive[open]>summary{margin-bottom:14px}.current-focus{border-left:4px solid #60c9fb;padding-left:12px;margin-top:10px}
.guide{margin:14px 0;border:1px solid #36516b;border-radius:8px;background:#101b2b;padding:10px 14px}.guide>summary{cursor:pointer;color:#d8e8f5;font-weight:650}.guide-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px;margin-top:10px}.guide-grid div{font-size:12px;color:#aec1d3;line-height:1.55}.guide-grid b{color:#e5eff9}
.field-stack{display:grid;grid-template-columns:1fr;gap:14px;margin-top:12px}.field-card{background:#101b2b;border:1px solid #2a3d53;border-radius:7px;padding:12px}.field-card h3{margin-bottom:5px}.field-note{line-height:1.6;margin-top:10px}
@media(max-width:750px){.grid,.resources,.summary,.casegrid,.guide-grid{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><main>
<div class="top"><div><h1>串列双圆柱流动控制 · 实时进展</h1><div class="muted">目标：降低两圆柱总阻力，同时报告侧向载荷与动作代价</div></div><div class="stamp" id="clock">连接中…</div></div>
<nav class="guide"><a href="#lead-resources" style="color:#79d5a3">计算资源</a> · <a href="#lead-models" style="color:#79d5a3">当前实验</a> · <a href="#flow-current" style="color:#79d5a3">流场预测与误差图</a> · <a href="#legacy-details" style="color:#79d5a3">完整历史证据</a><span class="small"> · 新版科研总览 / 2026-10-04</span></nav>
<section id="lead-overview">
<div class="banner"><b>研究目标不变：</b>只研究串列双圆柱、后圆柱旋转控制。最终在真实 CFD 中验证：总阻力降低 ≥2%，后圆柱升力波动不超过无控制的 1.05 倍，平均升力偏置不超过基准波动的 10%。<br><span class="small">保留已有数据、模型和 PPO。当前重点是确认 FNO 能否准确预测控制动作的长期影响，而不只是单步流场。</span></div>
<div class="card"><h3>当前工作与下一步</h3><div id="lead-now">正在读取实际运行记录…</div><p class="small" id="lead-monitor"></p><div class="small">判断依据：真实任务进程、模型评估结果和实验记录。GPU 忙碌不等于科研目标已完成；训练结束也不等于模型通过验证。</div></div>
<h2>两台计算节点 · 实际资源</h2><div class="grid" id="lead-resources"></div>
<p class="small">Spark 的 CPU 与 GPU 共享物理内存；这里显示系统可用统一内存，不把它当作独立显存。至少保留 20 GiB。页面每 5 秒刷新，资源采样约 10 秒，任务监控约 60 秒。</p>
<h2>当前 FNO 对照实验</h2><div class="grid" id="lead-models"></div>
<p class="small">λ=0：不加配对统计损失；λ=10：加入配对统计损失，比较同一初态下不同动作的阻力与升力变化。两者使用同一评估协议。100 步表示连续预测 10 D/U，并非 100 轮训练。最终还需动态动作和时间窗口内的受力统计检验。</p>
<div class="card" id="current-trial-evidence" hidden><h3>当前长程试验 · 真实 CFD 状态与实际受力</h3><div id="current-trial-recovered-metrics"></div><p id="current-trial-field-note" class="small"></p><img id="current-trial-field" alt="当前周期起点的真实CFD速度模长和ROI去均值压力" loading="lazy"><canvas class="actual-series" id="current-trial-actions" width="1000" height="180"></canvas><canvas class="actual-series" id="current-trial-forces" width="1000" height="180"></canvas><canvas class="actual-series" id="current-trial-lift" width="1000" height="180"></canvas><p class="small">折线只来自已经完成的真实 OpenFOAM 周期：流场样本在周期起点 t，受力在终点 t+0.1；预测值没有混入这些折线。图像仅覆盖 x=8…25、y=4…11 的采样 ROI，不是完整求解域。</p></div>
<div class="card" id="exploratory-h5-ppo-training" hidden><h3>探索性 H5 PPO · 训练证据</h3><div id="exploratory-h5-ppo-training-body"></div></div>
<div class="card" id="exploratory-diverse-h5-ppo-training" hidden><h3>24 个固定真实重置态 · H5 PPO 训练</h3><div id="exploratory-diverse-h5-ppo-training-body"></div></div>
<div class="card" id="exploratory-diverse-h5-32768-ppo" hidden><h3>当前阶段 · 24-reset PPO 延长训练</h3><div id="exploratory-diverse-h5-32768-ppo-body"></div></div>
<div class="card" id="exploratory-diverse-ppo-real-cfd" hidden><h3>24-reset PPO · 新一轮真实 CFD 配对运行</h3><div id="exploratory-diverse-ppo-real-cfd-summary"></div><canvas class="actual-series" id="exploratory-diverse-ppo-actions" width="1000" height="180"></canvas><canvas class="actual-series" id="exploratory-diverse-ppo-drag" width="1000" height="180"></canvas><canvas class="actual-series" id="exploratory-diverse-ppo-lift" width="1000" height="180"></canvas><p class="small">这是新策略的实时真实 OpenFOAM 配对执行；与上方已完成且未达标的旧 PPO 结果分开。当前没有新场图，也不复用旧 PPO/MPC 场图；完成前不声明物理收益或科学准入。</p></div>
<div class="card" id="exploratory-diverse-32768-long-cfd" hidden><h3>当前阶段 · 32768-step PPO 长窗口真实 CFD</h3><div id="exploratory-diverse-32768-long-cfd-summary"></div><canvas class="actual-series" id="exploratory-diverse-32768-long-actions" width="1000" height="180"></canvas><canvas class="actual-series" id="exploratory-diverse-32768-long-drag" width="1000" height="180"></canvas><canvas class="actual-series" id="exploratory-diverse-32768-long-lift" width="1000" height="180"></canvas><p class="small">唯一实际800周期配对运行，不重复另做124周期。主物理窗口预注册为 t=168→228（先丢弃20 D/U）；前124周期只用于与历史短窗作次级比较。CPU策略推理+真实CFD，GPU空闲是预期，不代表任务停滞。</p></div>
<div class="card" id="final-ppo-real-cfd" hidden><h3>冻结最终 PPO · 真实 CFD 配对运行</h3><div id="final-ppo-real-cfd-summary"></div><div id="final-ppo-field-evidence" hidden><p id="final-ppo-field-note" class="small"></p><img id="final-ppo-field" alt="最终PPO实际CFD速度模长和ROI去均值压力" loading="lazy"></div><canvas class="actual-series" id="final-ppo-real-cfd-actions" width="1000" height="180"></canvas><canvas class="actual-series" id="final-ppo-real-cfd-drag" width="1000" height="180"></canvas><canvas class="actual-series" id="final-ppo-real-cfd-lift" width="1000" height="180"></canvas><p class="small">这些折线只来自本次 PPO/zero 两支真实 OpenFOAM 周期终点；不使用上方旧 MPC 流场图。这里若显示场图，只来自本次最终 PPO 分支在 t=160.4 的真实 CFD，也不包含模型预测。运行结束前不声明减阻或科学准入。</p></div>
<section id="flow-current"><h2>历史流场预测 · 真实 CFD / FNO / 误差</h2>
<div class="card"><div class="row"><h3>历史 C 模型 · 第一轮训练预览</h3><select id="c-preview-step"><option value="001">1 步 / 0.1 D/U</option><option value="010">10 步 / 1 D/U</option><option value="050">50 步 / 5 D/U</option><option value="100" selected>100 步 / 10 D/U</option></select></div><p id="c-preview-status">等待预测图及数据校验完成。</p><img id="c-preview-image" alt="第一轮模型：真实 CFD、连续预测及绝对误差" style="width:100%" hidden><p class="small">历史模型可视化：仅一条 b01 动态转速验证轨迹，从 tU/D=130 的真实流场出发，之后连续预测；不是当前长程试验的流场，不是完整验证集的精度，也不是最终模型或闭环控制结果。左列：真实 CFD；中列：模型预测；右列：绝对误差。</p></div>
<p><label for="c-preview-profile">当前候选图的动作轨迹：</label><select id="c-preview-profile" disabled><option value="plus" selected>正向起始旋转</option><option value="zero">无旋转</option><option value="minus">负向起始旋转</option></select></p>
<details><summary>历史模型流场与完整验证结果（不是当前 C 模型）</summary><section>
<div class="card"><div class="row"><h3>最新模型在动态动作上的预测精度</h3><select id="flow-model"><option value="lambda0">λ=0 对照模型</option><option value="lambda10">λ=10 配对统计模型</option></select></div>
<div id="flow-metrics">读取最新评估结果…</div><p class="small">速度误差是 u、v 联合相对 L2；压力误差单独计算。这里是六条动态验证轨迹上全部可用起点的汇总，不能与单个起点的终点阻力门槛混用。ROI 是下游局部流场，不是整个 CFD 域。</p></div>
<div class="card" style="margin-top:12px"><div class="row"><div><h3 id="flow-current-title">流场图片 · 加载中</h3><div class="small" id="flow-current-note"></div></div><div><select id="flow-profile"><option value="zero">无旋转</option><option value="minus">负向起始旋转</option><option value="plus">正向起始旋转</option></select> <select id="flow-step"><option value="001">1 步 / 0.1 D/U</option><option value="010">10 步 / 1 D/U</option><option value="050">50 步 / 5 D/U</option><option value="100" selected>100 步 / 10 D/U</option></select></div></div><img id="flow-current-image" alt="真实 CFD、FNO 流场预测和绝对误差对照" loading="lazy"><p class="small">同一初始流场、相同动作序列下比较；预测越远，误差可能累积。图片仅用于观察流场，控制是否合格仍由阻力和升力统计检验决定。</p></div></section>
</details></section>
<h2>从数据到在线控制 · 哪一步已完成？</h2><div class="grid">
<div class="card"><h3>1 · CFD 数据与物理检查</h3><p>已有真实 OpenFOAM 数据与固定训练／验证／冻结测试划分。继续检查动作、相位和预测时长的覆盖。</p><div class="small">当前 Re=100；不能据此声称跨雷诺数泛化。冻结测试不用于挑选模型。</div></div>
<div class="card"><h3>2 · PhysicsNeMo 流场预测</h3><p>官方 FNO 已完成本轮训练与评估；旋转动作下的升力波动预测仍需改善。</p><div class="small">分开看速度、压力、阻力、升力波动和长时间递推。上方提供最新候选的流场对照与误差；历史图仍保留原模型标签。</div></div>
<div class="card"><h3>3 · MPC / 强化学习</h3><p>MPC 是后续可解释控制对照，尚未启动本轮 MPC 实验。已有 CFD-only PPO 基线保留；当前不是新 PPO 训练。</p><div class="small">HydroGym 提供控制环境接口，SB3 提供 PPO。FNO 通过完整检验后，才开展对应候选模型的策略训练。</div></div>
<div class="card"><h3>4 · 真实 CFD 在线闭环</h3><p>已有 CFD-only PPO 两个初始相位约 4.22% / 4.25% 减阻证据；不等于 FNO 辅助闭环已经成功。</p><div class="small">最终需要 CFD → 状态 → 策略 → 转速 → CFD 的反馈验证，同时满足三项物理指标。两个相位不足以证明广泛泛化。</div></div></div>
<h2>Lead 与专业智能体 · 本阶段职责</h2><div class="grid">
<div class="card"><h3>Lead / Astra · 科研负责人</h3><p>保持研究目标和评估协议，审核实验结论，决定继续控制实验还是先修复模型问题。</p></div>
<div class="card"><h3>Physics / Data · 物理与数据</h3><p>检查无量纲参数、动作覆盖和数据来源；管理计算节点及结果回传。</p></div>
<div class="card"><h3>Surrogate · 代理模型</h3><p>检查两组 FNO 多步预测；按工况、动作和预测时长定位误差。</p></div>
<div class="card"><h3>Control / Evaluation · 控制与评估</h3><p>独立核查协议和指标，维护实验记录与运行监控；判断模型是否可以进入控制阶段。</p></div></div>
<p class="small">上述是本阶段任务分工，不是智能体实时心跳。实际计算任务与采样时间见上方；已完成的评估会保留证据，不伪装成仍在训练。当前两组评估由监控发现异常后交由 agent 诊断处理，不无条件自动重启，也不自动降低验收要求。</p>
</section>
<details class="archive" id="legacy-details"><summary>展开详细证据、真实流场图片、历史实验与术语说明</summary>
<details class="guide" id="page-guide"><summary>怎么看这页 / 术语说明</summary><div class="guide-grid">
<div><b>三层结论：</b>数据完成只说明样本已生成；模型准确要看独立验证误差；控制成功还须真实 CFD 同时通过降阻、升力波动和平均升力三项门槛。</div>
<div><b>epoch：</b>完整看一遍训练数据。<b>H20 / H100：</b>从真实初态连续递推 20 / 100 步，用来检查误差是否随时间累积，不是训练轮数。</div>
<div><b>matched-start：</b>不同动作从完全相同的流场初态出发，才可公平比较。<b>full40 / dev30：</b>40 条完整规划；dev30 只含训练20和验证10，冻结测试10不参与开发。</div>
<div><b>VTK / HDF：</b>VTK 是 OpenFOAM 导出的逐时刻网格场；HDF 是整理后供 PhysicsNeMo 读取的数据。<b>QC：</b>质量与来源校验通过，不等于模型准确或控制有效。</div>
<div><b>joint gate：</b>三项物理指标必须同时达标。<b>macro：</b>先对每个案例算指标、再对案例等权平均；与把所有样本混在一起计算不同。</div>
<div><b>D/U：</b>无量纲流动时间；流体以速度 U 走过一个圆柱直径 D 所需的时间。<b>历史 v4：</b>只作归档对照，不代表当前 full40/dev30 结论。</div>
<div><b>RAW：</b>从计算节点回传的原始求解器数据，不是 rollout。<b>低幅 H100：</b>动作幅值小（±0.75）的100步递推，不是低频。</div>
<div><b>t=120–160：</b>真实 CFD 的时间平均窗口，不是预测第120到160步。</div>
</div></details>
<details class="archive"><summary>历史 v3/v4 与旧开环动作证据（非当前 full40/dev30，点击展开）</summary>
<div class="banner" id="decision">读取历史 v4 阶段判定…</div>
<h2>旧 v4 历史严格证据（非 full40 新链）</h2><div class="summary"><div class="card"><span class="label">v4 验证集 · 第 100 步终点总阻力</span><b id="v4-terminal">—</b><span class="small" id="v4-terminal-detail">PhysicsNeMo FNO 对真实 CFD；冻结测试未访问</span></div><div class="card"><span class="label">v4 验证集 · 100 步窗口平均总阻力</span><b id="v4-window">—</b><span class="small" id="v4-window-detail">PhysicsNeMo FNO 对真实 CFD；等待独立窗口审计</span></div><div class="card"><span class="label">两相位交替旋转 · 真实 OpenFOAM CFD</span><b id="two-phase">—</b><span class="small" id="two-phase-detail">相位匹配零控制；不是代理预测</span></div><div class="card"><span class="label">旧 v4 三项联合状态</span><b id="joint-status">等待</b><span class="small" id="joint-detail">仅历史参考，不代表 full40/dev30 精度</span></div></div>
<div class="small" id="watchdog">读取科研监控…</div>
<div class="summary"><div class="card"><span class="label">旧 v3 冻结测试 · 第 100 步终点参考</span><b id="heldout">—</b><span class="small" id="heldout-scope">旧模型历史证据，不代表 v4</span></div><div class="card"><span class="label">PhysicsNeMo FNO</span><b id="epoch">—</b><span class="small" id="second-seed">计算节点：读取中</span></div><div class="card"><span class="label">OpenFOAM CFD</span><b id="cfd-progress">—</b><span class="small" id="cfd-sub">真实 CFD 生成与审计</span></div><div class="card"><span class="label">HydroGym / PPO</span><b id="hydro-status">—</b><span class="small">等待可信代理和真实 CFD 联合门槛</span></div></div>
<h2>真实 CFD 收益与代理决策是否一致</h2><div class="grid"><div class="card"><h3>配对长窗口 · t=120–160</h3><div class="number" id="physical-drag">读取中…</div><div class="small" id="physical-tradeoff">—</div><div class="small" id="periodic-detail">—</div><div class="small" id="cohort-detail">—</div><div class="small" id="feedback-detail">—</div></div><div class="card"><h3>最新 FNO 与 CFD 动作排序 · 100 步</h3><div class="number" id="ranking-score">读取中…</div><div class="small" id="ranking-detail">—</div><div class="small" id="crossphase-ranking">独立相位短窗口：等待审计。</div></div></div>
</details>
<h2>两台 DGX Spark · 当前任务与算力</h2><div class="grid">
<div class="card"><h3>主节点 · SPARK_HOST</h3><div class="task" id="primary-task">读取中…</div><div class="resources"><div><div class="label">GPU 计算利用率</div><div class="number" id="primary-gpu">—</div></div><div><div class="label">CPU 利用率</div><div class="number" id="primary-cpu">—</div></div><div><div class="label">可用统一内存</div><div class="number" id="primary-mem">—</div></div></div><svg id="primary-chart" role="img" aria-label="主节点 GPU 与 CPU 利用率历史"></svg><div class="small" id="primary-more"></div></div>
<div class="card"><h3>计算节点 · WORKER_HOST</h3><div class="task" id="worker-task">读取中…</div><div class="resources"><div><div class="label">GPU 计算利用率</div><div class="number" id="worker-gpu">—</div></div><div><div class="label">CPU 利用率</div><div class="number" id="worker-cpu">—</div></div><div><div class="label">可用统一内存</div><div class="number" id="worker-mem">—</div></div></div><svg id="worker-chart" role="img" aria-label="计算节点 GPU 与 CPU 利用率历史"></svg><div class="small" id="worker-more"></div></div>
</div><div class="legend"><span><i class="sw" style="background:#60c9fb"></i>GPU</span><span><i class="sw" style="background:#e9ae68"></i>CPU</span><span>GB10 采用统一内存；训练保护线：至少剩余 20 GiB。</span></div>
<h2>双节点运行守护</h2><div class="card"><div class="row"><div><div class="number" id="dual-watch-status">读取 watchdog…</div><div class="small" id="dual-watch-time">等待最新状态 JSON。</div></div><div class="small" id="dual-watch-alerts">告警状态读取中…</div></div><div class="summary"><div class="card"><span class="label">Spark · SPARK_HOST</span><b id="dual-watch-spark">—</b><span class="small" id="dual-watch-spark-detail">读取 CPU / 内存 / 空闲计时…</span></div><div class="card"><span class="label">Worker · WORKER_HOST</span><b id="dual-watch-worker">—</b><span class="small" id="dual-watch-worker-detail">读取 CPU / 内存 / 空闲计时…</span></div><div class="card"><span class="label">采集与整理进度</span><b id="dual-watch-progress">—</b><span class="small" id="dual-watch-progress-detail">读取 receipt / HDF 状态…</span></div><div class="card"><span class="label">自动处置边界</span><b>只告警，不执行</b><span class="small">每 15 秒采样；任务未完成且连续 300 秒无有效项目计算时告警。Spark 保留 20 GiB、Worker 保留 40 GiB；禁止自动启动未授权训练或科研动作。</span></div></div></div>
<details class="archive"><summary>历史数据生产与 commissioning 明细（已完成/非当前核心，点击展开）</summary>
<h2>Matched-start 九案 commissioning</h2><div class="card"><div class="row"><div><div class="number" id="matched-summary">等待真实采样…</div><div class="small">真实 OpenFOAM / RAW / VTK / HDF staging 状态；不是训练结果或控制收益。</div></div><div class="small" id="matched-resource">读取 Worker CPU / 内存与 Spark GPU…</div></div><div class="casegrid" id="matched-cases"></div><div class="small" id="matched-physics">九案开放环物理汇总：等待权威 JSON。</div><div class="small">求解 <code>End</code>、RAW_TRANSFER_VERIFIED、VTK_READY 和 staging HDF 是四个独立门槛；模型训练尚未由此 commissioning 启动，HDF 完成不等于模型已训练。</div></div>
<h2>Matched-start 新增 31 案采集</h2><div class="card"><div class="row"><div><div class="number" id="full40-summary">等待 scheduler 状态…</div><div class="small">顺序为 train → validation → frozen test；求解完成不等于 RAW QC 通过。</div></div><div class="small" id="full40-resource">复用 Worker 真实资源采样…</div></div><div class="small" id="full40-splits">读取 split 摘要…</div><div class="casegrid" id="full40-cases"></div><div class="small">冻结测试仅显示采集与验收状态，绝不展示或用于筛选物理收益；31 案 CFD/RAW 也不代表 Curator、模型训练或精度提升。</div></div>
<div class="card" style="margin-top:12px"><h3>Train20 固定动作物理对照 · TRAIN ONLY</h3><div class="number" id="train20-physics">等待 train20 权威 JSON…</div><div class="small" id="train20-actions">仅显示四个训练相位、同相位零动作参考。</div><div class="small" id="train20-scope">此卡不替代九案 commissioning，不读取 validation / frozen 结果，也不是泛化或闭环证据。</div></div>
</details>
<section id="current-focus" class="current-focus">
<h2>串联双圆柱流动控制 · 三层目标现状</h2><div class="summary"><div class="card"><span class="label">1 · 真实 CFD 反馈控制</span><b id="goal-real-cfd">读取中…</b><span class="small" id="goal-real-cfd-detail">目标：减阻≥2%，同时满足后柱升力约束。</span></div><div class="card"><span class="label">2 · FNO 流场与受力预测</span><b id="goal-fno">读取中…</b><span class="small" id="goal-fno-detail">训练轮次内误差不能替代完整动作响应验证。</span></div><div class="card"><span class="label">3 · FNO 驱动 PPO → 真实 CFD 回放</span><b id="goal-surrogate-control">尚未完成</b><span class="small">必须先通过完整动作响应，再训练代理闭环并回到真实CFD验收。</span></div></div>
<div class="banner"><b>当前结论：</b>真实 OpenFOAM 闭环在两个已测起点减阻约 4.2% 且升力约束通过；PhysicsNeMo FNO 的 Dynamic6 动作精度验收仍为 FAIL，尤其后圆柱升力误差仍需降低；新增 16 条真实 PPO 交互轨迹已通过官方 DataPipe 读取。三项事实相互独立，不能合并成“代理闭环成功”。</div>
<div class="card"><span class="label">正式主线 · train16 真实 PPO 交互数据增强 H100 训练</span><div class="number" id="train16-formal-progress">读取中…</div><div class="small" id="train16-formal-detail">固定正式目录；one-batch 技术 probe 只作接口检查，永不计作候选、epoch或精度成果。</div></div>
<h2>固定真实流场对照 · b01 validation-only</h2><div class="card"><div class="row"><div><b>OpenFOAM 真值 / PhysicsNeMo FNO / 绝对误差</b><div class="small">默认 H100；同一真实初态 start=0，同时展示零转速、负转速、正转速动作。</div></div><label class="small">预测步长 <select id="h50-horizon"><option value="001">H1 · 0.1 D/U</option><option value="010">H10 · 1.0 D/U</option><option value="050">H50 · 5.0 D/U</option><option value="100" selected>H100 · 10.0 D/U</option></select></label></div><div class="field-stack"><div class="field-card"><h3>zero · 零转速基准</h3><img id="h50-zero" hidden alt="b01零转速真实流场、FNO预测与绝对误差"><div class="small" id="h50-zero-status">读取图片…</div></div><div class="field-card"><h3>minus · 负向后圆柱旋转</h3><img id="h50-minus" hidden alt="b01负向旋转真实流场、FNO预测与绝对误差"><div class="small" id="h50-minus-status">读取图片…</div></div><div class="field-card"><h3>plus · 正向后圆柱旋转</h3><img id="h50-plus" hidden alt="b01正向旋转真实流场、FNO预测与绝对误差"><div class="small" id="h50-plus-status">读取图片…</div></div></div><div class="small field-note"><b>范围：</b>仅为圆柱尾迹局部 ROI（x/D=8–25，y/D=4–11，z/D=0.05），不是完整 CFD 计算域。图中 u/U∞、v/U∞ 与 p/(ρU∞²) 为无量纲流体变量；每步为 0.1 D/U。每行真实值与预测值共用色标，绝对误差单独着色。为防极端点遮蔽主体，图内明确标注 1–99 百分位显示范围；这只是可视化截色，不改变误差计算。H100 长递推出现明显空间结构误差，并可见疑似非物理小尺度波动；成因仍需空间谱/梯度诊断，不能在此归因为 aliasing 或只解释为相位偏移。该模型 Dynamic6 正式动作差值门槛未通过，图片不是闭环控制成功证据。</div></div>
<h2>Full40 / dev30 · 当前模型修复实验</h2><div class="card"><span class="label">当前主线：官方 PhysicsNeMo FNO · 纯自回归、无 teacher forcing · 冻结测试未访问</span><div class="summary"><div class="card"><span class="label">A · Spark · 训练 H20 / 验证 H100</span><b id="free-ar-h20">读取中…</b><span class="small" id="free-ar-h20-detail">8轮；每轮直接看H100误差</span></div><div class="card"><span class="label">B · Worker · 训练 H50 / 验证 H100</span><b id="free-ar-h50">读取中…</b><span class="small" id="free-ar-h50-detail">8轮；checkpoint逐轮验SHA回传Spark</span></div><div class="card"><span class="label">上一候选 · Dynamic6 FNO真实CFD诊断</span><b id="free-ar-dynamic">读取中…</b><span class="small" id="free-ar-dynamic-detail">validation only；失败则PPO继续BLOCKED</span></div><div class="card"><span class="label">并行方法 · HydroGym + 真实 OpenFOAM PPO</span><b id="free-ar-direct-ppo">实现与测试中</b><span class="small" id="free-ar-direct-ppo-detail">直接CFD反馈，不依赖FNO代理；尚无运行日志或控制收益</span><img id="free-ar-direct-ppo-figure" hidden alt="真实 OpenFOAM PPO 与零控制的总阻力、后柱升力和动作原始时序" style="margin-top:10px"><span class="small" id="free-ar-direct-ppo-figure-note" hidden></span></div></div><div class="small" id="free-ar-scope" style="margin-top:9px">训练内H100用于逐轮诊断；最终仍须完整validation10、动态动作与真实CFD门槛，不能用训练loss宣称控制成功。</div></div>
<h2>数据与控制证据链</h2><div class="card"><div class="summary"><div class="card"><span class="label">真实 CFD 与严格 QC</span><b id="chain-raw">读取中…</b><span class="small" id="chain-raw-detail">RAW31 与九案状态分开统计</span></div><div class="card"><span class="label">开发集 HDF / immutable 发布</span><b id="chain-hdf">读取中…</b><span class="small" id="chain-release">仅 train20 + validation10；不读取冻结 HDF</span></div><div class="card"><span class="label">Parent qs1 · 非当前训练</span><b id="chain-train">读取中…</b><span class="small" id="chain-train-detail">当前A/B从此固定parent继续训练</span></div><div class="card"><span class="label">Dynamic6 · 真实 OpenFOAM</span><b id="chain-dynamic">读取中…</b><span class="small" id="chain-dynamic-detail">六案严格串行；validation only</span></div><div class="card"><span class="label">HydroGym canonical PPO</span><b id="chain-ppo">BLOCKED</b><span class="small" id="chain-ppo-detail">等待 promotion 与正式 validation gates</span></div></div><div class="small" id="chain-validation" style="margin-top:9px">validation10 诊断尚未生成；quick-screen 不是 formal Gate。</div></div>
<div class="card" style="margin-top:9px"><span class="label">Dynamic6 validation-only 开环物理筛选 · 同相位 zero 对照</span><div class="small" id="chain-dynamic-results">等待六案 aggregate CFD QC；不是FNO预测、PPO闭环或末60D/U最终验收。</div></div>
</section>
<details class="archive"><summary>历史模型图表、冻结测试图与旧控制入口（非当前，点击展开）</summary>
<h2>FNO 训练和推理结果</h2><div class="grid"><div class="card"><h3 id="train-title">FNO：训练轮次 → 预测误差</h3><svg class="tall" id="train-chart" role="img" aria-label="多步训练验证误差"></svg><div class="small">蓝：流场平均绝对误差；橙：四个受力系数平均绝对误差。数值来自验证数据。</div></div><div class="card"><h3 id="error-title">递推步数 → 总阻力预测误差</h3><svg class="tall" id="error-chart" role="img" aria-label="不同预测步长的总阻力误差"></svg><div class="small" id="error-legend">读取评估结果…</div></div></div>
<div class="small" id="v3-metrics" style="margin-top:8px">正在读取新数据 FNO 训练指标…</div><div class="small" id="infer-speed" style="margin-top:4px">正在读取 FNO 推理耗时…</div>
<h2>真实流场 / FNO 预测 / 误差</h2><div class="card"><div class="row"><div class="small" id="figure-label">读取图片…</div><div><select id="case"><option value="expanded_test_00">测试 00</option><option value="expanded_test_01">测试 01</option><option value="expanded_test_02">测试 02</option><option value="expanded_test_04">测试 04</option><option value="expanded_test_05">新测试 05</option></select> <select id="horizon"><option value="001">1 步</option><option value="010">10 步</option><option value="050">50 步</option><option value="100" selected>100 步</option></select></div></div><img id="flow" alt="真实 OpenFOAM 流场、FNO 预测、误差对照"></div>
<h2>HydroGym 闭环控制</h2><div class="card"><div id="cem">—</div><div class="small" id="ppo">—</div></div>
</details>
</details><div class="foot">图表读取原始训练与评估记录。真实 CFD 控制收益仍须通过相位匹配的 OpenFOAM 回放验证。</div>
</main><script>
const $=x=>document.getElementById(x);let latest=null;
function renderLead(d){
 const w=d.training_evaluation_watchdog||{}, p=w.progress||{}, age=(t)=>t?Math.max(0,(Date.now()-Date.parse(t))/1000):Infinity;
 const esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const active=w.active_units||[];
 $('lead-now').textContent=age(w.timestamp_utc)>180?'任务状态已过期，等待重新采样；不能确认任务仍在运行。':active.length?`实际运行 ${active.length} 个任务：两组 FNO 候选的预测评估。训练已结束，完整控制精度结论尚须读取最终评估。`:'当前没有监控到运行任务；需检查阶段完成记录及后续安排，不能据此认定项目完成。';
 $('lead-monitor').textContent=`任务监控：${w.timestamp_utc||'无记录'}；告警 ${ (w.alerts||[]).length } 条；最新输出 ${w.latest_relevant_file?.modified_at_utc||'未知'}。整体目标${w.project_goal_complete?'记录为完成，须核对验收证据':'尚未完成'}。`;
 $('lead-resources').innerHTML=['primary','worker'].map((k,i)=>{const r=(d.resources?.[k]||[]).at(-1)||{}, stale=age(r.time)>60;return `<div class="card"><h3>${i?'计算节点 · WORKER_HOST':'主节点 · SPARK_HOST'}</h3><div class="resources"><div><div class="label">GPU 利用率</div><div class="number">${stale?'—':num(r.gpu)}%</div></div><div><div class="label">CPU 整机利用率</div><div class="number">${stale?'—':num(r.cpu)}%</div></div><div><div class="label">可用统一内存</div><div class="number">${stale?'—':num(r.mem_available_gib)} GiB</div></div></div><p>${stale?'采样过期，不能确认当前负载':esc((r.tasks||[]).join('；')||'未检测到项目计算进程')}</p><div class="small">采样 ${esc(r.time)} · ${!stale&&r.mem_available_gib<20?'警告：低于 20 GiB 保留要求':'保留要求：至少 20 GiB'}</div></div>`}).join('');
 $('lead-models').innerHTML=['lambda0','lambda10'].map((k,i)=>{const c=d.research_overview?.[k]||{}, g=c.endpoint||{}, f=g.h100_force_gate||{}, a=g.h100_start0_action_difference||{};return `<div class="card"><h3>${i?'λ=10 · 配对统计训练':'λ=0 · 对照训练'}</h3><p>本轮训练：${p[k+'_training_complete']?'2 / 2 轮完成':'待核实'}；完整评估：${p[k+'_posteval_complete']?'记录已完成，需查看科学判定':'未完成或结果尚未回传'}</p><div class="number">${pct(f.pooled_total_cd_nrmse)}</div><div class="label">验证集 · 第 100 步总阻力归一化误差（越低越好）</div><p class="small">动作间阻力差预测误差：${num(a.pairwise_delta_cd_mae,5)}；已评估 100 步片段：${f.segments??'待回传'}。</p><div>${g.status==='FULL40_VALIDATION_SURROGATE_READINESS_PASS'?'静态动作终点检验通过；不代表动态／窗口检验通过':g.status?'静态动作终点检验未通过':'等待验证记录回传'}</div><div class="small">证据：${esc(c.path)} · ${esc(c.updated_at)}</div></div>`}).join('');
}
function renderCurrentFlow(d){
 const finalReady=d.fc_p003c_final_preview?.ready===true;
 const preview=finalReady?d.fc_p003c_final_preview:(d.fc_p003c_epoch1_preview||{}), image=$('c-preview-image'), previewH=$('c-preview-step').value;
 const previewProfile=$('c-preview-profile');previewProfile.disabled=!finalReady;if(!finalReady)previewProfile.value='plus';
 image.closest('.card').querySelector('h3').textContent=finalReady?'历史 C 候选 · 第二轮模型预测图':'历史 C 模型 · 第一轮训练预览';
 image.alt=finalReady?'第二轮候选：真实 CFD、连续预测及绝对误差':'第一轮模型：真实 CFD、连续预测及绝对误差';
 image.closest('.card').querySelector('p.small').textContent='每张图对应一条 b01 动作轨迹，从 tU/D=130 的真实流场出发，之后连续预测；不是完整验证集汇总，也不证明闭环控制通过。左列：真实 CFD；中列：模型预测；右列：绝对误差。真实值与预测值共用每幅图的 1–99% 色标，误差使用独立色标。';
 $('c-preview-status').textContent=finalReady?'第二轮候选的三条 b01 动作轨迹已核验，可切换查看；均为单起点可视化，不是全部验证集汇总，也不代表闭环验收通过。':preview.ready?'第二轮候选图尚未就绪，明确保留第一轮模型预览；不是最终候选结果。':'预览尚未生成或文件校验未通过；不展示历史图片代替当前模型。';
 image.hidden=!preview.ready;
 if(preview.ready){const url=finalReady?`/figure/c-final/${previewProfile.value}/${previewH}.png`:`/figure/c-epoch1/${previewH}.png`;if(image.getAttribute('src')!==url)image.src=url;}else image.removeAttribute('src');
 const k=$('flow-model').value,c=d.research_overview?.[k]||{},s=c.dynamic_summary||{};
 const rows=['1','10','50','100'].map(h=>{const r=s[h]||{};return `<tr><td>${h} 步</td><td>${pct(r.velocity_relative_l2)}</td><td>${pct(r.field_relative_l2_u_v_p?.[2])}</td><td>${r.segments??'待回传'}</td></tr>`}).join('');
 $('flow-metrics').innerHTML=`<table style="width:100%;text-align:left;line-height:2"><thead><tr><th>预测时长</th><th>速度相对误差</th><th>压力相对误差</th><th>评估片段</th></tr></thead><tbody>${rows}</tbody></table>`;
 const profile=$('flow-profile').value,h=$('flow-step').value,key=profile+'/'+h,ready=c.figures?.includes(key),im=$('flow-current-image');
 $('flow-current-title').textContent=ready?`最新 ${k==='lambda0'?'λ=0':'λ=10'} · b01 初态 · ${Number(h)} 步预测`:'历史 H50 模型流场参考 · 不是最新候选';
 $('flow-current-note').textContent=ready?'来自本轮候选模型重新生成的真实 CFD / FNO / 误差对照。':'最新候选图片正在生成；暂展示已有真实对比图，模型不同，不能代表上方最新数值。';
 const url=ready?`/figure/paired/${k}/${profile}/${h}.png`:`/figure/h50-dynamic/${profile}/${h}.png`;
 if(im.getAttribute('src')!==url)im.src=url;
}
function drawActualSeries(id,rows,series,title){
 const canvas=$(id),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height,pad=36;
 ctx.clearRect(0,0,W,H);ctx.fillStyle='#101b2b';ctx.fillRect(0,0,W,H);ctx.font='12px sans-serif';ctx.fillStyle='#d8e8f5';ctx.fillText(title,10,16);
 if(!rows.length){ctx.fillStyle='#9aafc4';ctx.fillText('等待首个完整真实CFD周期',10,42);return}
 const values=series.flatMap(s=>rows.map(r=>r[s.key])).filter(Number.isFinite),lo=Math.min(...values),hi=Math.max(...values),span=Math.max(hi-lo,1e-9);
 ctx.strokeStyle='#36516b';ctx.beginPath();ctx.moveTo(pad,48);ctx.lineTo(pad,H-pad);ctx.lineTo(W-10,H-pad);ctx.stroke();
 ctx.fillStyle='#9aafc4';ctx.fillText(hi.toFixed(4),2,54);ctx.fillText(lo.toFixed(4),2,H-pad+4);
 for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>{const x=pad+(W-pad-12)*(rows.length===1?0:i/(rows.length-1)),y=48+(H-pad-48)*(1-(r[s.key]-lo)/span);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
 let x=pad;for(const s of series){ctx.fillStyle=s.color;ctx.fillRect(x,29,12,3);ctx.fillStyle='#bed0df';ctx.fillText(s.label,x+16,34);x+=150}
 ctx.fillStyle='#9aafc4';ctx.fillText(rows[0].force_time.toFixed(1),pad,H-8);ctx.fillText(rows.at(-1).force_time.toFixed(1),W-45,H-8);ctx.fillText('tU/D',W/2,H-8);
}
function renderCurrentTrialEvidence(active){
 const card=$('current-trial-evidence'),field=active?.current_cfd_field,rows=active?.actual_timeseries||[];
 const visible=active?.progress_kind==='exploratory_accelerated_long_h5_feedback'&&field?.verified===true;card.hidden=!visible;if(!visible)return;
 const image=$('current-trial-field'),url=`/current-trial-field.png?v=${field.sha256}`;if(image.getAttribute('src')!==url)image.src=url;
 $('current-trial-field-note').textContent=`真实CFD周期预定起点 t=${field.field_time.toFixed(1)}（NPZ记录 ${field.stored_sample_time.toFixed(8)}，float32时间容差 ${field.time_tolerance.toExponential(2)}）；对应实际受力在终点 t=${field.force_time.toFixed(1)}。速度为求解器单位；压力为CFD pressure with ROI mean removed（solver units），不是绝对Pa，也不是FNO预测场。证据SHA ${field.sha256.slice(0,12)}…`;
 const recovered=active.recovered_metrics,labels={full:'完整 12.4 D/U',first_6p2:'前 6.2 D/U',trailing_6p2:'后 6.2 D/U'};
 $('current-trial-recovered-metrics').innerHTML=recovered?.verified===true?`<p><b>离线后处理已独立复核；原执行 unit 仍为 exit1，未重写成成功。</b></p><table><thead><tr><th>窗口</th><th>MPC / zero 平均总 Cd</th><th>减阻率</th><th>MPC / zero 后柱 Cl′ RMS</th><th>Cl′ RMS 变化</th><th>MPC 平均后柱 Cl</th><th>|均值| / zero Cl′ RMS</th></tr></thead><tbody>${recovered.windows.map(r=>`<tr><td>${labels[r.name]}</td><td>${r.mpc_total_cd_mean.toFixed(5)} / ${r.zero_total_cd_mean.toFixed(5)}</td><td>${(r.drag_reduction_percent>=0?'+':'')+r.drag_reduction_percent.toFixed(3)}%</td><td>${r.mpc_rear_cl_rms.toFixed(5)} / ${r.zero_rear_cl_rms.toFixed(5)}</td><td>${(r.rear_cl_rms_change_percent>=0?'+':'')+r.rear_cl_rms_change_percent.toFixed(3)}%</td><td>${r.mpc_rear_cl_mean.toFixed(5)}</td><td>${r.mean_bias_over_zero_rms_percent.toFixed(3)}%</td></tr>`).join('')}</tbody></table><p class="small">减阻率为正表示相对配对零控制阻力降低；Cl′ RMS变化为负表示波动降低。完整窗口减阻率为负，因此整体未减阻。本表不是正式准入或PPO结果。</p>`:'';
 drawActualSeries('current-trial-actions',rows,[{key:'omega',label:'后圆柱转速',color:'#60c9fb'}],'实际执行动作（周期终点）');
 drawActualSeries('current-trial-forces',rows,[{key:'mpc_total_cd',label:'MPC total Cd',color:'#79d5a3'},{key:'zero_total_cd',label:'zero total Cd',color:'#f2c879'}],'实际总阻力 Cd（终点 t+0.1；无预测值）');
 drawActualSeries('current-trial-lift',rows,[{key:'mpc_rear_cl',label:'MPC rear Cl',color:'#d994ff'},{key:'zero_rear_cl',label:'zero rear Cl',color:'#f69d97'}],'实际后圆柱升力 Cl（终点 t+0.1；无预测值）');
}
function renderAdmission(d){
 if(['lambda0','lambda10'].every(k=>d.research_overview?.[k]?.development?.status==='DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL'))$('lead-now').textContent='两组完整评估已结束：终点指标通过，但旋转动作的升力波动预测未通过。当前进行误差诊断与流场可视化，尚未开始新一轮 PPO。实际计算负载见下方采样。';
 const watch=d.training_evaluation_watchdog||{}, stage=watch.scientific_next_stage||{};
 const fresh=watch.timestamp_utc && Date.now()-Date.parse(watch.timestamp_utc)<180000;
 if(fresh&&stage.status==='FC_P003_RUNNING'){const p=d.current_training_log||{};const progress=Number.isFinite(p.batch_percent)?`最近一条批次记录：${num(p.batch_percent,2)}%，时间 ${p.logged_at_utc}；这不是项目完成比例。`:'';$('lead-now').textContent=`正在训练：配对监督均匀调度对照实验（FC-P003）。已完成 ${p.completed_epochs??'待核实'} / 2 轮。${progress} 模型、数据、监督次数保持不变，只把监督分布到整轮训练中，随后进行完整评估。下方 λ=0/λ=10 是上一轮模型结果。`;}
 const cards=$('lead-models').children;
 ['lambda0','lambda10'].forEach((k,i)=>{const g=d.research_overview?.[k]?.development;if(!g||!cards[i])return;const line=document.createElement('p');const branches=g.window_gate?.branches||[];line.className=g.ppo_authorized===true?'good':'bad';line.textContent=`完整控制精度检验：${g.status==='DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS'?'通过':g.status==='DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL'?'未通过':'待核实'}；时间窗口 ${branches.filter(x=>x.joint_pass===true).length}/${branches.length} 案通过。${g.ppo_authorized===true?'仍需策略训练和真实 CFD 验证。':'当前候选不得进入代理 PPO 训练，继续分析误差。'}`;cards[i].appendChild(line)});
 const isTrueState=!!d.research_overview?.fc_p003c?.receipt_bound;
 const isDynamic=!!d.research_overview?.fc_p003b?.receipt_bound;
 const current=isTrueState?d.research_overview.fc_p003c:(isDynamic?d.research_overview.fc_p003b:d.research_overview?.fc_p003);
 if(current?.receipt_bound){
  const g=current.development, branches=g.window_gate?.branches||[], f=current.endpoint?.h100_force_gate||{}, rejected=g.status==='DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL';
  const card=document.createElement('div');card.className='card';
  const title=document.createElement('h3');title.textContent=isTrueState?'最新结果 · 真实状态动作响应监督（FC-P003C）':isDynamic?'最新结果 · 动态动作配对监督（FC-P003B）':'最新结果 · 均匀配对监督（FC-P003）';card.appendChild(title);
  const verdict=document.createElement('p');verdict.className=rejected?'bad':'';verdict.textContent=`训练与完整评估已完成；控制预测精度${rejected?'未通过':'需结合完整记录复核'}。受力时间窗口 ${branches.filter(x=>x.joint_pass===true).length} / ${branches.length} 个工况通过。`;card.appendChild(verdict);
  const detail=document.createElement('p');detail.textContent=`验证集第100步整体阻力归一化误差 ${pct(f.pooled_total_cd_nrmse)}。端点阻力准确不代表升力波动准确；尚未完成新策略与真实CFD闭环验收。下面 λ=0 / λ=10 为历史对照。`;card.appendChild(detail);
  if(isTrueState){const fields=document.createElement('div');fields.innerHTML='<p>六条动态验证轨迹、全部可用起点的误差（不是下方单起点图片汇总）：</p><table><tr><th>预测步数</th><th>速度相对 L2</th><th>压力相对 L2</th><th>后柱 Cl 平均绝对误差</th></tr>'+['1','100'].map(h=>{const s=current.dynamic_summary?.[h]||{};return `<tr><td>${h}</td><td>${pct(s.velocity_relative_l2)}</td><td>${pct(s.field_relative_l2_u_v_p?.[2])}</td><td>${num(s.rear_cl_mae,4)}</td></tr>`}).join('')+'</table>';card.appendChild(fields);}
  $('lead-models').prepend(card);
  if(fresh&&rejected)$('lead-now').textContent=`${isTrueState?'FC-P003C 当前候选':isDynamic?'动态动作配对候选':'均匀配对候选'}已完成评估，但旋转动作下的升力波动预测未通过。${isTrueState?'下一步为同一模型在训练前段、训练后段与验证后段的单步受力误差诊断；不是新的 PPO 或模型训练。':''}当前计算状态以实时采样为准；完整评估结束不等于闭环完成。`;
 }
}
function renderExploratoryH5PPO(ppo){
 const card=$('exploratory-h5-ppo-training');card.hidden=ppo?.verified!==true;if(card.hidden)return;
 const state=ppo.training_complete?'训练完成，等待真实 CFD 配对验证':ppo.running?'训练中':'训练证据不完整';
 $('exploratory-h5-ppo-training-body').innerHTML=`<p><b>${state}</b> · ${ppo.timesteps}/${ppo.target_timesteps} transitions · ${ppo.environments} 个训练环境</p><table><tbody><tr><th>PPO更新</th><td>${ppo.ppo_updates} epoch-updates；${ppo.optimizer_steps} optimizer steps</td><th>策略张量</th><td>${ppo.policy_changed?'已改变':'未改变'}</td></tr><tr><th>末次 value loss</th><td>${num(ppo.value_loss,6)}</td><th>末次 approx KL</th><td>${num(ppo.approx_kl,6)}</td></tr><tr><th>episode reward均值</th><td>${num(ppo.episode_reward_mean,6)}</td><th>最低 MemAvailable</th><td>${num(ppo.minimum_available_gib,2)} GiB</td></tr></tbody></table><p class="small">官方 K1 双FNO仅作为冻结训练环境：权重SHA前后不变；策略已更新。这里没有执行真实CFD，也没有证明减阻、PPO物理收益或科学准入。下一步必须使用冻结策略做真实CFD配对评价。</p>`;
}
function renderExploratoryDiverseH5PPO(ppo){
 const card=$('exploratory-diverse-h5-ppo-training');card.hidden=ppo?.verified!==true;if(card.hidden)return;
 $('exploratory-diverse-h5-ppo-training-body').innerHTML=`<p><b>训练已完成，等待新的真实 CFD 配对评价</b> · ${ppo.timesteps}/4096 transitions · ${ppo.reset_count} 个固定真实重置态</p><table><tbody><tr><th>PPO更新</th><td>${ppo.ppo_updates} epoch-updates；${ppo.optimizer_steps} optimizer steps</td><th>策略张量</th><td>${ppo.policy_changed?'已改变':'未改变'}</td></tr><tr><th>四相位使用次数</th><td>${ppo.phase_reset_counts.join(' / ')}</td><th>最低 MemAvailable</th><td>${num(ppo.minimum_available_gib,2)} GiB</td></tr><tr><th>episode reward均值</th><td>${num(ppo.episode_reward_mean,6)}</td><th>动作RMS</th><td>${num(ppo.applied_omega_rms,6)}</td></tr></tbody></table><p class="small">与旧四个zero起点训练分开显示：本次唯一训练分布变化是四相位×六种固定真实重置态。官方K1双FNO全程冻结且张量SHA不变；策略已改变。这里没有执行真实CFD，也没有证明减阻、泛化或科学准入。</p>`;
}
function renderExploratoryDiverse32768PPO(ppo){
 const card=$('exploratory-diverse-h5-32768-ppo');card.hidden=ppo?.verified!==true;if(card.hidden)return;
 const pctDone=100*ppo.timesteps/ppo.target_timesteps;
 $('exploratory-diverse-h5-32768-ppo-body').innerHTML=`<p><b>${ppo.running?'实际GPU策略训练中':'训练已停止，等待终态核验'}</b> · ${ppo.timesteps}/${ppo.target_timesteps} transitions（${pctDone.toFixed(1)}%）· 24个固定真实重置态</p><table><tbody><tr><th>最近PPO更新</th><td>${ppo.ppo_updates} epoch-updates</td><th>运行设备</th><td>GPU 0（仅PPO策略）</td></tr><tr><th>当前 MemAvailable</th><td>${num(ppo.current_available_gib,2)} GiB</td><th>最低 MemAvailable</th><td>${num(ppo.minimum_available_gib,2)} GiB</td></tr><tr><th>episode reward均值</th><td>${num(ppo.episode_reward_mean,6)}</td><th>FNO</th><td>冻结，不更新权重</td></tr></tbody></table><p class="small">R1因审批JSON数值类型不一致在训练前失败，未产生GPU训练；本卡只显示已观察到实际transition的R2。训练进度不是物理收益。此前4096步训练及其真实CFD结果仍保留：完整窗仅减阻0.506%，未达到≥2%目标。</p>`;
}
function renderExploratoryDiversePPORealCFD(run){
 const card=$('exploratory-diverse-ppo-real-cfd');card.hidden=run?.verified!==true;if(card.hidden)return;
 const rows=run.actual_timeseries||[],latest=run.latest||{},terminal=run.terminal_result,labels={full:'完整 12.4 D/U',first_6p2:'前 6.2 D/U',trailing_6p2:'后 6.2 D/U'};
 const table=terminal?.verified?`<p><b>独立终态复核：124周期真实CFD已完成，但没有达到原物理目标。</b> 完整窗口减阻 ${terminal.windows[0].drag_reduction_percent.toFixed(3)}%（目标≥2%）；Cl′ RMS变化 ${terminal.windows[0].rear_cl_rms_change_percent.toFixed(3)}%；平均Cl偏置比 ${terminal.windows[0].mean_bias_percent.toFixed(2)}%（原门槛≤10%）。</p><table><thead><tr><th>窗口</th><th>PPO / zero平均总Cd</th><th>减阻率</th><th>Cl′ RMS变化</th><th>|平均Cl| / zero RMS</th></tr></thead><tbody>${terminal.windows.map(r=>`<tr><td>${labels[r.name]}</td><td>${r.ppo_total_cd_mean.toFixed(5)} / ${r.zero_total_cd_mean.toFixed(5)}</td><td>${r.drag_reduction_percent.toFixed(3)}%</td><td>${r.rear_cl_rms_change_percent.toFixed(3)}%</td><td>${r.mean_bias_percent.toFixed(2)}%</td></tr>`).join('')}</tbody></table><p class="small">20%均值偏置只作敏感性参考：三窗会通过20%但仍未达到2%减阻，且原10%门槛全部失败。动作最大绝对值 ${terminal.max_abs_applied_omega.toFixed(3)}，没有饱和终点。12.4 D/U也不是80 D/U正式准入；本卡不构成科学准入。</p>`:'';
 $('exploratory-diverse-ppo-real-cfd-summary').innerHTML=`<p><b>${terminal?.verified?'真实 CFD 已完成，目标未完成':run.running?'真实 CFD 正在运行':'真实 CFD 已停止，等待终态复核'}</b> · ${run.completed_cycles}/124 个配对周期 · CPU策略推理，无GPU训练</p><p>最近周期 t=${num(latest.force_time,1)}：请求/实际动作 ${num(latest.requested_omega,3)} / ${num(latest.omega,3)}；PPO/zero 总 Cd ${num(latest.ppo_total_cd,5)} / ${num(latest.zero_total_cd,5)}；PPO/zero 后柱 Cl ${num(latest.ppo_rear_cl,5)} / ${num(latest.zero_rear_cl,5)}。最低 MemAvailable ${num(run.minimum_available_gib,2)} GiB。</p>${table}<p class="small">策略来自24个固定真实重置态的4096步训练；FNO训练权重保持冻结。本卡与旧策略的真实CFD负面结果分开，也不把下一轮32768步准备误写成正在训练。</p>`;
 drawActualSeries('exploratory-diverse-ppo-actions',rows,[{key:'requested_omega',label:'请求转速',color:'#60c9fb'},{key:'omega',label:'实际转速',color:'#79d5a3'}],'24-reset PPO动作（周期终点）');
 drawActualSeries('exploratory-diverse-ppo-drag',rows,[{key:'ppo_total_cd',label:'PPO total Cd',color:'#79d5a3'},{key:'zero_total_cd',label:'zero total Cd',color:'#f2c879'}],'新策略真实CFD总阻力 Cd（周期终点）');
 drawActualSeries('exploratory-diverse-ppo-lift',rows,[{key:'ppo_rear_cl',label:'PPO rear Cl',color:'#d994ff'},{key:'zero_rear_cl',label:'zero rear Cl',color:'#f69d97'}],'新策略真实CFD后柱升力 Cl（周期终点）');
}
function renderExploratoryDiverse32768LongCFD(run){
 const card=$('exploratory-diverse-32768-long-cfd');card.hidden=run?.verified!==true;if(card.hidden)return;
 const rows=run.actual_timeseries||[],latest=run.latest||{},pctDone=100*run.completed_cycles/800;
 $('exploratory-diverse-32768-long-cfd-summary').innerHTML=`<p><b>${run.running?'800周期真实CFD正在运行':'运行已停止，等待终态复核'}</b> · ${run.completed_cycles}/800（${pctDone.toFixed(1)}%）· t=${num(latest.force_time,1)} / 228.0</p><p>最近周期：请求/实际动作 ${num(latest.requested_omega,3)} / ${num(latest.omega,3)}；PPO/zero总Cd ${num(latest.ppo_total_cd,5)} / ${num(latest.zero_total_cd,5)}；PPO/zero后柱Cl ${num(latest.ppo_rear_cl,5)} / ${num(latest.zero_rear_cl,5)}。当前/最低MemAvailable ${num(run.current_available_gib,2)} / ${num(run.minimum_available_gib,2)} GiB。</p><p class="small">32768-step策略训练已完成且冻结；这里没有GPU训练或在线FNO。主窗口尚未完整时不计算最终物理成败，也不把前124周期当作正式80 D/U结论。</p>`;
 drawActualSeries('exploratory-diverse-32768-long-actions',rows,[{key:'requested_omega',label:'请求转速',color:'#60c9fb'},{key:'omega',label:'实际转速',color:'#79d5a3'}],'32768-step PPO实际动作（周期终点）');
 drawActualSeries('exploratory-diverse-32768-long-drag',rows,[{key:'ppo_total_cd',label:'PPO total Cd',color:'#79d5a3'},{key:'zero_total_cd',label:'zero total Cd',color:'#f2c879'}],'长窗口真实CFD总阻力 Cd（周期终点）');
 drawActualSeries('exploratory-diverse-32768-long-lift',rows,[{key:'ppo_rear_cl',label:'PPO rear Cl',color:'#d994ff'},{key:'zero_rear_cl',label:'zero rear Cl',color:'#f69d97'}],'长窗口真实CFD后柱升力 Cl（周期终点）');
}
function renderFinalPPORealCFD(run){
 const card=$('final-ppo-real-cfd');card.hidden=run?.verified!==true;if(card.hidden)return;
 const rows=run.actual_timeseries||[],latest=run.latest||{},terminal=run.terminal_review,labels={full:'完整 12.4 D/U',first_6p2:'前 6.2 D/U',trailing_6p2:'后 6.2 D/U'};
 const table=terminal?.verified?`<p><b>独立终态复核：没有同时满足物理约束，不能认定控制成功。</b> 124次策略请求均为 +0.75；实际动作经速率限制后117个终点饱和在+0.75。</p><table><thead><tr><th>窗口</th><th>PPO / zero平均总Cd</th><th>减阻率</th><th>后柱Cl′ RMS变化</th><th>PPO平均后柱Cl</th><th>|均值| / 配对zero RMS</th><th>10%原门槛 / 20%敏感性</th></tr></thead><tbody>${terminal.windows.map(r=>`<tr><td>${labels[r.name]}</td><td>${r.ppo_total_cd_mean.toFixed(5)} / ${r.zero_total_cd_mean.toFixed(5)}</td><td>${(r.drag_reduction_percent>=0?'+':'')+r.drag_reduction_percent.toFixed(3)}%</td><td>${(r.rear_cl_rms_change_percent>=0?'+':'')+r.rear_cl_rms_change_percent.toFixed(3)}%</td><td>${r.ppo_rear_cl_mean.toFixed(5)}</td><td>${r.mean_bias_percent.toFixed(2)}%</td><td>${r.passes_original_10_percent_mean_bias?'PASS':'FAIL'} / ${r.passes_sensitivity_20_percent_mean_bias?'PASS':'FAIL'}</td></tr>`).join('')}</tbody></table><p class="small">20%只作敏感性参考，不改变原10%均值偏置标准。以固定train-b00 zero RMS作分母时三窗也均失败。完整窗虽减阻0.412%，但Cl′ RMS增加10.586%、均值偏置52.731%，因此不是受约束物理成功。</p>`:'';
 $('final-ppo-real-cfd-summary').innerHTML=`<p><b>${terminal?.verified?'真实 CFD 已完成并独立复核':run.running?'真实 CFD 正在运行':'真实 CFD 已停止，等待终态复核'}</b> · ${run.completed_cycles}/${run.planned_cycles} 个配对周期 · CPU策略推理，无GPU训练</p><p>最近周期 t=${num(latest.force_time,1)}：实际动作 ${num(latest.omega,3)}；PPO/zero 总 Cd ${num(latest.ppo_total_cd,5)} / ${num(latest.zero_total_cd,5)}；PPO/zero 后柱 Cl ${num(latest.ppo_rear_cl,5)} / ${num(latest.zero_rear_cl,5)}。最低 MemAvailable ${num(run.minimum_available_gib,2)} GiB。</p>${table}<p class="small">策略训练已完成4096步，本卡是冻结最终策略直接驱动真实CFD的配对执行；没有在线FNO、没有MPC替代，也没有科学准入。</p>`;
 const field=run.actual_cfd_field,fieldBox=$('final-ppo-field-evidence');fieldBox.hidden=field?.verified!==true;
 if(!fieldBox.hidden){const image=$('final-ppo-field'),url=`/final-ppo-field.png?v=${field.sha256}`;if(image.getAttribute('src')!==url)image.src=url;$('final-ppo-field-note').textContent=`最终 PPO 分支实际 CFD t=${field.intended_time.toFixed(1)}（NPZ记录 ${field.stored_sample_time.toFixed(8)}；float32时间容差 ${field.time_tolerance.toExponential(2)}）。速度为求解器单位；压力为 CFD pressure with ROI mean removed（solver units），不是绝对Pa，也不是模型预测场。证据SHA ${field.sha256.slice(0,12)}…`;}
 drawActualSeries('final-ppo-real-cfd-actions',rows,[{key:'omega',label:'PPO实际转速',color:'#60c9fb'}],'最终PPO实际动作（周期终点）');
 drawActualSeries('final-ppo-real-cfd-drag',rows,[{key:'ppo_total_cd',label:'PPO total Cd',color:'#79d5a3'},{key:'zero_total_cd',label:'zero total Cd',color:'#f2c879'}],'真实CFD总阻力 Cd（周期终点）');
 drawActualSeries('final-ppo-real-cfd-lift',rows,[{key:'ppo_rear_cl',label:'PPO rear Cl',color:'#d994ff'},{key:'zero_rear_cl',label:'zero rear Cl',color:'#f69d97'}],'真实CFD后柱升力 Cl（周期终点）');
}
function renderActiveExperiment(d){
 const active=d.registered_experiment;
 renderCurrentTrialEvidence(active);
 renderExploratoryH5PPO(d.exploratory_h5_ppo_training);
 renderExploratoryDiverseH5PPO(d.exploratory_diverse_h5_ppo_training);
 renderExploratoryDiverse32768PPO(d.exploratory_diverse_h5_32768_ppo_training);
 renderExploratoryDiversePPORealCFD(d.exploratory_diverse_ppo_real_cfd);
 renderExploratoryDiverse32768LongCFD(d.exploratory_diverse_32768_long_cfd);
 renderFinalPPORealCFD(d.exploratory_final_ppo_real_cfd);
 if(active?.mpc_trial===true&&active.verified===true){
  const causal=active.progress_kind==='exploratory_causal_history_h2_feedback';
  const causalH5=active.progress_kind==='exploratory_causal_history_h5_feedback';
  const longH5=active.progress_kind==='exploratory_accelerated_long_h5_feedback';
  const title=active.terminal_review_verified?(causal?'10/10因果历史H2真实闭环完成；全部HOLD、配对收益为零':'10/10真实闭环已跑通；短窗口阻力上升0.23%，不是减阻达标'):longH5&&active.recovered_metrics?.verified?'124周期CFD完成；后处理已恢复，整体未减阻':longH5&&active.postprocessing_failed?'真实CFD加速H5：124个求解周期完成；统计后处理失败，等待恢复复核':(longH5?'真实CFD加速H5闭环（148→160.4） · ':causalH5?'真实CFD因果历史H5短时控制试验 · ':'真实CFD短时控制试验 · ')+(active.running?'正在计算':active.exited_success?'计算退出，结果和清理待独立复核':'已停止，需检查');
  const reviewedDetail=causal?' 独立复核：10次动作均为0，两支各200点真实CFD与首轮零控制字节一致；没有阻力或升力波动收益。':' 独立复核：每支200点，升力波动短窗口下降约5.69%；短窗口阻力上升0.23%。';
  const detail=`已完成 ${active.completed_cycles}/${active.planned_cycles??10} 个配对周期。${longH5?'GPU运行官方FNO推理选动作':'CPU运行官方FNO选动作'}，真实OpenFOAM求解；不是模型训练，也不是HydroGym求解器。`+(active.terminal_review_verified?reviewedDetail+' 仅1 D/U，不满足原80 D/U评价长度，没有新增PPO或长期达标结论。':longH5&&active.recovered_metrics?.verified?' 原执行 unit 在统计后处理阶段 exit1；离线恢复只读取已完成CFD受力文件，没有新CFD或模型执行。完整12.4 D/U平均阻力未降低，仍非准入、非PPO完成。':longH5?' 本轮计划覆盖12.4 D/U；运行中不声明控制收益、正式准入或PPO完成。':'');
  $('lead-now').textContent=title+'。'+detail;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',detail],['p',active.latest?`最近周期 ${active.latest.step}：后圆柱转速 ${num(active.latest.omega,3)}；预测/实际总阻力系数 ${num(active.latest.predicted_cd,5)} / ${num(active.latest.actual_cd,5)}；预测/实际后升力系数 ${num(active.latest.predicted_cl,5)} / ${num(active.latest.actual_cl,5)}；配对零控制总阻力/后升力 ${num(active.latest.zero_cd,5)} / ${num(active.latest.zero_cl,5)}。`:'等待首个真实周期。'],['p','这是探索性短时反馈，不是长期稳定控制或模型准入。P031资源失败和K1原正式未通过结论保留；R4两步接口已完成。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;$('train16-formal-detail').textContent=detail;return;
 }
 if(active?.bridge_completed===true&&active.verified===true){
  const title='真实CFD两步接口已完成（不是训练或控制完成）';
  const detail='两段零控制求解和当前流场输入已完成，148.0 → 148.1 → 148.2；没有加载模型或策略。机器计算：本任务已结束。当前工作：短时控制接口的软件集成与审查，尚未启动控制试验。';
  $('lead-now').textContent=title+'。'+detail;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',detail],['p','P031历史：两次资源测试退出，未完成训练；原失败证据与正式模型未通过的结论保留。下方图像仍是历史结果。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;$('train16-formal-detail').textContent=detail;return;
 }
 if(active){
  const age=Date.now()-Date.parse(active.sampled_at_utc||'');
  const fresh=active.verified===true&&Number.isFinite(age)&&age>=0&&age<60000;
  const reviewed=fresh&&active.exited_success&&active.review?.verified===true;
  const title=!fresh?'当前任务状态暂未核实':active.label+(active.running?' · 正在运行':reviewed?(active.review.diagnostic_completed?' · 诊断完成，已独立复核':active.review.engineering_pass?' · 工程复核通过，非精度验收':' · 已复核，未满足全部预测要求'):active.exited_success?' · 计算结束，结果待复核':' · 已停止，正在检查原因');
  const progress=fresh?Object.entries(active.planned_updates).map(([arm,total])=>arm+' 已完成 '+active.updates[arm]+'/'+total+' '+(active.progress_unit||'次更新')+(active.windows?'；训练窗口 '+active.windows[arm]+'/'+(total*8):'')).join('；')+(active.progress_detail?'。'+active.progress_detail:''):'不使用历史任务代替未知状态。';
  $('lead-now').textContent=title+'。'+progress;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',progress],['p',reviewed?active.review.summary:fresh?active.description:''],['p',reviewed?active.review.next_action:''],['p','计算完成不等于模型通过验收；尚无新的代理辅助CFD闭环结论。下方流场图是已标注的历史结果。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;$('train16-formal-detail').textContent=progress;return;
 }
 const p024=d.p024_live;
 if(p024?.verified===true){
  const age=Date.now()-Date.parse(p024.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const title=!fresh?'响应诊断状态采样已过期':p024.running?'P024 输入响应诊断进行中':p024.exited_success?'P024 诊断计算结束 · 结果待复核':'P024 诊断已停止 · 正在检查原因';
  const detail='不训练模型：先精确重现原模型与上次训练结果，再检查固定的几组输入系数。用于判断受力输入作用强弱和误差变化，不选择新模型或策略。';
  $('lead-now').textContent=title;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',detail],['p','P023已复核：部分预测误差略有改善，但仍未满足全部要求。原验收标准不变；流场图为历史结果。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;$('train16-formal-detail').textContent=detail;return;
 }
 const p023=d.p023_live;
 if(p023){
  const age=Date.now()-Date.parse(p023.sampled_at_utc||'');
  const fresh=p023.verified===true&&Number.isFinite(age)&&age>=0&&age<60000;
  const reviewed=fresh&&p023.exited_success&&p023.terminal_review?.verified===true;
  const title=!fresh?'当前训练状态暂未核实':p023.running?'P023 当前受力输入训练进行中':reviewed?'P023 已复核 · 预测略有改善但仍未满足全部要求':p023.exited_success?'P023 计算结束 · 精度结果待复核':'P023 任务已停止 · 正在检查原因';
  const progress=fresh?'低学习率组：'+p023.updates.LOW+'/16 次更新；较高学习率组：'+p023.updates.HIGH+'/16 次更新。':'当前进度未知，不用历史任务代替。';
  const detail=reviewed?'较高学习率组：连续预测的升力波动幅值绝对误差降低约0.041%，单步平均升力偏差平方增加约0.071%。关闭新增输入后，预测回到训练前结果，说明输入有作用，但改善幅度仍小。':'原模型权重固定，只训练新增的96个受力输入系数。两组都使用当前受力；每次更新包含六个真实CFD窗口，每窗连续预测100步。';
  const note=reviewed?'下一项输入响应诊断正在准备，尚未启动GPU计算；先重现已有结果，再检查不同输入系数幅度下的误差变化。没有新的PPO或代理辅助CFD闭环结果。下方流场图是历史结果。':'检查平均升力、波动幅值及波形误差。P022已复核但未满足全部精度改进要求；尚无新的PPO或代理辅助CFD闭环结果。下方流场图仍是标注的历史结果。';
  $('lead-now').textContent=title+'。'+progress;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',progress],['p',detail],['p',note]]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=progress+' '+detail;return;
 }
 const p022=d.p022_live;
 if(p022){
  const age=Date.now()-Date.parse(p022.sampled_at_utc||'');
  const fresh=p022.verified===true&&Number.isFinite(age)&&age>=0&&age<60000;
  const title=!fresh?'P022 状态暂未核实':p022.running?'P022 当前受力输入训练对照进行中':p022.exited_success?'P022 对照计算结束 · 精度结果待复核':'P022 任务已停止 · 正在检查原因';
  const progress=fresh?'不输入当前受力：'+p022.updates.A_zero+'/16 次更新；输入当前受力：'+p022.updates.B_causal+'/16 次更新。':'当前进度未知，不用旧实验代替。';
  const detail='两组使用同一初始模型、六个真实CFD窗口和相同训练目标。每次更新包含六个窗口，每个窗口预测100步；连续预测只在起点读取真实受力，之后使用自身预测。';
  const note='比较平均升力、波动幅值和波形误差。尚无新的PPO或CFD闭环验收结果；下方流场图是已标注的历史模型结果。';
  $('lead-now').textContent=title+'。'+progress;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',progress],['p',detail],['p',note]]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=progress+' '+detail;return;
 }
 const p020=d.p020_live;
 if(p020?.verified===true){
  const age=Date.now()-Date.parse(p020.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const rejected=p020.terminal_result?.verified===true&&!p020.running;
  const title=!fresh?'训练对照状态采样已过期':p020.running?'P020 升力预测训练对照进行中':rejected?'P020 对照已复核 · 未同时改善全部预测指标':p020.exited_success?'P020 训练对照已结束 · 结果待独立复核':'P020 训练对照未运行 · 正在检查原因';
  const progress='原损失函数：'+p020.updates.A_original+'/16 次更新；增加升力均值与波动监督：'+p020.updates.B_symmetric_tail+'/16 次更新。';
  const detail=rejected?'新增统计监督相对原损失组更好，但相对训练前，单步平均升力平方误差增加1.37%，去均值后的波形平方误差增加10.53%。不进入全量训练；当前核查加入当前受力输入的数据时序与连续预测方法。':'两组使用同一初始模型和六个真实CFD训练窗口；比较升力均值、波动幅值及连续预测误差，不改变控制验收标准。';
  const note='这是训练诊断，不是新的PPO或CFD闭环成功。下方流场图仍是标注的历史结果。';
  $('lead-now').textContent=title+'。'+progress;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',progress],['p',detail],['p',note]]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=progress+' '+detail;return;
 }
 const formal018=d.p018_formal_live;
 if(formal018?.observed===true){
  const age=Date.now()-Date.parse(formal018.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const task=formal018.task;
  const result=d.p018_formal_result;
  const failed=result?.verified===true;
  const title=!fresh?'评估状态采样已过期':task.running?'P018 训练结束 · 完整预测精度评估中':failed?'P018 完整评估未达标 · 升力波动预测需改进':'P018 评估进程未运行 · 结果待复核';
  const names={validation10:'十条验证轨迹',dynamic6:'六条动态动作轨迹',force_window:'受力时间窗口'};
  const done=(task.progress?.completed_steps||[]).map(x=>names[x]||x);
  const detail=failed&&!task.running?'六个工况通过数量：阻力 '+result.cd_pass+'/6；平均升力 '+result.mean_pass+'/6；升力波动 '+result.rms_pass+'/6；全部指标同时通过 '+result.joint_pass+'/6。':'已完成的评估阶段：'+(done.length?done.join('、'):'尚无完整阶段结果')+'。检查单步及连续预测的流场、阻力、升力均值与波动误差。';
  const note=failed?'当前工作：梯度诊断已完成，训练目标改进仍待有限步数对照验证。没有新PPO或代理辅助CFD闭环成功。下方图片保留历史模型标注。':'本轮精度与控制效果尚未验收，未启动新策略训练。下方流场图片保留历史模型标注，不代表本轮结果。';
  $('lead-now').textContent=title+'。'+detail;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',detail],['p',note],['p',task.identity_issues?.length?'运行核查提示：'+task.identity_issues.join('；'):'按实际服务、进程和资源采样更新；评估日志暂时无输出不代表停止。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=detail+' '+note;return;
 }
 const reduced=d.reduced_rate_training;
 if(reduced?.verified===true){
  const age=Date.now()-Date.parse(reduced.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const title=!fresh?'当前采样已过期':reduced.running?'全量训练进行中 · 降低学习率对照':'训练进程已结束或异常 · 终态与精度待核';
  const progress='参数更新 '+reduced.updates+' / 171；训练窗口 '+reduced.consumed_windows+' / 1368。';
  const detail='使用相同44条真实CFD训练轨迹和官方FNO；仅将学习率降为原来的1/64（1.5625×10⁻⁷）。每8个窗口合并梯度，再更新一次参数。';
  const pending='当前尚无本轮精度验收、强化学习或真实CFD闭环结果。下方流场图片是已标注的历史结果。';
  $('lead-now').textContent=title+'。'+progress;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',progress],['p',detail],['p',pending],['p',reduced.running&&!reduced.progress_fresh?'日志超过五分钟未更新，需核查进度；进程仍在运行。':'依据实际运行进程和日志更新，不以GPU占用率判断科学成功。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=progress+' '+pending;return;
 }
 const probe=d.fixed_panel_probe;
 if(probe?.verified===true){
  const age=Date.now()-Date.parse(probe.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const title=!fresh?'当前采样已过期':probe.terminal?'固定六窗口诊断已结束 · 不代表控制准入':probe.running?'固定六窗口局部拟合诊断进行中':'诊断未运行 · 终态待核验';
  const progress=probe.terminal?'完成 32 / 32 次更新；结果仍须独立复核。':probe.minimum_completed_updates==null?'尚无可信更新记录，不估算进度。':'日志确认至少完成 '+probe.minimum_completed_updates+' / 32 次更新；这是检查面板记录，不推算当前正在更新的序号。';
  const detail='保持原官方神经算子、数据和损失函数，重复使用六个训练窗口，检验均值与波动误差能否同时改善。不保存候选模型，不读取验证或冻结测试，不启动强化学习策略。';
  const prior=d.p015_formal_result;
  const history=prior?.verified?'上一轮八窗口累积训练（P015）正式评估失败：联合通过 '+prior.joint_pass+'/6；阻力 '+prior.cd_pass+'/6、升力波动 '+prior.rms_pass+'/6、平均升力 '+prior.mean_pass+'/6。':'上一轮正式评估证据暂未核验，不能推断通过。';
  $('lead-now').textContent=title+'。'+progress;
  const card=document.createElement('div');card.className='card';
  for(const [tag,text] of [['h3',title],['p',progress],['p',detail],['p',history],['p',probe.running&&!probe.progress_fresh?'进程仍在运行，但日志超过五分钟未更新，请检查进度。下方图片均为历史模型结果。':'下方图片均为已标注的历史模型结果，不是本次诊断产生的新结果。']]){const el=document.createElement(tag);el.textContent=text;card.appendChild(el);}
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=progress+' '+history;
  return;
 }
 const grouped=d.window_accumulation_training;
 if(grouped?.verified===true){
  const age=Date.now()-Date.parse(grouped.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const evaluation=grouped.evaluation;
  const title=!fresh?'当前采样已过期':d.p015_formal_result?.verified?'八窗口累积训练正式评估失败 · 联合通过 1/6':evaluation?.verified?(evaluation.running?'训练已完成 · 完整精度评估中':'精度评估进程已结束 · 结论待核'):grouped.running?'八窗口梯度累积 · 训练进行中':'八窗口训练进程未运行 · 结果待核';
  $('lead-now').textContent=title+'。检查流场、阻力与升力预测是否满足控制要求；尚无本轮闭环减阻结论。';
  const card=document.createElement('div');card.className='card';
  const heading=document.createElement('h3');heading.textContent=title;card.appendChild(heading);
  const progress=document.createElement('p');progress.textContent='已处理 '+grouped.consumed_windows+' / 1368 个流动窗口；完成 '+grouped.updates+' / 171 次参数更新。每八个窗口的梯度取平均，再更新模型一次。';card.appendChild(progress);
  const detail=document.createElement('p');detail.textContent='官方 FNO、真实 CFD 数据和损失函数不变，原流场预测模型保持冻结。固定六个训练窗口在训练前、处理中和结束时重复检查；不挑选中间最好模型。';card.appendChild(detail);
  const note=document.createElement('p');note.textContent=evaluation?.verified?'模型保存及官方双模型重载已核验。评估范围包括 1、10、50、100 步预测、动态动作轨迹及六个受力时间窗口；全部通过后才训练兼容的新 PPO，再用真实 CFD 验证。下方流场图片仍为已标注的历史结果。':'训练日志距今 '+num(grouped.log_age_seconds,0)+' 秒。训练完成后仍须完整精度评估，合格后才训练新 PPO 并开展真实 CFD 闭环。下方流场图片仍为已标注的历史结果。';card.appendChild(note);
  $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=progress.textContent+' 上一模型正式受力窗口联合通过 0/6，本轮不得以训练损失代替验收。';
  return;
 }
 const independent=d.independent_force_training;
 if(independent?.ready===true){
  const age=Date.now()-Date.parse(independent.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const terminal=independent.terminal;
  if(terminal?.verified===true){
   const title=!fresh?'当前状态采样已过期':terminal.formal.running?'训练完成 · 正式预测精度评估进行中':'训练完成 · 正式评估结果待核';
   $('lead-now').textContent=title+'。训练窗口诊断未显示受力预测改善；尚未开始新模型 PPO 或真实 CFD 闭环验收。';
   const card=document.createElement('div');card.className='card';
   const heading=document.createElement('h3');heading.textContent=title;card.appendChild(heading);
   const summary=document.createElement('p');summary.className='bad';summary.textContent='1368 次训练更新及模型完整性核验完成。后圆柱升力平均绝对误差：单步在 '+terminal.h1_regressions+'/6 个窗口变大，多步在 '+terminal.ar_regressions+'/6 个窗口变大。流场预测指标'+(terminal.fields_unchanged?'保持一致':'需复核')+'。';card.appendChild(summary);
   const table=document.createElement('table');
   table.innerHTML='<tr><th>真实 CFD 训练窗口</th><th>单步 Cl 误差：原 → 新</th><th>连续 100 步 Cl 误差：原 → 新</th></tr>'+terminal.rows.map((r,i)=>'<tr><td>'+['零转速 · b00','变化转速 · b00','变化转速 · b02','变化转速 · b04','变化转速 · b06','历史控制动作 · b00'][i]+'</td><td>'+num(r.h1_parent,4)+' → '+num(r.h1_candidate,4)+'</td><td>'+num(r.ar_parent,4)+' → '+num(r.ar_candidate,4)+'</td></tr>').join('');card.appendChild(table);
   const note=document.createElement('p');note.textContent='误差越小越好。单步每次输入真实 CFD 流场；连续预测使用模型上一步的流场。这里是六个固定训练窗口的诊断，不是验证集结论、减阻率或闭环成功。正式评估仍使用原指标；下方流场图片属于已标注的历史模型。';card.appendChild(note);
   $('lead-models').prepend(card);$('train16-formal-progress').textContent=title;
   $('train16-formal-detail').textContent='阶段：训练完整性已核验 → 训练窗口诊断已完成（存在退步）→ 正式精度评估 → 合格后训练新 PPO → 真实 CFD 在线反馈。最终减阻与升力约束尚未验收。';
   return;
  }
  const title=!fresh?'训练状态采样已过期':independent.running?'独立气动力 FNO · 训练进行中':independent.service_state==='unknown'?'训练状态暂时读取失败':'独立气动力 FNO · 训练进程未运行，结果待核';
  $('lead-now').textContent=title+'。原流场FNO保持不变，另一个官方FNO学习阻力和升力；尚未通过独立精度评估。';
  const card=document.createElement('div');card.className='card';
  const h=document.createElement('h3');h.textContent=title;card.appendChild(h);
  const progress=document.createElement('p');progress.textContent='恢复后第2次执行 · 已记录更新：'+(independent.step==null?'尚无有效记录':independent.step+' / 1368')+'；'+(independent.progress_fresh?'最近更新正常':independent.running?'等待首批记录或检查进度':'不把旧日志当作正在训练');card.appendChild(progress);
  const losses=document.createElement('p');losses.textContent='最新训练批次损失：真实当前流场输入 '+(independent.h1_balanced==null?'—':num(independent.h1_balanced,6))+'；连续预测流场输入 '+(independent.ar_balanced==null?'—':num(independent.ar_balanced,6))+'；等权合计 '+(independent.total==null?'—':num(independent.total,6));card.appendChild(losses);
  const note=document.createElement('p');note.textContent='每次更新使用一个真实CFD的100步窗口，两类输入各占一半。这里显示的是归一化训练误差，不是验证精度或减阻率。训练结束后仍需原定流场、阻力和升力波动评估，不能自动进入PPO。';card.appendChild(note);
  $('lead-models').prepend(card);
  $('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent='本阶段共1368次更新，固定终态模型、不挑选验证集最优轮次。资源要求为至少20 GiB可用统一内存；下方流场图片仍属于明确标注的历史模型。';
  return;
 }
 const paired=d.decoder_scope_training;
 if(paired?.ready===true){
  const age=Date.now()-Date.parse(paired.sampled_at_utc||'');
  const fresh=Number.isFinite(age)&&age>=0&&age<60000;
  const completed=paired.arms.every(a=>a.terminal?.verified===true);
  const rejected=paired.arms.every(a=>a.terminal?.formal_result?.verified_fail===true);
  const title=rejected?'两组独立评估均未通过 · 升力波动预测需改进':completed?'两组训练完成 · 独立精度评估':'FNO 升力预测训练 · 两组同条件对照';
  $('lead-now').textContent=title+'。A只训练升力输出；B同时训练现有解码器最后一层。尚未开始新模型PPO或闭环验收。';
  const card=document.createElement('div');card.className='card';
  const h=document.createElement('h3');h.textContent=title;card.appendChild(h);
  const table=document.createElement('table');
  table.innerHTML='<tr><th>节点 / 方法</th><th>更新次数</th><th>最新批次损失</th><th>实际状态</th></tr>';
  paired.arms.forEach(a=>{
   const tr=document.createElement('tr');
   const state=!fresh?'页面采样已过期':a.terminal?.formal_result?.verified_fail?'独立评估未通过；不可进入代理PPO':a.terminal?.verified?(a.formal?.running?'训练已核验；独立评估运行中':a.formal?.service_state==='unknown'?'训练已核验；评估状态读取失败':'训练已核验；评估未运行，结果待复核'):a.running?(a.progress_fresh?'训练运行中':'进程运行，进度待检查'):a.service_state==='unknown'?'状态读取失败':'训练进程未运行，终态待核';
   [a.label,a.terminal?.verified?'1368 / 1368':a.step==null?'尚无已核步数':a.step+' / 1368',a.total_loss==null?'—':num(a.total_loss,6),state].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.appendChild(td)});
   table.appendChild(tr);
  });card.appendChild(table);
  const note=document.createElement('p');note.textContent='每次更新使用一个真实CFD的100步训练窗口。最新批次损失不是验证精度，不能跨不同批次直接排名。独立评估检查速度、压力、阻力和升力波动；训练完成不是模型合格。';card.appendChild(note);
  if(completed){
   const diag=document.createElement('p');
   diag.textContent='固定6个训练窗口：连续预测的升力波动误差（训练前 → 训练后） '+paired.arms.map(a=>a.label+'：'+num(a.terminal.rms_before,5)+' → '+num(a.terminal.rms_after,5)).join('；')+'。越小越好；这不是验证集成绩，也不是减阻率。';card.appendChild(diag);
  }
  if(rejected){
   const summary=document.createElement('p');summary.textContent='独立受力窗口评估（各6个工况）：'+paired.arms.map(a=>{const f=a.terminal.formal_result;return a.label+'：阻力 '+f.cd_pass+'/6、升力波动 '+f.rms_pass+'/6、平均升力 '+f.mean_pass+'/6'}).join('；')+'。两组都只有零转速的两个工况全部通过；旋转工况的升力波动误差仍过大。';card.appendChild(summary);
   if(d.gradient_diagnostic?.verified){
    const p=document.createElement('p');p.textContent='后续诊断已完成：两模型各检查6个真实训练窗口。5个旋转窗口中，均未触发事先规定的强梯度冲突或尺度失衡条件。不支持继续盲调损失权重；下一步评估分离流场与气动力预测的方案，尚未启动新训练或PPO。';card.appendChild(p);
    $('lead-now').textContent='训练目标梯度诊断完成；未发现预设的强冲突信号。正在设计气动力预测改进，完整FNO辅助闭环尚未完成。';
   }
  }
  $('lead-models').prepend(card);
  $('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent='完整训练各1368次更新，数据、顺序和损失相同。资源要求：每节点可用统一内存至少20 GiB，实际采样见资源面板。现有流场图属于已标注的历史模型，并非本轮模型的新结果。';
  return;
 }
 const joint=d.joint_readout_diagnostic;
 if(joint?.ready===true){
  const formal=joint.formal||{}, age=Date.now()-Date.parse(formal.sampled_at_utc||'');
  const running=Number.isFinite(age)&&age>=0&&age<60000&&formal.service_state==='active';
  const stage={validation10:'10条验证轨迹',dynamic6:'动态转速轨迹',force_window:'阻力与升力时间窗口'}[formal.stage]||'验证数据';
  const title=formal.verified_fail?'正式评估未达标 · 后圆柱升力波动预测需改进':running?'正式模型评估进行中 · '+stage:formal.complete_recorded?'正式评估记录已完成 · 结论待复核':'受力联合拟合已完成 · 当前未确认正式评估运行';
  $('lead-now').textContent=title+'。单步与连续预测受力误差的训练内改善，不代表独立验证通过；新模型 PPO 和真实 CFD 闭环验收仍未完成。';
  const card=document.createElement('div');card.className='card';
  const h=document.createElement('h3');h.textContent=title;card.appendChild(h);
  const table=document.createElement('div');table.innerHTML='<table><tr><th>预测范围 / 系数平均绝对误差</th><th>原模型</th><th>联合拟合</th></tr>'+joint.rows.map(r=>`<tr><td>${r.label}</td><td>${num(r.before,5)}</td><td>${num(r.after,5)}</td></tr>`).join('')+'</table>';card.appendChild(table);
  const note=document.createElement('p');note.textContent='训练相位留出诊断，数值越低越好；不是减阻比例，也不是最终测试成绩。使用同一批真实CFD目标，单步与连续预测特征各占一半，只拟合既有受力输出。原流场预测保持不变。';card.appendChild(note);
  const risk=document.createElement('p');risk.className='bad';risk.textContent=formal.verified_fail?'正式结果：6个时间窗口中，总阻力均值6/6通过，后柱平均升力5/6通过，升力波动幅度仅2/6通过，4个旋转工况均未通过。下一步分析训练窗口与未见相位的波动幅度误差；暂不启动新模型PPO。':'仍有不足：第100步总阻力误差仅小幅改善，且一个初始相位有所退化。必须完成原定独立验证，不能据此启动PPO。';card.appendChild(risk);
  $('lead-models').prepend(card);
  $('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent='候选已保存并通过加载与输出一致性检查。'+(formal.verified_fail?'正式结果已独立复核，整体未达标；后续重点是时间窗口误差诊断。是否正在计算请看实时资源及任务状态。':running?'正在评估：'+stage+'；该评估器不逐批输出百分比，实际负载见资源采样。':formal.complete_recorded?'完整结果记录已生成，仍需独立复核科学指标。':'当前服务状态：'+(formal.service_state||'未知')+'，不据历史文件推断正在计算。')+' 下表是此前的训练相位留出诊断，不是当前验证成绩。';
  return;
 }
 const ar=d.free_ar_diagnostic;
 if(ar?.ready===true){
  const age=Date.now()-Date.parse(ar.sampled_at_utc);
  const live=Number.isFinite(age)&&age>=0&&age<60000&&ar.service_state==='active';
  const title=live?'正在计算 · 连续预测受力误差对照':'连续预测受力误差对照 · 当前未确认运行';
  const detail=`已记录 ${ar.windows} / 1368 个预测窗口（${ar.rows} / 136800 条预测记录）。每个窗口连续预测100步；这些窗口复用已有44条CFD轨迹，不是新增独立仿真。`;
  $('lead-now').textContent=title+'。'+detail;
  const card=document.createElement('div');card.className='card';
  const h=document.createElement('h3');h.textContent=title;card.appendChild(h);
  const p=document.createElement('p');p.textContent=detail;card.appendChild(p);
  const note=document.createElement('p');note.textContent='目的：比较真实流场输入与模型连续预测输入下的阻力、升力误差，判断受力输出层是否需要针对连续预测重新拟合。本轮不更新流场模型，不训练PPO；下方保留既有真实CFD与FNO流场对照。';card.appendChild(note);
  $('lead-models').prepend(card);
  $('train16-formal-progress').textContent=title;
  $('train16-formal-detail').textContent=detail+' 最近进度距今 '+num(ar.progress_age_seconds,0)+' 秒。数值有限：'+(ar.finite===true?'是':'未确认')+'。实验完成不代表闭环验收通过。';
  return;
 }
 const fc=d.full_train_calibration;
 if(fc?.ready===true){
  const liveAge=Date.now()-Date.parse(fc.sampled_at_utc);
  const fresh=Number.isFinite(liveAge)&&liveAge>=0&&liveAge<60000;
  const running=fresh&&fc.service_state==='active';
  const stages={validation10:'独立验证轨迹：1 / 10 / 50 / 100 步预测',dynamic6:'动态旋转动作：多步预测',force_window:'时间窗口内的阻力与升力统计'};
  const title=running?'当前任务 · '+(stages[fc.stage]||'正式模型评估'):'全量受力校准完成 · '+(fc.formal_complete?'正式评估记录已完成，待科学结论复核':'正式评估未确认运行');
  $('lead-now').textContent=title+'。尚未完成新模型 PPO 与真实 CFD 闭环验收。';
  const card=document.createElement('div');card.className='card';
  const h=document.createElement('h3');h.textContent=title;card.appendChild(h);
  const t=document.createElement('div');t.innerHTML='<table><tr><th>训练集平均绝对误差</th><th>原模型</th><th>校准后</th></tr>'+fc.rows.map(r=>`<tr><td>${r.channel==='rear_cd'?'后圆柱阻力系数':'后圆柱升力系数'}</td><td>${num(r.before,4)}</td><td>${num(r.after,4)}</td></tr>`).join('')+'</table>';card.appendChild(t);
  const p=document.createElement('p');p.textContent='44 条真实 CFD 训练轨迹、19,648 个时间步；只校准力输出层，原流场输出不变。表中是训练集误差，不是减阻比例，也不是独立验证成绩。下方流场图仍为原 FNO 的预测。';card.appendChild(p);
  if(fc.validation?.status==='FULL40_VALIDATION_SURROGATE_READINESS_FAIL'){
   const warning=document.createElement('p');warning.className='bad';
   warning.textContent=`独立验证尚未通过：旋转相对无旋转的阻力变化预测误差 ${num(fc.validation.delta_cd_mae,4)}，既定上限 ${num(fc.validation.maximum,4)}。后续动态验证仍需完成，当前不可进入 PPO。`;card.appendChild(warning);
  }
  $('lead-models').prepend(card);
  $('train16-formal-progress').textContent=title;
  $('train16-formal-progress').className='number';
  $('train16-formal-detail').textContent='实际服务状态：'+(fresh?fc.service_state:'采样过期')+'。模型校准已完成，正式评估通过后才能开展新策略训练。';
  return;
 }
 const w=d.training_evaluation_watchdog||{}, units=w.active_units||[];
 const age=Date.now()-Date.parse(w.timestamp_utc||'');
 if(!Number.isFinite(age)||age<0||age>180000){$('lead-now').textContent='任务状态已过期或时间异常，不能确认当前训练或评估是否运行。';return;}
 const fr=d.fixed_feature_readout;
 if(fr?.ready===true&&units.length===0){
  const title='最新受力诊断 · 前段可拟合，后段仍不可靠';
  const detail='固定原有 FNO 特征，只检查最后一层受力映射。前100步用于拟合，后100步未用于此次拟合，但两段都来自已有训练轨迹；不是独立测试。';
  $('lead-now').textContent=title+'。下一项是训练数据内的拟合稳定性检查，不是 PPO。当前是否有计算任务，以资源与进程采样为准；闭环目标尚未完成。';
  const card=document.createElement('div');card.className='card';
  const heading=document.createElement('h3');heading.textContent=title;card.appendChild(heading);
  const note=document.createElement('p');note.textContent=detail;card.appendChild(note);
  const table=document.createElement('div');
  table.innerHTML='<table><tr><th>数据时段</th><th>后柱阻力 MAE：原映射 → 拟合</th><th>后柱升力 MAE：原映射 → 拟合</th></tr>'+fr.rows.map(r=>`<tr><td>${r.panel==='prefix_targets_1_100'?'前100步（拟合）':'后100步（检查）'}</td><td>${num(r.rear_cd_before,4)} → ${num(r.rear_cd_after,4)}</td><td>${num(r.rear_cl_before,4)} → ${num(r.rear_cl_after,4)}</td></tr>`).join('')+'</table>';card.appendChild(table);
  const caution=document.createElement('p');caution.textContent='MAE 是力系数的平均绝对误差，越小越好，不是百分比。上表仅旋转工况；后段阻力反而变差，因此不能据前段改善开始控制训练。两方使用同一最高 FP32 精度；下方流场图仍属于原模型，本诊断没有生成新模型。';card.appendChild(caution);
  $('lead-models').prepend(card);
  $('train16-formal-progress').textContent='受力拟合诊断完成 · 尚不可用于闭环';
  $('train16-formal-progress').className='number';
  $('train16-formal-detail').textContent=detail+' 没有参数更新、模型保存或 PPO 训练。';
  return;
 }
 const training=units.includes('fluid-control-fcp003c-true-state-step-20261005.service');
 const evaluating=units.includes('fluid-control-fcp003c-posteval-wait-fa08ce0-20261005.service');
 const absoluteRunning=units.includes('fluid-control-fcp003c-train-fit-absolute-calibration-20261005.service');
 const calibrating=absoluteRunning||units.includes('fluid-control-fcp003c-train-fit-calibration-20261005.service');
 const cp=w.progress?.train_fit_calibration||{};
 const absoluteCalibration=absoluteRunning||cp.paired_force_objective==='absolute';
 const calibrationDone=cp.completion_verified===true&&cp.state==='EXECUTION_COMPLETE_NOT_ADMISSION';
 if(!training&&!evaluating&&!calibrating&&!calibrationDone)return;
 const p=d.fc_p003c_training_log||{};
 const progress=Number.isFinite(p.batch_percent)?`最近批次记录为该轮的 ${p.batch_percent.toFixed(2)}%，记录时间 ${p.logged_at_utc}；不是整个项目完成比例。`:'当前尚无批次进度记录，不估算百分比。';
 const validSteps=Number.isInteger(cp.completed_steps)&&cp.completed_steps>=0&&cp.completed_steps<=128&&cp.total_steps===128&&Number.isInteger(cp.paired_steps_completed)&&cp.paired_steps_completed>=0&&cp.paired_steps_completed<=64;
 const calibrationDetail=(validSteps?`最近记录：已完成 ${cp.completed_steps} / 128 次参数更新，其中 ${cp.paired_steps_completed} / 64 次包含旋转配对监督。`:'正在加载模型或计算训练前对照，尚无有效更新步数。')+(absoluteCalibration?' 保持流场训练和预算不变，分别监督旋转与不旋转时的真实受力；':' 保持原有流场训练，增加旋转响应训练占比；')+'不读取验证集，不据训练 loss 判断成功，完成后比较同一批流场与受力误差。';
 const title=calibrating?(absoluteCalibration?'当前训练 · 绝对受力监督对照':'当前训练 · 旋转响应小规模校准'):calibrationDone?(absoluteCalibration?'绝对受力监督对照已完成 · 尚未通过模型验收':'小规模校准已完成 · 尚未通过模型验收'):training?'当前训练 · 动作响应监督（FC-P003C）':'当前任务 · FC-P003C 评估队列';
 const detail=calibrating?calibrationDetail:calibrationDone?'128 次更新和 64 次旋转配对监督已完成，训练进程已结束。仍需依据训练前后流场与受力误差判断效果，不能据运行成功认定闭环目标达成。':training?`已记录 ${p.completed_epochs??'待核实'} / 2 个完整训练轮次。${progress} 保持现有模型和数据，调整动作引起的受力差训练项；训练后检查流场、阻力和升力预测。`:'训练服务当前不在运行；评估队列正在运行或等待模型完成记录，不能仅凭队列存活认定 GPU 正在评估。';
 $('lead-now').textContent=title+'。'+detail+' 尚未完成新模型的 PPO 与真实 CFD 闭环验收。';
 const card=document.createElement('div');card.className='card';
 const heading=document.createElement('h3');heading.textContent=title;card.appendChild(heading);
 const note=document.createElement('p');note.textContent=detail+' 下方为已完成候选的结果，不是本轮训练精度。';card.appendChild(note);
 $('lead-models').prepend(card);
 $('train16-formal-progress').textContent=title;
 $('train16-formal-progress').className='number';
 $('train16-formal-detail').textContent=detail+((calibrating||calibrationDone)?(absoluteCalibration?' 本轮目录：fcp003c_train_fit_absolute_calibration_20261005。':' 本轮目录：fcp003c_train_fit_calibration_20261005。'):' 本轮目录：tandem_fno_true_state_paired_step_lambda10_20261005。');
}
function pct(x){return Number.isFinite(x)?(x*100).toFixed(2)+'%':'—'}
function num(x,d=1){return Number.isFinite(x)?x.toFixed(d):'—'}
function passText(value){return value===true?'PASS':value===false?'FAIL':'等待'}
function dragEffect(value){return Number.isFinite(value)?(value>=0?`降阻 ${pct(value)}`:`增阻 ${pct(-value)}`):'—'}
function renderLatestEvidence(d){
 let decision=d.v4_validation_decision,terminal=decision?.candidate_terminal_100step,parentTerminal=decision?.v3_parent_terminal_100step;
 let terminalValue=terminal?.pooled_total_drag_nrmse,terminalReady=decision?.split==='validation'&&Number.isFinite(terminalValue),terminalPass=terminalReady?decision.candidate_meets_fixed_threshold===true:null;
 let workerSingle=d.resources?.worker?.at(-1)?.v4_single_validation_pooled_nrmse,singleNote=Number.isFinite(workerSingle)?`计算节点v4单步FNO参考为 ${pct(workerSingle)}（不是H20）。`:'';
 $('v4-terminal').textContent=terminalReady?pct(terminalValue):'等待 H20 pooled';$('v4-terminal').className='number '+(terminalPass===true?'good':terminalPass===false?'bad':'');
 $('v4-terminal-detail').textContent=terminalReady?`v4 H20验证集4条真实CFD；FNO第100步终点pooled NRMSE；macro ${pct(terminal.macro_total_drag_nrmse)}；v3 parent pooled ${pct(parentTerminal?.pooled_total_drag_nrmse)}，二者均未过10%，candidate ${decision.candidate_minus_parent_pooled_100step_nrmse<0?'改善':'未改善'}。${singleNote} 冻结测试未访问。`:`v4 H20 formal decision.json尚未生成；13.8187% macro只是初值，不能替代pooled判定。${singleNote} 冻结测试未访问。`;
 let window=d.v4_window_mean_cd,parentWindow=d.v3_parent_window_mean_cd,windowValue=window?.pooled_window_mean_cd_nrmse,parentWindowValue=parentWindow?.pooled_window_mean_cd_nrmse,windowReady=window?.split==='validation'&&Number.isFinite(windowValue),parentWindowReady=parentWindow?.split==='validation'&&Number.isFinite(parentWindowValue),windowPass=windowReady?(window.status==='PASS_VALIDATION_WINDOW_MEAN_CD'):null,windowGain=windowReady&&parentWindowReady?windowValue<parentWindowValue:null,windowDeltaPp=windowReady&&parentWindowReady?100*(windowValue-parentWindowValue):null;
 $('v4-window').textContent=windowReady?`${pct(windowValue)}${parentWindowReady?` · ${windowGain?'GAIN':'NO GAIN'}`:''}`:'等待正式报告';$('v4-window').className='number '+(windowPass===false||windowGain===false?'bad':windowPass===true&&windowGain===true?'good':'');
 $('v4-window-detail').textContent=windowReady?`v4 H20固定10%阈值 ${passText(windowPass)}；验证集真实CFD对FNO，步骤1–100总Cd算术均值。${parentWindowReady?`v3 parent ${pct(parentWindowValue)}，candidate ${windowGain?'改善':'退化'} ${num(Math.abs(windowDeltaPp),3)}个百分点；`: 'v3 parent对照尚未生成；'}macro ${pct(window.macro_window_mean_cd_nrmse)}，最差 ${pct(window.worst_case_window_mean_cd_nrmse)}。不是终点指标或控制收益。`:'窗口均值JSON尚未生成；与第100步终点指标分开展示。';
 let physical=d.two_phase_alternating,phases=physical?.phases||{},t90=phases.t90?.comparison,t94=phases.t94?.comparison,physicalReady=physical?.status==='TWO_PHASE_ALTERNATING_OPENFOAM_AUDIT_COMPLETED'&&t90&&t94,physicalPass=physicalReady?physical.robustness?.canonical_joint_pass_both_phases===true:null;
 let low=d.low_action_phase94_canonical,lowComparisons=low?.comparisons_to_same_phase_zero||{},lowMinus=Object.entries(lowComparisons).find(([name])=>name.includes('_m_phase94_'))?.[1],lowPlus=Object.entries(lowComparisons).find(([name])=>name.includes('_p_phase94_'))?.[1],lowReady=low?.status==='LOW_ACTION_PHASE94_CANONICAL_PHYSICAL_AUDIT_V3_COMPLETE'&&lowMinus&&lowPlus,lowNote=lowReady?` 低幅脉冲v3 canonical：−0.75降阻 ${num(-lowMinus.mean_total_cd_change_percent,3)}%<2%，Cl′比 ${num(lowMinus.rear_cl_fluctuation_rms_ratio,3)}≤1.05、平均Cl比 ${num(lowMinus.abs_mean_rear_cl_to_zero_cl_fluctuation_rms_ratio,3)}>0.1，joint ${lowMinus.canonical_joint_gate}；+0.75增阻 ${num(lowPlus.mean_total_cd_change_percent,3)}%，joint ${lowPlus.canonical_joint_gate}。真实OpenFOAM，仅t94单相位开环。`:'';
 let dwell=d.long_dwell075,dwellPhases=dwell?.phases||{},d90=dwellPhases.t90?.comparison,d94=dwellPhases.t94?.comparison,dwellReady=dwell?.status==='TWO_PHASE_LONG_DWELL075_OPENFOAM_AUDIT_COMPLETED'&&d90&&d94,dwellPass=dwellReady?dwell.robustness?.canonical_joint_pass_both_phases===true:null,dwellNote=dwellReady?` 低幅长驻留±0.75（T=40）真实OpenFOAM：t90/t94降阻 ${pct(d90.total_drag_reduction)} / ${pct(d94.total_drag_reduction)}，均<2%；Cl′比 ${num(d90.rear_cl_fluctuation_rms_ratio,3)} / ${num(d94.rear_cl_fluctuation_rms_ratio,3)}，两相位joint ${passText(dwellPass)}。仅预声明开环物理筛查，不是代理或闭环达标。`:'';
 $('two-phase').textContent=physicalReady?`t90 ${dragEffect(t90.total_drag_reduction)} · t94 ${dragEffect(t94.total_drag_reduction)}`:'等待真实 CFD';$('two-phase').className='number '+(physicalPass===true?'good':physicalPass===false?'bad':'');
 $('two-phase-detail').textContent=physicalReady?`真实OpenFOAM CFD长窗口、各自相位匹配零控制；两相位联合 ${passText(physicalPass)}；后柱Cl′ RMS比 ${num(t90.rear_cl_fluctuation_rms_ratio,2)} / ${num(t94.rear_cl_fluctuation_rms_ratio,2)}。不是代理预测或闭环结果。${dwellNote}${lowNote}`:'两相位OpenFOAM result.json尚未生成；不以代理结果代替。';
 let ready=[terminalReady,windowReady,Boolean(physicalReady)],passes=[terminalPass,windowPass,physicalPass],readyCount=ready.filter(Boolean).length,passCount=passes.filter(x=>x===true).length,jointReady=readyCount===3,jointPass=jointReady&&passCount===3;
 $('joint-status').textContent=jointReady?(jointPass?'PASS':'FAIL'):`等待 ${readyCount}/3`;$('joint-status').className='number '+(jointReady?(jointPass?'good':'bad'):'');
 $('joint-detail').textContent=`终点精度 ${passText(terminalPass)} · 窗口固定阈值 ${passText(windowPass)}${parentWindowReady?`、相对parent ${windowGain?'GAIN':'NO GAIN'}`:''} · 两相位真实CFD联合收益 ${passText(physicalPass)}。这是展示层联合状态；v4冻结测试未访问。`;
 $('decision').textContent=jointReady?`历史 v4（非当前 full40/dev30）：三项联合 ${jointPass?'PASS':'FAIL'}。终点/窗口是v4 H20验证集上的代理预测误差${parentWindowReady?`；窗口虽过固定阈值但相对v3 parent为 ${windowGain?'GAIN':'NO GAIN'}`:''}；物理收益来自两相位真实OpenFOAM CFD；v4冻结测试未访问。`:`历史 v4（非当前 full40/dev30）：严格证据已到 ${readyCount}/3 项；尚不能给出联合结论。终点/窗口只读v4 H20验证集，两相位收益只认真实OpenFOAM CFD；冻结测试未访问。`;
}
function plot(id,series,ymax=100){const el=$(id),w=Math.max(300,el.clientWidth),h=el.clientHeight;
 el.setAttribute('viewBox',`0 0 ${w} ${h}`);let s=`<line x1="45" y1="${h-30}" x2="${w-12}" y2="${h-30}" stroke="#668096"/>`;
 for(let tick=0;tick<=4;tick++){let y=12+(h-42)*tick/4;s+=`<line x1="45" y1="${y}" x2="${w-12}" y2="${y}" stroke="#25374b"/><text x="39" y="${y+4}" text-anchor="end" fill="#9aafc4" font-size="11">${num(ymax*(1-tick/4),ymax<=1?2:0)}</text>`}
 for(const entry of series){let values=entry.values.map((v,i)=>({x:45+i*(w-60)/Math.max(1,entry.values.length-1),y:12+(h-42)*(1-v/ymax)})).filter(v=>Number.isFinite(v.y));if(values.length){s+=`<polyline fill="none" stroke="${entry.color}" stroke-width="2.3" points="${values.map(v=>v.x+','+Math.max(12,Math.min(h-30,v.y))).join(' ')}"/>`;let v=values.at(-1);s+=`<circle cx="${v.x}" cy="${Math.max(12,Math.min(h-30,v.y))}" r="3" fill="${entry.color}"/>`}}
 let labels=id==='error-chart'?['1','10','50','100']:id==='train-chart'?['1','10']:[];
 labels.forEach((label,i)=>{let x=45+i*(w-60)/Math.max(1,labels.length-1);s+=`<text x="${x}" y="${h-9}" text-anchor="middle" fill="#9aafc4" font-size="11">${label}</text>`});
 el.innerHTML=s}
function resources(name,items){let last=items.at(-1);if(!last)return;
 if(last.error){$(name+'-more').textContent='采样失败：'+last.error;return}
 $(name+'-gpu').textContent=num(last.gpu,0)+'%';$(name+'-cpu').textContent=num(last.cpu,0)+'%';$(name+'-mem').textContent=num(last.mem_available_gib,1)+' GiB';
 $(name+'-mem').className='number '+(last.mem_available_gib>=20?'good':'bad');
 $(name+'-task').textContent=last.tasks?.length?`${last.tasks.join('、')} ${last.task_count>1?'×'+last.task_count:''}`:(name==='worker'&&last.second_seed_epoch>=10?'10 轮训练已完成；模型在主节点接受独立测试':'当前无计算任务');
 $(name+'-more').textContent=`采样时间 ${last.time} · GPU ${num(last.temp_c,0)}°C${name==='primary'&&Number.isFinite(last.cuda_free_gib)?` · CUDA 当前可直接分配 ${num(last.cuda_free_gib,1)} GiB（不同于上方可回收内存）`:''}`;
 plot(name+'-chart',[{values:items.map(x=>x.gpu),color:'#60c9fb'},{values:items.map(x=>x.cpu),color:'#e9ae68'}])}
function renderDualWatchdog(d){let w=d.dual_node_watchdog;
 if(!w){$('dual-watch-status').textContent='状态暂不可用';$('dual-watch-status').className='number bad';$('dual-watch-time').textContent='latest.json 尚未生成或不可读。';$('dual-watch-alerts').textContent='请检查 watchdog 服务。';return}
 let alerts=w.alerts||[],ok=w.status==='MONITORING'&&!alerts.length;$('dual-watch-status').textContent=ok?'运行正常':`ALERT · ${alerts.length} 项`;$('dual-watch-status').className='number '+(ok?'good':'bad');$('dual-watch-time').textContent=`最新采样 ${w.timestamp_utc||'未知'} · 状态来自只读 latest.json`;$('dual-watch-alerts').textContent=alerts.length?`告警：${alerts.join('、')}`:'当前无告警';$('dual-watch-alerts').className='small '+(alerts.length?'bad':'good');
 for(const [key,id,label] of [['spark','spark','Spark'],['worker78','worker','Worker']]){let n=w.nodes?.[key],title=$('dual-watch-'+id),detail=$('dual-watch-'+id+'-detail');if(!n){title.textContent='不可达/无数据';title.className='bad';continue}let active=n.useful_compute_active===true,idle=n.idle_duration_seconds||0;title.textContent=`CPU ${num(n.node_cpu_utilization_pct,1)}% · ${num(n.mem_available_gib,1)} GiB`;title.className=(n.memory_guard_pass&&idle<300)?'good':'bad';detail.textContent=`${label} 项目计算 ${active?'活跃':'空闲'} · 匹配进程 ${n.project_compute_process_count||0} · 连续空闲 ${idle} 秒 / 300 秒告警门槛 · 内存保护 ${num(n.memory_floor_gib,0)} GiB ${n.memory_guard_pass?'PASS':'BREACH'}`;}
 let p=w.progress||{};$('dual-watch-progress').textContent=`RAW ${p.strict_full40_receipt_count||0}/${p.strict_full40_receipt_target||31} · HDF ${p.matched9_staging_hdf5_count||0}/9`;$('dual-watch-progress-detail').textContent=`31案 aggregate ${p.raw_31_case_aggregate_complete?'完成':'未完成'} · full40 数据 ${p.full40_final_manifest_complete?'完成':'未完成'} · 正式闭环研究 ${p.project_complete?'完成':'未完成'}；数据完成不会被误写为训练/RL/闭环完成。`;
}
function renderMatchedStart(d){let worker=d.resources?.worker?.at(-1)||{},primary=d.resources?.primary?.at(-1)||{},rows=worker.matched_start||[],receipts=Object.fromEntries((d.matched_start_transfer||[]).map(x=>[x.case,x])),pipeline=Object.fromEntries((d.matched_start_pipeline||[]).map(x=>[x.case,x]));
 if(!rows.length){$('matched-summary').textContent='等待 Worker 真实采样';$('matched-cases').textContent='';return}
 let counts={complete:0,running:0,pending:0,stopped_incomplete:0};rows.forEach(x=>counts[x.status]=(counts[x.status]||0)+1);
 let verified=rows.filter(x=>receipts[x.case]?.raw_transfer_verified===true).length,vtkReady=rows.filter(x=>pipeline[x.case]?.vtk_ready===true).length,hdfReady=rows.filter(x=>pipeline[x.case]?.hdf_staging_status==='complete').length;
 $('matched-summary').textContent=`${counts.complete}/9 求解 · ${verified}/9 RAW · ${vtkReady}/9 VTK · ${hdfReady}/9 HDF`;
 $('matched-resource').textContent=`Curator 实际任务 ${primary.matched_start_curator_tasks||0} · Worker 整机 CPU ${num(worker.cpu,1)}% / 可用 ${num(worker.mem_available_gib,1)} GiB · Spark GPU ${num(primary.gpu,0)}% / 可用 ${num(primary.mem_available_gib,1)} GiB`;
 const state={complete:'求解完成',running:'运行中',pending:'待启动',stopped_incomplete:'异常停止/待核查'};
 $('matched-cases').innerHTML=rows.map(x=>{let accepted=receipts[x.case]?.raw_transfer_verified===true,p=pipeline[x.case]||{},hdf=p.hdf_staging_status||'pending',hdfText={complete:'HDF staging 完成',writing:'HDF staging 写入中',started:'Curator 已启动',conflict:'HDF staging 冲突',pending:'HDF staging 待处理'}[hdf]||'HDF staging 状态未知';return `<div class="caseitem"><b>${x.label}</b><span class="${x.status==='complete'?'good':x.status==='stopped_incomplete'?'bad':''}">${state[x.status]||'状态未知'}</span><div>${x.iteration}/16000 · ${num(100*x.iteration/16000,1)}%</div><div class="${accepted?'good':''}">${accepted?'RAW 已回传验收':'RAW 待回传验收'}</div><div class="${p.vtk_ready?'good':''}">${p.vtk_ready?'VTK_READY 801帧':'VTK 待验证'}</div><div class="${hdf==='complete'?'good':hdf==='conflict'?'bad':''}">${hdfText}</div><div class="bar"><i style="width:${Math.min(100,100*x.iteration/16000)}%"></i></div></div>`}).join('');}
function renderFull40(d){let rows=d.full40_extension?.cases||[],worker=d.resources?.worker?.at(-1)||{},physics=d.matched_start_physics_summary;
 if(physics?.status==='MATCHED_START_9_CASE_TRAIN_COMMISSIONING_PHYSICS_SUMMARY'){$('matched-physics').textContent=`九案真实OpenFOAM、仅train-phase开放环：−0.75 / +0.75平均降阻 ${pct(physics.actions?.m075?.mean_drag_reduction)} / ${pct(physics.actions?.p075?.mean_drag_reduction)}，平均Cl偏置比 ${num(physics.actions?.m075?.mean_lift_bias_ratio,3)} / ${num(physics.actions?.p075?.mean_lift_bias_ratio,3)}，联合目标 ${physics.joint_pass_count}/${physics.comparison_count} 通过。不是验证/冻结测试或闭环收益。`;}
 if(!rows.length){$('full40-summary').textContent='scheduler 状态不可用';$('full40-cases').textContent='';return}
 let qc=rows.filter(x=>x.raw_qc_verified).length,active=rows.filter(x=>x.scheduler_status==='ACQUIRED'&&x.alive).length,ended=rows.filter(x=>x.solver_completed).length;
 $('full40-summary').textContent=`${active} 求解中 · ${ended}/31 求解完成 · ${qc}/31 RAW QC`;
 $('full40-resource').textContent=`Worker 整机 CPU ${num(worker.cpu,1)}% · 可用统一内存 ${num(worker.mem_available_gib,1)} GiB · scheduler ${d.full40_extension.status||'未知'}`;
 let labels={train:'训练',validation:'验证',frozen_test:'冻结测试'};$('full40-splits').textContent=['train','validation','frozen_test'].map(s=>{let r=rows.filter(x=>x.split===s);return `${labels[s]} ${r.filter(x=>x.raw_qc_verified).length}/${r.length} RAW QC`}).join(' · ');
 let state={ACQUIRED:'求解中',COMPLETED:'求解完成、待RAW QC',NOT_GENERATED:'待生成',SPARK_GENERATED:'Spark已生成',WORKER_STAGED:'Worker已暂存',PENDING:'待启动',FAILED:'失败',ORPHANED:'进程异常',DIRTY_PENDING:'脏状态待核查'};
 $('full40-cases').innerHTML=rows.map(x=>`<div class="caseitem"><b>${labels[x.split]} · b${String(x.phase_bin).padStart(2,'0')} · ${x.action_target>0?'+':''}${x.action_target}</b><span class="${x.raw_qc_verified?'good':['FAILED','ORPHANED','DIRTY_PENDING'].includes(x.scheduler_status)?'bad':''}">${x.raw_qc_verified?'RAW QC 已通过':state[x.scheduler_status]||x.scheduler_status}</span><div>${x.raw_qc_verified?'严格receipt已验收':x.solver_completed?'solver完成≠QC通过':'RAW QC待验收'}</div>${x.split==='frozen_test'?'<div>冻结测试：仅显示采集状态</div>':''}</div>`).join('');}
function renderTrain20Physics(d){let p=d.full40_train20_physics,s=p?.all_train_control_macro_and_worst,a=p?.per_action_macro_and_worst;
 if(p?.status!=='FULL40_TRAIN20_OPEN_LOOP_PHYSICS_SUMMARY'||!s||!a){$('train20-physics').textContent='等待 train20 权威 JSON';return}
 $('train20-physics').textContent=`TRAIN ONLY · 联合门槛 ${s.joint_pass_count}/${s.comparison_count} 通过 · 全动作 macro 降阻 ${pct(s.macro_mean_total_drag_reduction_fraction)}`;$('train20-physics').className='number '+(s.joint_pass_count?'good':'bad');
 let minus=a.m075,plus=a.p075;$('train20-actions').textContent=`−0.75 / +0.75 macro 降阻 ${pct(minus.macro_mean_total_drag_reduction_fraction)} / ${pct(plus.macro_mean_total_drag_reduction_fraction)}；worst phase ${pct(minus.worst_drag_reduction.total_drag_reduction_fraction_positive_is_better)} / ${pct(plus.worst_drag_reduction.total_drag_reduction_fraction_positive_is_better)}；macro 平均Cl偏置比 ${num(minus.macro_mean_abs_rear_cl_bias_ratio,3)} / ${num(plus.macro_mean_abs_rear_cl_bias_ratio,3)}（门槛≤0.10）。`;
 $('train20-scope').textContent=`四个train相位、16个非零固定动作、各自同相位zero、固定末60D/U。validation/frozen结果读取=${p.scope?.validation_or_frozen_results_read?'是（异常）':'否'}；不替代九案 commissioning，不代表泛化、PPO或真实闭环收益。`;}
function renderFreeAR(d){let x=d.free_ar_ablation||{},a=x.h20||{},b=x.h50||{},dh=x.dynamic_h100||{},am=a.last_metrics||{},bm=b.last_metrics||{},dhm=dh.last_metrics||{},bval=b.validation10||{},b100=bval.horizons?.['100']||{},dyn=x.dynamic6_fno||{},screen=x.b5_dynamic6_start0||{},direct=x.direct_cfd_ppo||{},pe=a.post_evaluation||{};
 const stage=(v,m)=>`${v.epoch||0}/${v.expected_epochs||8}轮${Number.isFinite(m.selection_score)?` · H100 selection ${num(m.selection_score,4)}`:''}`;
 $('free-ar-h20').textContent=stage(a,am);$('free-ar-h20').className=a.status==='COMPLETE'?'good':a.status==='FAILED'?'bad':'';$('free-ar-h20-detail').textContent=`训练已完整 ${a.epoch||0}/${a.expected_epochs||8} 轮；完整validation10后评估 ${pe.complete?'完成':pe.active?'修正路径后重试中':'待完成'}。首次后评估仅因可视化写入只读路径失败，不是模型训练失败；失败日志保留。pure-AR H20→验证H100${Number.isFinite(am.terminal_state_mae)?` · 终点反归一化流场综合MAE / 四个力系数平均MAE ${num(am.terminal_state_mae,4)} / ${num(am.terminal_force_mae,4)}`:''}${Number.isFinite(am.terminal_total_drag_pooled_nrmse)?` · 训练内stride100终点pooled总Cd NRMSE ${pct(am.terminal_total_drag_pooled_nrmse)}`:''}。训练内统计不等于stride25完整评估，也不是减阻率或正式Gate。`;
 $('free-ar-h50').textContent=b.status==='INTENTIONALLY_REALLOCATED_AT_EPOCH5'?`5/8轮 · e5冻结；动态H100 ${dh.epoch||0}/2轮`:stage(b,bm);$('free-ar-h50').className=b.status==='COMPLETE'||dh.status==='COMPLETE'?'good':b.status==='FAILED'||dh.status==='FAILED'?'bad':'';let dynamicMetrics=Number.isFinite(dhm.terminal_state_mae)?`动态H100最新epoch内validation10：终点流场综合MAE / 四力平均MAE ${num(dhm.terminal_state_mae,4)} / ${num(dhm.terminal_force_mae,4)}${Number.isFinite(dhm.terminal_total_drag_pooled_nrmse)?`，终点总Cd NRMSE ${pct(dhm.terminal_total_drag_pooled_nrmse)}`:''}；相对B5父模型流场${dhm.terminal_state_mae<bm.terminal_state_mae?'改善':'变差'}、四力${dhm.terminal_force_mae<bm.terminal_force_mae?'改善':'变差'}。这是epoch内validation10，不是dynamic6或最终门槛。`:'';$('free-ar-h50-detail').textContent=`${b.status==='INTENTIONALLY_REALLOCATED_AT_EPOCH5'?'固定动作训练按研究决策在e5有意结束（计划8、实际5），不是8轮完成或故障；资源转投真实动态动作H100缺陷。':b.service_state||b.status||'等待'} 新动态run=${dh.status||'等待'}${Number.isFinite(dhm.selection_score)?` · 最新H100 selection ${num(dhm.selection_score,4)}`:''}；${dynamicMetrics} e5纯AR H50${Number.isFinite(bm.terminal_state_mae)?`终点反归一化流场综合MAE / 四个力系数平均MAE ${num(bm.terminal_state_mae,4)} / ${num(bm.terminal_force_mae,4)}`:''} · 已验SHA回传 ${b.synced_epochs||0}/8轮。${Number.isFinite(b100.pooled_total_cd_nrmse)?`e1独立validation10 H100 pooled总Cd NRMSE ${pct(b100.pooled_total_cd_nrmse)}（macro ${pct(b100.macro_total_cd_nrmse)}），仍高于10%门槛。`:''}不是减阻率。`;
 let evaluating=x.dynamic6_service_state==='active'&&dyn.checkpoint_epoch!==1,failed=dyn.status==='DYNAMIC6_FNO_DIAGNOSTIC_FAIL',screenReady=screen.status==='PARENT_START0_SCREEN_COMPLETE_NOT_FULL_QUALIFICATION';$('free-ar-dynamic').textContent=screenReady?`B5有限筛查 FAIL · H100总Cd ${pct(screen.pooled_total_drag_nrmse)}`:evaluating?'H50 e1 · Dynamic6评估中':dyn.status?`${failed?'FAIL':'诊断完成'} · H100 pooled ${Number.isFinite(dyn.pooled_h100_total_cd_nrmse)?pct(dyn.pooled_h100_total_cd_nrmse):'—'}`:'等待诊断';$('free-ar-dynamic').className=screenReady||(!evaluating&&failed)?'bad':'';$('free-ar-dynamic-detail').textContent=screenReady?`B5父模型、6条真实动态CFD、共同start0的单个H100有限筛查：总Cd NRMSE ${pct(screen.pooled_total_drag_nrmse)}，速度场相对L2 ${pct(screen.velocity_vector_relative_l2)}，动作效应符号 ${screen.correct_action_effect_signs||0}/${screen.nonzero_action_pairs||4}。绝非完整rollout或formal gate；FNO未通过，继续动态数据训练。`:evaluating?'正在用H50 e1固定checkpoint评估全部6条动态动作真实CFD；parent e5失败仅为上一候选。':dyn.status?`${dyn.checkpoint_epoch===1?'H50 e1':'parent e5'} · 动作差值MAE ${num(dyn.strict_start0_h100_delta_total_cd_mae,4)}（限值 ${num(dyn.strict_delta_limit,3)}）；PPO授权=${dyn.ppo_authorized?'是':'否'}。全6案真实CFD validation-only，不是控制收益。`:'等待H50 e1对全6条动态动作轨迹的FNO诊断。';
 let completed=direct.completed_transitions||0,live=direct.live_collection_steps||0,updates=direct.ppo_update_count||0,last=direct.last_checkpoint||{},reward=last.raw_physical_reward||{},pair=direct.paired_80d_evaluation||{},independent=direct.independent_b01_evaluation||{},replay=direct.b00_sequence_b01_replay||{},rs=replay.physical_summary||{},comparison=pair.physical_result?.comparison||{},icomparison=independent.physical_result?.comparison||{},joint=comparison.canonical_physical_joint_check===true,ijoint=icomparison.canonical_physical_joint_check===true,checks=comparison.canonical_physical_checks||{};$('free-ar-direct-ppo').textContent=independent.physical_result?.status?`真实CFD：b00 ${joint?'PASS':'FAIL'} · b01另一启动时刻 ${ijoint?'PASS':'FAIL'}`:pair.service_state==='active'?`训练完成 2048/2048 · 80D配对CFD ${pair.completed_steps_per_branch||0}/800步`:pair.physical_result?.status?`80D配对CFD完成 · joint ${joint?'PASS':'FAIL'}`:direct.training_complete?'真实CFD训练完成 · 2048/2048步':direct.running?`真实CFD训练中 · ${completed}/2048步${completed<256?`（首轮 ${live}/256）`:''}`:'实现与测试中';$('free-ar-direct-ppo').className=independent.physical_result?.status?(joint&&ijoint?'good':'bad'):pair.physical_result?.status?(joint?'good':'bad'):direct.training_complete?'good':'';let replayLines=replay.status==='B00_ACTIONS_B01_OPENLOOP_INDEPENDENT_AUDIT_PASS'?`\n零控制：Cd ${num(rs.b01_zero_total_cd_mean,6)}，Cl′基准1.000。\nb00固定动作序列回放：Cd ${num(rs.openloop_total_cd_mean,6)}（增阻 ${pct(-rs.openloop_drag_reduction_vs_zero)}），Cl′比 ${num(rs.openloop_rear_cl_fluctuation_rms_ratio_vs_zero,6)}。\n反馈策略：Cd ${num(rs.b01_feedback_total_cd_mean,6)}（减阻 ${pct(rs.feedback_drag_reduction_vs_zero)}），Cl′比 ${num(icomparison.rear_cl_fluctuation_rms_ratio,6)}。此起点上反馈对减阻有附加价值，固定序列更抑制升力波动；不表示所有指标都更好，也不证明统计独立或广泛泛化。`:'';let pdetail=$('free-ar-direct-ppo-detail');pdetail.style.whiteSpace='pre-line';pdetail.textContent=(independent.physical_result?.status?`同一冻结final2048策略、真实OpenFOAM配对末60D/U：b00训练起点总阻力降低 ${pct(comparison.total_drag_reduction)}、Cl′比 ${num(comparison.rear_cl_fluctuation_rms_ratio,3)}、偏置比 ${num(comparison.abs_rear_cl_mean_over_zero_fluctuation_rms,3)}；b01未用于训练的另一启动时刻总阻力降低 ${pct(icomparison.total_drag_reduction)}、Cl′比 ${num(icomparison.rear_cl_fluctuation_rms_ratio,3)}、偏置比 ${num(icomparison.abs_rear_cl_mean_over_zero_fluctuation_rms,3)}。两起点相差18D/U（约3个脱涡周期），统计独立性尚未证明；仍不是多相位泛化、冻结测试、净能耗或论文最终结论。`:pair.physical_result?.status?`b00 train-phase配对末60D/U：总阻力降低 ${Number.isFinite(comparison.total_drag_reduction)?pct(comparison.total_drag_reduction):'—'}，后柱Cl′比 ${num(comparison.rear_cl_fluctuation_rms_ratio,3)}，mean-Cl偏置比 ${num(comparison.abs_rear_cl_mean_over_zero_fluctuation_rms,3)}；三项门槛 ${Object.entries(checks).filter(([,v])=>!v).map(([k])=>k).join('、')||'均通过'}。仅训练相位初步物理验证，不是独立泛化或论文结论。`:direct.running||direct.training_complete?`PPO更新 ${updates} 次${Number.isFinite(reward.mean)?` · 最近256步真实reward均值 ${num(reward.mean,4)}`:''}。FNO未用于奖励；${pair.service_state==='active'?'最终策略与zero正在做80D真实OpenFOAM配对。':'物理减阻尚待80D配对CFD验收。'}`:'直接CFD反馈，不依赖FNO代理；尚无运行日志或控制收益。')+replayLines;let pfig=$('free-ar-direct-ppo-figure'),pnote=$('free-ar-direct-ppo-figure-note');if(pair.figure?.available){pfig.src=pair.figure.path+'?v='+pair.figure.version;pfig.hidden=false;pnote.textContent='b00原始未平滑时序：灰色为前20D/U过渡，蓝色为后60D/U统计；上图总Cd、中图后柱Cl、下图为每0.1D/U记录的动作端点及CFD实际执行的区间线性ramp（含t=0初始ω=0），不是阶梯保持。';pnote.hidden=false}else{pfig.hidden=true;pnote.hidden=true}}
function renderThreeGoals(d){let x=d.free_ar_ablation||{},direct=x.direct_cfd_ppo||{},b00=direct.paired_80d_evaluation?.physical_result?.comparison||{},b01=direct.independent_b01_evaluation?.physical_result?.comparison||{},bothReal=b00.canonical_physical_joint_check===true&&b01.canonical_physical_joint_check===true,v=d.h50_dynamic_visualization||{},train16=v.train16_manifest||{};$('goal-real-cfd').textContent=bothReal?'两个已测起点均约4.2%减阻 · 升力约束PASS':'真实CFD反馈验收进行中';$('goal-real-cfd').className=bothReal?'good':'';$('goal-real-cfd-detail').textContent=bothReal?'b00训练起点4.22%，b01另一启动时刻4.25%；目标≥2%，后柱Cl′与均值偏置均达标。两起点统计独立性未证明。':'等待真实OpenFOAM成对结果。';$('goal-fno').textContent=v.figure_count===12?'Dynamic6 正式验收 FAIL · 后柱升力仍是主要误差':'等待固定流场诊断';$('goal-fno').className='bad';$('goal-fno-detail').textContent=v.figure_count===12?'已完成评估配置修复后的完整推理。H50候选虽改善总阻力预测，但动态动作差值MAE 0.0301 > 0.023；validation10 H100后柱Cl MAE 0.0602。下方12张固定图如实展示空间误差。':'固定validation-only图尚未齐全。';$('goal-surrogate-control').textContent='未获代理PPO授权';$('goal-surrogate-control').className='bad';$('goal-surrogate-control').nextElementSibling.textContent=train16.status==='DIRECTPPO_TRAIN16_TRAIN_ONLY_CURATED'?'16条真实PPO交互轨迹已通过官方DataPipe读取；仅供下一轮训练，不把技术probe或训练完成误写成控制成功。':'新增真实交互数据等待严格发布与官方DataPipe读取。';}
function renderH50Figures(d){let v=d.h50_dynamic_visualization||{},h=$('h50-horizon').value;for(const role of ['zero','minus','plus']){let image=$('h50-'+role),status=$('h50-'+role+'-status'),found=v.figures?.[role+'/'+h];if(found){image.src=found.path+'?v='+found.version;image.hidden=false;status.textContent=`b01 ${role} · H${Number(h)}（${num(Number(h)*.1,1)} D/U）· validation-only · Dynamic6 formal FAIL`}else{image.hidden=true;status.textContent='固定图缺失；不回退到其他checkpoint或案例。'}}}
function renderTrain16Formal(d){let s=d.free_ar_ablation?.control_train16_h100||{},m=s.last_metrics||{},w=d.training_evaluation_watchdog||{},next=w.scientific_next_stage||{},blocked=w.status==='ALERT'&&w.workflow_pending,evaluating=w.workflow_pending&&w.active_units?.length,pairedPending=next.paired_posteval_status==='PAIRED_POSTEVAL_PENDING',active=s.service_state==='active',complete=s.epoch===s.expected_epochs&&s.expected_epochs===2,sciencePending=w.stage_complete&&w.project_goal_complete===false,title=blocked?`红色阻塞 · ${w.alerts?.join(' / ')||'评估待办无运行任务'}`:pairedPending?'λ0/λ10训练已完成 · paired后评估待批准':sciencePending?'后评估阶段完成 · 模型仍需改进':evaluating?`正式训练 ${s.epoch||0}/2已完成 · 后评估运行中`:complete?`正式训练 ${s.epoch}/2轮已落盘`:active?`正式训练运行中 · ${s.epoch||0}/2轮已落盘`:`正式训练 ${s.epoch||0}/2轮 · ${s.service_state||'未启动'}`;$('train16-formal-progress').textContent=title;$('train16-formal-progress').className=blocked||s.status==='FAILED'||sciencePending?'number bad':complete?'number good':'number';let state=`最近文件 ${w.latest_relevant_file?.path||'无'} @ ${w.latest_relevant_file?.modified_at_utc||'未知'}；GPU ${num(w.resources?.gpu_utilization_pct,0)}%，CPU ${num(w.resources?.cpu_utilization_pct,1)}%，MemAvailable ${num(w.resources?.mem_available_gib,1)} GiB。`;let authority=Object.entries(w.authority_tasks||{}).map(([k,v])=>`${k}=${v.state}`).join('，');let recovery=w.policy?.auto_recovery_enabled?'仅允许已审查幂等恢复':'自动恢复未启用：'+(w.policy?.auto_recovery_reason||'尚无已审查幂等入口');let historical=(w.superseded_failures||[]).length?` 历史superseded失败 ${w.superseded_failures.length} 个，仅审计不计当前故障。`:'';let nextWork=next.formal_training_units_assigned?`λ0正式PhysicsNeMo训练=${next.lambda0?.state||'未知'}，λ10=${next.lambda10?.state||'未知'}；训练完成仍须独立posteval，不等于科学达标。`:'正在准备train-only paired统计DataPipe与下一轮训练；尚无正式训练unit时不虚称GPU在训。';let operational=blocked?` 运维阻塞原因：${w.blocker_reasons?.join('；')||`待办存在且连续 ${w.no_running_duration_seconds||0} 秒无运行unit`}。当前权威：${authority||'未知'}；${recovery}。${state}${historical}未知故障需要agent分析，不会自动改代码或绕过科学FAIL。`:pairedPending?` λ0/λ10训练与λ10回传均已核验；FC-P001 paired posteval尚未执行，下一owner=${next.next_owner||'Lead'}、批准项=${next.approval_required||'FC-P001'}。当前没有训练或评估GPU任务，不虚报运行。${state}`:evaluating?` 后评估unit：${w.active_units.join('、')}；当前权威：${authority||'未知'}；${recovery}。${state}${historical}`:sciencePending?` 评估收据完成不等于项目目标完成；科学FAIL保持 NEEDS_MODEL_IMPROVEMENT。当前工作=${next.status||'模型改进设计'}，${nextWork}`:'';$('train16-formal-detail').textContent=`唯一正式目录 tandem_fno_control_train16_h100_20261004；${s.epoch?`最新完整epoch ${s.epoch}${Number.isFinite(m.selection_score)?`，selection ${num(m.selection_score,5)}`:''}。`:'首轮指标尚未原子落盘，不虚报进度。'} one-batch技术probe明确排除，不计候选、epoch或精度成果；完整训练后仍须独立validation10与Dynamic6，当前不宣称控制成功。${operational}`;}
const renderTrain16FormalBase=renderTrain16Formal;
renderTrain16Formal=function(d){renderTrain16FormalBase(d);let w=d.training_evaluation_watchdog||{},next=w.scientific_next_stage||{};if(next.approval_state==='LEAD_APPROVED'&&next.paired_posteval_complete!==true){let running=next.paired_posteval_status==='PAIRED_POSTEVAL_RUNNING';$('train16-formal-progress').textContent=running?'FC-P001 paired后评估运行中':'FC-P001 Lead已批准 · 实现/血缘预检中';$('train16-formal-progress').className='number';$('train16-formal-detail').textContent=`λ0/λ10训练与回传已核验；FC-P001已由Lead批准，不等待用户确认。${running?'权威评估unit已运行。':'尚无权威评估unit，不虚报GPU评估。'} 固定validation10、Dynamic6、force-window协议及原门槛；不重训、不访问frozen、不预填结果。owner=${next.next_owner||'Surrogate + Physics/Data'}；批准记录=${next.approval_reference||'docs/FC-P001_APPROVAL.md'}。`;}};
function renderFull40Chain(d){let c=d.full40_development_chain||{},raw=c.raw_qc||{},hdf=c.development_hdf||{},release=c.dev30_release||{},one=c.quickscreen?.onestep||{},h20=c.quickscreen?.h20||{},diag=c.validation_diagnostic||{},ppo=c.canonical_ppo||{},dyn=d.dynamic6_runtime||{},incident=c.execution_incident||{},spark=d.dual_node_watchdog?.nodes?.spark||{},worker=d.dual_node_watchdog?.nodes?.worker78||{},legacy=d.watchdog||{};
 $('chain-raw').textContent=`RAW31 ${raw.full40_verified||0}/31 · 九案 ${raw.commissioning_qc_pass?'QC PASS':'QC待通过'}`;$('chain-raw').className=raw.full40_verified===31&&raw.commissioning_qc_pass?'good':'';$('chain-raw-detail').textContent=`新增31案严格receipt ${raw.full40_verified||0}/31；九案独立QC ${raw.commissioning_qc_pass?'已通过':'尚未通过'}。solver完成不能替代RAW/HDF QC。`;
 $('chain-hdf').textContent=`train ${hdf.train_ready||0}/20 · validation ${hdf.validation_ready||0}/10`;$('chain-hdf').className=hdf.train_ready===20&&hdf.validation_ready===10?'good':'';$('chain-release').textContent=release.published?`dev30 immutable 已发布 · manifest ${String(release.manifest_sha256||'').slice(0,12)}…`:`Spark本机已验收HDF；dev30 BLOCKED · ${release.reason||'等待30个开发HDF'}`;
 let om=one.last_metrics||{},hm=h20.last_metrics||{};$('chain-train').textContent=`one-step ${one.epoch||0}/10 · H20 ${h20.epoch||0}/5`;$('chain-train').className=h20.epoch===5?'good':'';$('chain-train-detail').textContent=`${one.run_id||'one-step未启动'}${Number.isFinite(om.train_loss)?` · train loss ${num(om.train_loss,5)} · val field/force ${num(om.state_mae_physical_units,4)}/${num(om.force_mae_normalized,4)}`:''}；${h20.run_id||'H20未启动'}${Number.isFinite(hm.train_loss)?` · train loss ${num(hm.train_loss,5)} · val rollout field/force ${num(hm.rollout_state_mae,4)}/${num(hm.rollout_force_mae,4)} · selection ${num(hm.selection_score,4)}`:''}。${incident.status?'原pipeline误停已记录，当前H20由独立服务恢复；旧RUNNING JSON不作实时状态。':''}阶段候选，不是formal Gate。`;
 let dc=dyn.current_case;$('chain-dynamic').textContent=`${dyn.completed_cases||0}/6 完成${dc?` · b${String(dc.phase_bin).padStart(2,'0')}/${dc.profile} ${dc.steps||0}/4000步`:''}`;$('chain-dynamic').className=dyn.status==='SOLVER_COMPLETE_PENDING_PANEL_QC'?'good':dyn.status==='FAILED'?'bad':'';$('chain-dynamic-detail').textContent=`${dyn.service_state||'服务状态未知'} · AI训练叶进程 ${dyn.ai_training_processes??'—'} · CFD求解进程 ${dyn.cfd_solver_processes??'—'}；Spark CPU ${num(spark.node_cpu_utilization_pct,1)}% / GPU ${num(legacy.primary?.gpu_utilization_pct,0)}% / Mem ${num(spark.mem_available_gib,1)} GiB；Worker CPU ${num(worker.node_cpu_utilization_pct,1)}% / GPU ${num(legacy.worker?.gpu_utilization_pct,0)}% / Mem ${num(worker.mem_available_gib,1)} GiB。旧watchdog混合计数不作任务类型解释；真实CFD validation-only，未做收益结论。`;
 let physical=d.dynamic6_physical_qc,primary=[],segments=[];for(const phase of ['b01','b05']){let comparisons=physical?.phases?.[phase]?.comparisons_to_paired_zero||{};for(const role of ['minus','plus']){let m=comparisons.elapsed_0_20?.[role];if(m)primary.push(`${phase} ${role==='minus'?'−':'+'} · 0–20D/U：Cd ${m.total_drag_reduction>=0?'降':'增'} ${pct(Math.abs(m.total_drag_reduction))}，Cl′比 ${num(m.rear_cl_fluctuation_rms_ratio,3)}，mean-Cl比 ${num(m.abs_rear_cl_mean_over_zero_fluctuation_rms,3)}，joint ${m.canonical_physical_joint_check?'PASS':'FAIL'}`)}for(const [key,label] of [['elapsed_0_10','0–10'],['elapsed_10_20','10–20']]){for(const role of ['minus','plus']){let m=comparisons[key]?.[role];if(m)segments.push(`${phase} ${role==='minus'?'−':'+'} · ${label}D/U：Cd ${m.total_drag_reduction>=0?'降':'增'} ${pct(Math.abs(m.total_drag_reduction))}，Cl′比 ${num(m.rear_cl_fluctuation_rms_ratio,3)}，mean-Cl比 ${num(m.abs_rear_cl_mean_over_zero_fluctuation_rms,3)}，joint ${m.canonical_physical_joint_check?'PASS':'FAIL'}`)}}}$('chain-dynamic-results').innerHTML=primary.length?`${primary.join('<br>')}<br><b>b01选minus、b05选plus是读到validation结果后的post-hoc per-phase oracle，不是已验证策略或政策收益。仅20D/U开环筛选，不是FNO/PPO或最终末60D/U验收。</b>${segments.length?`<details><summary>展开 0–10 / 10–20D/U 分段明细（8行）</summary><div class="small" style="margin-top:7px">${segments.join('<br>')}</div></details>`:''}`:'等待六案 aggregate CFD QC；不是FNO预测、PPO闭环或末60D/U最终验收。';
 let h100=diag.horizons?.['100']?.pooled_total_cd_nrmse;$('chain-validation').textContent=Number.isFinite(h100)?`dev30 validation10：H100 pooled总Cd NRMSE ${pct(h100)}；动作排序 ${pct(diag.h100_start0_action_ranking?.cross_action_ordering_accuracy)}。仅开发诊断，冻结测试未访问。`:'dev30 validation10诊断尚未生成；不以训练loss替代H100与动作排序。';
 $('chain-ppo').textContent=ppo.started?'已启动':`BLOCKED`;$('chain-ppo').className=ppo.started?'':'bad';$('chain-ppo-detail').textContent=ppo.started?`canonical PPO已有真实run记录：${ppo.status||'状态待审计'}。`:`${ppo.status||'promotion/formal validation gates未满足'}；未训练即明确BLOCKED，不沿用旧v4 PPO。`;}
function figure(){if(!latest)return;let key=$('case').value+'/'+$('horizon').value;let found=latest.figures[key];if(found){$('flow').src=found.path+'?v='+found.version;$('flow').hidden=false;$('figure-label').textContent=found.label}else{$('flow').hidden=true;$('figure-label').textContent='该工况暂无导出的对照图'}}
function render(d){latest=d;$('clock').textContent='服务器 '+d.server_time+' · 页面每 5 秒更新';let candidates=[{label:'20 步＋后柱阻力加权训练',audit:d.v3_h20_rear_drag_audit,observed:d.v3_h20_rear_drag_observed},{label:'主节点种子 20 步训练',audit:d.v3_primary_seed_h20_audit,observed:d.v3_primary_seed_h20_observed},{label:'主节点后圆柱加权训练',audit:d.v3_rear_weighted_audit,observed:d.v3_rear_weighted_observed},{label:'计算节点种子 20 步训练',audit:d.v3_h20_audit,observed:d.v3_h20_observed},{label:'主节点 10 步训练',audit:d.v3_primary_rollout_audit,observed:d.v3_primary_rollout_observed},{label:'计算节点 10 步训练',audit:d.v3_rollout_audit,observed:d.v3_rollout_observed},{label:'主节点单步训练',audit:d.v3_audit,observed:d.v3_observed},{label:'计算节点单步训练',audit:d.v3_worker_audit,observed:d.v3_worker_observed}];let picked=candidates.find(x=>x.audit?.status==='GATE_B_PASS')||candidates.find(x=>x.observed)||candidates.find(x=>x.audit);let audited=picked?.audit,a=audited||d.audit,c=a?.checks||{};
 let long=d.control_landscape_long?.cases||[],zero=long.find(x=>x.omega_final===0),plus=long.find(x=>x.omega_final===1),minus=long.find(x=>x.omega_final===-1);
 $('physical-drag').textContent=plus&&minus?`+1: ${num(plus.relative_to_zero.total_cd_change_percent,2)}% · −1: ${num(minus.relative_to_zero.total_cd_change_percent,2)}%`:'等待真实 CFD 审计';
 $('physical-tradeoff').textContent=plus&&minus?`恒转两案的平均后柱升力均超出既定上限；总升力 RMS 分别为零控制的 ${num(plus.relative_to_zero.rear_cl_total_rms_ratio,2)}、${num(minus.relative_to_zero.rear_cl_total_rms_ratio,2)} 倍。`:'统计窗口 t=120–160；同时检查阻力与侧向载荷。';
 let periodic=d.periodic_benchmark?.decisions||[];$('periodic-detail').textContent=periodic.length===2?`零均值周期 P10/P20：总阻力变化 ${num(-100*periodic[0].total_cd_reduction,2)}% / ${num(-100*periodic[1].total_cd_reduction,2)}%；后柱升力脉动比 ${num(periodic[0].rear_cl_fluctuation_rms_ratio,2)} / ${num(periodic[1].rear_cl_fluctuation_rms_ratio,2)}；均未通过联合门槛。仅单相位开环 CFD。`:'零均值周期转速基线：等待真实 CFD 审计。';
 let cohort=d.existing_open_loop;$('cohort-detail').textContent=cohort?`已有真实 CFD 训练/验证轨迹：${cohort.joint_pass_count}/${cohort.trajectory_count} 条同时满足降阻与升力约束；冻结测试未用于筛选。`:'已有开环 CFD 轨迹联合验收：等待审计。';
 let pilots=d.phase_feedback_pilots||[],ready=pilots.filter(x=>x?.comparison),passed=ready.filter(x=>x.comparison.total_drag_reduction>=.02&&x.comparison.rear_cl_fluctuation_rms_ratio<=1.05&&x.comparison.abs_rear_cl_mean_over_zero_fluctuation_rms<=.1),bestPilot=ready.reduce((a,b)=>!a||b.comparison.total_drag_reduction>a.comparison.total_drag_reduction?b:a,null);$('feedback-detail').textContent=ready.length===3?`短时真实 CFD 反馈：${passed.length}/3 种方案满足联合目标；最佳减阻 ${num(100*bestPilot.comparison.total_drag_reduction,2)}%，但其后柱升力波动增大 ${num(100*(bestPilot.comparison.rear_cl_fluctuation_rms_ratio-1),1)}%。仅一周期试验，不等于最终结果。`:ready.length?`短时真实 CFD 反馈：已完成 ${ready.length}/3 组，继续核对。`:'短时真实 CFD 反馈：等待配对结果。';
 let rank=d.control_ranking_h20_rear_drag||d.control_ranking,lowFno=d.low_action_fno_h100,lowV3=lowFno?.models?.v3_parent,lowV4=lowFno?.models?.v4_candidate;
 $('ranking-score').textContent=lowV3&&lowV4?`低幅H100 pooled：v3 ${pct(lowV3.h100_pooled_total_drag_nrmse)} · v4 ${pct(lowV4.h100_pooled_total_drag_nrmse)}`:rank?`${Math.round(rank.pairwise_ranking_accuracy*6)}/6 对动作排序正确`:'等待动作排序审计';
 $('ranking-detail').textContent=lowV3&&lowV4?`低幅±0.75 validation瞬时终点：总Cd MAE / persistence MAE，v3 ${num(lowV3.h100_total_drag_mae,4)} / ${num(lowV3.h100_persistence_total_drag_mae,4)}，v4 ${num(lowV4.h100_total_drag_mae,4)} / ${num(lowV4.h100_persistence_total_drag_mae,4)}，两模型均差于persistence；唯一严格同初态start=0的动作差值误差为 ${num(lowV3.strict_start0_pairwise_delta_absolute_error,4)} / ${num(lowV4.strict_start0_pairwise_delta_absolute_error,4)}。仅验证集诊断，不是零控制收益或CFD闭环成功。`:d.control_ranking_h20_rear_drag?`20 步＋后柱阻力加权 FNO 选 −1，与该验证窗口的 CFD 最优一致；真实 CFD 相对零转速阻力降低 ${num(-rank.cfd_change_of_fno_selection_vs_zero_percent,2)}%，但 100 步预测误差仍超标，不能作为闭环控制结果。`:rank?`此前 20 步 FNO 选 +1，真实 CFD 在启动窗口选 −1；选错的阻力代价 ${num(rank.cfd_regret_of_fno_selection,4)}。`:'用相同初始流场检验真实 CFD 与 FNO 决策是否一致。';
 let phase=d.crossphase86_ranking;$('crossphase-ranking').textContent=phase?.status==='VALIDATION_ONLY_CROSSPHASE_ACTION_RANKING_DIAGNOSTIC'?`另一个初始相位 t=86，20 步、3 个动作：FNO 排序 ${num(100*phase.pairwise_ranking_accuracy,0)}% 正确，选 ${phase.fno_selected_case?.includes('_m100_')?'−1':phase.fno_selected_case?.includes('_p100_')?'+1':'0'}；窗口短于一个涡脱落周期，升力约束未通过，不能视为控制达标。`:'独立相位短窗口：等待审计。';
 let observed=picked?.observed,preliminary=observed?.summary?.['100']?.total_drag_nrmse;
 let strict=d.gate_b_metric_integrity, b=strict?.pooled_total_drag_nrmse??(audited?c.heldout_full_period_total_drag_nrmse?.total_drag_nrmse:(Number.isFinite(preliminary)?preliminary:c.heldout_full_period_total_drag_nrmse?.total_drag_nrmse)),i=c.independent_phase_full_period_total_drag_nrmse?.total_drag_nrmse;
 $('heldout').textContent=pct(b);$('heldout').className=audited&&Number.isFinite(b)&&b<=.1?'good':Number.isFinite(b)&&b>.1?'bad':'';
 let v4primary=d.v4_h20_history||[],v4gate=d.v4_validation_decision,v4err=v4gate?.candidate_terminal_100step?.pooled_total_drag_nrmse,v4threshold=v4gate?.fixed_gate_b_threshold;$('epoch').textContent=`新数据多步 ${v4primary.length}/5 轮`;$('hydro-status').textContent=Number.isFinite(v4err)&&Number.isFinite(v4threshold)&&v4err>v4threshold?`代理 ${pct(v4err)} > ${pct(v4threshold)}，PPO 禁止宣称收益`:d.cem?'CEM 已完成':'当前目标待 Gate B';
 let watch=d.watchdog||{}, sprintHours=Math.max(0,(Date.UTC(2026,9,3,5,23)-Date.now())/3600000).toFixed(1);$('watchdog').textContent=`两小时阶段验收剩余 ${sprintHours} 小时（北京时间 13:23） · 科研监控采样 ${watch.timestamp_utc||'待启动'} · 告警 ${watch.alerts?.length?watch.alerts.join('、'):'无'}`;$('watchdog').className='small '+(watch.alerts?.length?'bad':'good');
 let finished=d.cfd.filter(x=>x.status==='complete').length,average=d.cfd.reduce((s,x)=>s+x.percent,0)/Math.max(1,d.cfd.length);
 $('cfd-progress').textContent=`${finished}/${d.cfd.length} 配对 CFD 完成`;
 let v4=d.v4_curator||{};$('cfd-sub').textContent=v4.complete?'v4 训练数据 28/4/5 条已完成 Curator 与划分审计':`新增训练数据 Curator ${v4.frames||0}/1602 帧 · HDF5 ${v4.new_hdf5||0}/2 条`;
 let worker=d.resources.worker.at(-1)||{};$('second-seed').textContent=worker.v4_epoch>0||worker.tasks?.some(x=>x.includes('新数据训练'))?`计算节点：v4 真实 CFD 新数据 FNO ${worker.v4_epoch||0}/10 轮；可用统一内存 ${num(worker.mem_available_gib,1)} GiB`:`计算节点：20 步＋后柱阻力加权 ${worker.v3_h20_rear_drag_epoch||0}/10 轮已验收`;
 $('heldout-scope').textContent=strict?`五条冻结 CFD 测试合并误差；逐工况平均 ${pct(strict.macro_total_drag_nrmse)}；仅第 100 步瞬时阻力，窗口均值待测`:audited?`${picked.label}；v3 五条 CFD 测试，原始逐工况平均值` :Number.isFinite(preliminary)?`${picked.label}；v3 五条 CFD 测试初评，完整审计中`:'旧数据四条 CFD 测试；目标 ≤10%';
 $('decision').textContent=audited&&a.status==='GATE_B_PASS'?`${picked.label}已通过冻结的 100 步总阻力门槛（${pct(b)}）；下一步做 CEM 控制筛选，再用真实 CFD 验证。`:audited?`${picked.label}五工况 100 步总阻力误差 ${pct(b)}，未达到 10%；其他候选仍在训练或审计。CEM 与 PPO 暂不启动。`:Number.isFinite(preliminary)?`${picked.label}的 100 步五工况初评为 ${pct(b)}；动作扰动与独立相位审计未完成。CEM/PPO 暂不启动。`:`旧数据模型 100 步误差 ${pct(b)}，未达到 10%；v3 FNO 正在完成独立测试。CEM 与 PPO 暂不启动。`;
 renderLatestEvidence(d);
 resources('primary',d.resources.primary);resources('worker',d.resources.worker);
 renderDualWatchdog(d);
 renderMatchedStart(d);
 renderFull40(d);
 renderTrain20Physics(d);
 renderFreeAR(d);
 renderThreeGoals(d);
 renderH50Figures(d);
 renderTrain16Formal(d);
 renderFull40Chain(d);
 let trainHistory=v4primary.length?v4primary:d.history;$('train-title').textContent=v4primary.length?'v4 新数据 20 步 FNO：训练轮次 → 验证误差':'旧数据多步 FNO：训练轮次 → 验证误差';plot('train-chart',[{values:trainHistory.map(x=>x.terminal_state_mae),color:'#60c9fb'},{values:trainHistory.map(x=>x.terminal_force_mae),color:'#e9ae68'}],.05);
 let steps=['1','10','50','100'],v3series=[{result:d.v3_observed,color:'#60c9fb',label:'主节点单步'},{result:d.v3_worker_observed,color:'#79d5a3',label:'计算节点单步'},{result:d.v3_primary_rollout_observed,color:'#dc95e4',label:'主节点 10 步训练'},{result:d.v3_rollout_observed,color:'#e9ae68',label:'计算节点 10 步训练'},{result:d.v3_h20_observed,color:'#f49ab8',label:'计算节点种子 20 步训练'},{result:d.v3_rear_weighted_observed,color:'#97e1e4',label:'主节点后圆柱加权训练'},{result:d.v3_primary_seed_h20_observed,color:'#dce779',label:'主节点种子 20 步训练'},{result:d.v3_h20_rear_drag_observed,color:'#e66ac7',label:'20 步＋后柱阻力加权'}].filter(x=>x.result?.summary);
 if(v3series.length){let series=v3series.map(x=>({values:steps.map(h=>x.result.summary[h]?.total_drag_nrmse),color:x.color}));let top=Math.max(.2,...series.flatMap(x=>x.values.filter(Number.isFinite)));series.push({values:steps.map(()=>.1),color:'#d77979'});plot('error-chart',series,Math.ceil(top*10)/10);$('error-title').textContent='v3 真实 CFD：递推步数 → 终点阻力误差';$('error-legend').textContent=v3series.map(x=>x.label).join('、')+'；曲线为逐工况平均，红线为10%；正式判定用上方合并误差。'}
 else{plot('error-chart',[{values:steps.map(x=>d.evaluations.heldout?.[x]?.total_drag_nrmse),color:'#60c9fb'},{values:steps.map(x=>d.evaluations.independent?.[x]?.total_drag_nrmse),color:'#79d5a3'},{values:steps.map(()=>.1),color:'#d77979'}],.2);$('error-title').textContent='旧数据多步 FNO：递推步数 → 总阻力误差';$('error-legend').textContent='蓝：旧数据 4 条 CFD 测试；绿：独立初始相位；红：10% 门槛。'}
 let current=v4primary.at(-1)||d.v3_rear_weighted_history.at(-1)||d.v3_primary_rollout_history.at(-1),primaryValidation100=d.v3_primary_rollout_validation?.summary?.['100']?.total_drag_nrmse;$('v3-metrics').textContent=v4primary.length?`v4 多步 FNO 第 ${v4primary.length}/5 轮：验证滚动流场误差 ${num(current.rollout_state_mae,4)}，受力误差 ${num(current.rollout_force_mae,4)}。完整 100 步评估尚未完成，不能据此判断控制效果。`:current?`旧数据 FNO 验证滚动流场误差 ${num(current.rollout_state_mae,4)}；验证滚动受力误差 ${num(current.rollout_force_mae,4)}。${Number.isFinite(primaryValidation100)?`原 10 步模型验证集 100 步总阻力误差 ${pct(primaryValidation100)}。`:''}`:'新数据 FNO：多步训练首轮指标尚未产生。';
 $('infer-speed').textContent=d.benchmark?.status==='FNO_REAL_CFD_INFERENCE_BENCHMARK_OK'?`旧数据多步 FNO、真实 CFD 输入：单步中位 ${num(d.benchmark.step_median_ms,2)} ms；连续 100 步 ${num(d.benchmark.rollout_100_step_seconds,2)} s。仅模型前向，不含 CFD 或控制通信。`:'FNO 推理耗时尚未测量。';
 $('cem').textContent=d.cem?'CEM 控制筛选已完成，结果待审计。':`CEM：等待 FNO 的 100 步总阻力误差降至 10% 以下。新增 CFD 平均求解进度 ${num(average,0)}%。`;
 $('ppo').textContent=d.ppo?'HydroGym PPO 有当前目标的新记录。':'当前总阻力目标的 HydroGym PPO 尚未启动。历史末柱目标的 PPO 曾完成 32 步真实 CFD 闭环，但目标差 +0.003855（更差），不能视为当前控制收益。';figure()}
async function refresh(){try{let r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);const d=await r.json();renderLead(d);renderAdmission(d);renderCurrentFlow(d);render(d);renderActiveExperiment(d)}catch(e){$('clock').textContent='连接失败：'+e.message;$('lead-now').textContent='连接失败，当前页面数值仅是上次采样，不代表实时状态。'}}
for(const id of ['flow-model','flow-profile','flow-step','c-preview-step','c-preview-profile'])$(id).onchange=()=>{if(latest)renderCurrentFlow(latest)};
$('case').onchange=figure;$('horizon').onchange=figure;$('h50-horizon').onchange=()=>{if(latest)renderH50Figures(latest)};window.onresize=()=>{if(latest)render(latest)};refresh();setInterval(refresh,5000);
</script></body></html>'''


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _current_training_log(root: Path, experiment: str = "FC-P003") -> dict:
    """Bounded log evidence, never a replacement for live process verification."""
    runs = {
        "FC-P003": "artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005",
        "FC-P003C": "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005",
    }
    run = root / runs[experiment]
    history = _read_json(run / "training_history.json", [])
    epochs = [row.get("epoch") for row in history if isinstance(row, dict)] if isinstance(history, list) else []
    epochs = [epoch for epoch in epochs if isinstance(epoch, int) and 0 <= epoch <= 2]
    result = {"completed_epochs": max(epochs, default=0), "batch_percent": None,
              "logged_at_utc": None, "source": str(run / "train.log")}
    try:
        with (run / "train.log").open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return result
    matches = re.findall(r"\[([0-9-]+ [0-9:,]+)\]\[train\]\[INFO\] - \[([0-9.]+)%\] Mini-Batch Losses:", tail)
    if matches:
        timestamp, percent = matches[-1]
        try:
            stamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=UTC)
            value = float(percent)
            if math.isfinite(value) and 0 <= value <= 100:
                result.update(batch_percent=value, logged_at_utc=stamp.isoformat())
        except ValueError:
            pass
    return result


def _c_epoch1_preview(root: Path):
    return _c_preview(root, final=False)


def _c_final_preview(root: Path):
    return _c_preview(root, final=True)


def _c_preview(root: Path, *, final: bool):
    """Expose only a complete, hash-bound interim preview; never a gate result."""
    base = root / (C_FINAL_PREVIEW if final else C_EPOCH1_PREVIEW)
    status = ("FCP003C_FINAL_CANDIDATE_FLOW_VISUALIZATION_COMPLETE" if final
              else "FCP003C_INTERIM_EPOCH1_FLOW_VISUALIZATION_PREVIEW_COMPLETE")
    expected_sha = C_FINAL_SHA if final else C_EPOCH1_SHA
    receipt = _read_json(base / "receipt.json", {})
    if not isinstance(receipt, dict):
        return {"ready": False}
    if (receipt.get("status") != status
        or receipt.get("checkpoint_epoch" if final else "interim_epoch") != (2 if final else 1)
        or receipt.get("checkpoint_sha256") != expected_sha
        or any(receipt.get(key) is not False for key in (
            "formal_gate_modified", "ppo_launched", "training_performed", "frozen_test_accessed"))):
        return {"ready": False}
    required = ["evaluation.json", "segments.json"] + [
        f"figures/full40_dynamic_validation_b01_{profile}/horizon_{h}_start_0000.png"
        for profile in (("zero", "minus", "plus") if final else ("plus",))
        for h in ("001", "010", "050", "100")
    ]
    hashes = receipt.get("output_sha256", {})
    if not isinstance(hashes, dict):
        return {"ready": False}
    try:
        for relative in required:
            path = base / relative
            if (not path.resolve().is_relative_to(base.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(relative)):
                return {"ready": False}
    except OSError:
        return {"ready": False}
    return {"ready": True, "checkpoint_epoch": 2 if final else 1,
            "interim_epoch": None if final else 1, "checkpoint_sha256": expected_sha}


def _research_overview(root: Path):
    """Read current experiment evidence only; absent worker output is unknown."""
    result = {}
    for candidate in ("lambda0", "lambda10"):
        relative = (
            f"artifacts/tandem_fno_paired_stats_{candidate}_20261004/"
            "posteval_fc_p001/validation10/endpoint_gate.json"
        )
        path = root / relative
        payload = _read_json(path, None)
        updated = None
        if isinstance(payload, dict):
            try:
                updated = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            except OSError:
                payload = None
        result[candidate] = {
            "endpoint": payload,
            "path": relative,
            "updated_at": updated,
            "development": _read_json(path.parent.parent / "development_gate.json", None),
            "dynamic_summary": (_read_json(path.parent.parent / "dynamic6/evaluation.json", {}) or {}).get("summary", {}),
            "figures": [
                f"{profile}/{horizon}"
                for profile in ("zero", "minus", "plus")
                for horizon in ("001", "010", "050", "100")
                if (root / "artifacts" / f"paired_{candidate}_flow_visualization_20261004"
                    / "figures" / f"full40_dynamic_validation_b01_{profile}"
                    / f"horizon_{horizon}_start_0000.png").is_file()
            ],
        }
    result["fc_p003"] = _completed_interleaved_candidate(root)
    result["fc_p003b"] = _completed_interleaved_candidate(root, dynamic=True)
    result["fc_p003c"] = _completed_interleaved_candidate(root, true_state=True)
    return result


def _completed_interleaved_candidate(root: Path, dynamic: bool = False, *, true_state: bool = False) -> dict | None:
    """Display finalized candidate evidence only when bound to its receipt."""
    base = root / "artifacts/tandem_fno_paired_stats_interleaved_lambda10_20261005/posteval_fc_p003"
    expected_status = "FC_P003_POSTEVAL_COMPLETE"
    if dynamic:
        base = root / "artifacts/tandem_fno_dynamic_paired_interleaved_lambda10_20261005/posteval_fc_p003b"
        expected_status = "FC_P003B_POSTEVAL_COMPLETE"
    if true_state:
        base = root / "artifacts/tandem_fno_true_state_paired_step_lambda10_20261005/posteval_fc_p003c"
        expected_status = "FC_P003C_POSTEVAL_COMPLETE"
    receipt = _read_json(base / "receipt.json", None)
    if not isinstance(receipt, dict) or receipt.get("status") != expected_status:
        return None
    if dynamic and (receipt.get("candidate_kind") != "dynamic_paired_interleaved_lambda10"
                    or receipt.get("frozen_test_accessed") is not False):
        return None
    if true_state and (receipt.get("candidate_kind") != "true_state_paired_step_lambda10"
                       or receipt.get("checkpoint_sha256") != C_FINAL_SHA
                       or receipt.get("frozen_test_accessed") is not False
                       or receipt.get("ppo_auto_launched") is not False):
        return None
    checkpoint = receipt.get("checkpoint_sha256")
    hashes = receipt.get("sha256")
    if not isinstance(checkpoint, str) or len(checkpoint) != 64 or not isinstance(hashes, dict):
        return None
    values = {}
    for key, relative in (("endpoint", "validation10/endpoint_gate.json"),
                          ("development", "development_gate.json")):
        try:
            raw = (base / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != hashes.get(relative):
                return None
            value = json.loads(raw)
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict) or value.get("checkpoint_sha256") != checkpoint:
            return None
        values[key] = value
    gate_status = values["development"].get("status")
    if gate_status not in {"DYNAMIC_FNO_DEVELOPMENT_ADMISSION_PASS", "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL"}:
        return None
    if not dynamic and not true_state and gate_status != receipt.get("development_gate_status"):
        return None
    if true_state:
        try:
            raw = (base / "dynamic6/evaluation.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != hashes.get("dynamic6/evaluation.json"):
                return None
            values["dynamic_summary"] = json.loads(raw)["summary"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return {**values, "receipt_bound": True, "path": str(base.relative_to(root))}


def _dual_node_watchdog(root: Path):
    payload = _read_json(root / DUAL_NODE_WATCHDOG, None)
    if not isinstance(payload, dict) or payload.get("status") not in {
        "MONITORING",
        "ALERT",
    }:
        return None
    return payload


FCP009_JOINT_SHA = "931fcd2ddd6901ddfbb3ecdbfe9774d7b1e5fe6d17479c87c44ccaf51ff2b0bc"


def _parse_fcp011_live(output: str, now: float) -> dict:
    state, pid, command, latest = "unknown", 0, "", None
    lines = output.splitlines()
    for line in lines:
        if line.startswith("ActiveState="):
            state = line.partition("=")[2]
        elif line.startswith("MainPID="):
            try:
                pid = int(line.partition("=")[2])
            except ValueError:
                pid = 0
        elif line.startswith("ExecStart="):
            command = line.partition("=")[2]
    for line in lines:
        try:
            record = json.loads(line)
            if pid and str(record.get("_PID")) != str(pid):
                continue
            message = json.loads(record["MESSAGE"])
            step = message.get("step")
            loss = float(message["total_loss"])
            identity = message["identity"]
            stamp = int(record["__REALTIME_TIMESTAMP"]) / 1e6
            if type(step) is not int or not 1 <= step <= 1368 or not math.isfinite(loss):
                continue
            if identity.get("split") != "train" or identity.get("rollout_steps") != 100:
                continue
            if not math.isfinite(stamp) or stamp > now + 5:
                continue
            if latest is None or stamp > latest["timestamp"]:
                latest = {"step": step, "total_loss": loss, "timestamp": stamp}
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    bound = "train_fcp011_decoder_scope.py" in command and "--resource-probe" not in command
    running = state == "active" and pid > 0 and bound
    return {"service_state": state, "pid": pid, "running": running,
            "step": latest["step"] if latest else None,
            "total_loss": latest["total_loss"] if latest else None,
            "progress_fresh": bool(latest and 0 <= now - latest["timestamp"] <= 300),
            "progress_age_seconds": max(0, now - latest["timestamp"]) if latest else None,
            "admission": False}


def _fcp011_formal_result(base: Path, scope: str) -> dict:
    expected = {
        "head-only": "256a65c7d6b2cb7cdffc96c6a4de33117f4aee8fd3bda9e8ea72c9477a452a07",
        "decoder-tail": "e6c0a1712894bc4c6060258ee8903de1787b54f71fa72411a60ba2d4f0ecebcc",
    }
    try:
        base = base / "posteval_fc_p011"
        raw = (base / "receipt.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[scope]:
            return {"verified_fail": False}
        receipt = json.loads(raw)
        raw = (base / "development_gate.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != receipt["sha256"]["development_gate.json"]:
            return {"verified_fail": False}
        gate = json.loads(raw)
        if gate.get("status") != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL" or gate.get("checkpoint_sha256") != receipt["checkpoint_sha256"]:
            return {"verified_fail": False}
        branches = gate["window_gate"]["branches"]
        if len(branches) != 6:
            return {"verified_fail": False}
        return {"verified_fail": True, "admission": False,
                "cd_pass": sum(b["metric_pass"]["total_cd"] is True for b in branches),
                "rms_pass": sum(b["metric_pass"]["rear_cl_fluctuation_rms"] is True for b in branches),
                "mean_pass": sum(b["metric_pass"]["rear_cl_mean"] is True for b in branches)}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified_fail": False}


def _fcp011_terminal(root: Path, scope: str) -> dict:
    identities = {
        "head-only": ("fcp011_head_only_training_20261005", "6d3ecb9174f2adb20d5b95989d1438a0ba357b7ed1caeb03f9225da78d77b149", "1c0bb91bfee3ca33f0ca5060241b6992aca7b5c5be5a0adcc7c7714e05938960"),
        "decoder-tail": ("fcp011_decoder_tail_training_worker_20261005", "499b3b6c65af9a8a53318f4d8dfe771b0b0cea919c66879f520c978887f5c686", "d05f1d0713b424f07f558d364b9c43bc36852c7b39881ca0d8295d6444c6873e"),
    }
    try:
        directory, receipt_sha, result_sha = identities[scope]
        base = root / "artifacts" / directory
        receipt_raw = (base / "completion_receipt.json").read_bytes()
        result_raw = (base / "result.json").read_bytes()
        if hashlib.sha256(receipt_raw).hexdigest() != receipt_sha or hashlib.sha256(result_raw).hexdigest() != result_sha:
            return {"verified": False}
        result = json.loads(result_raw)
        means = []
        for step in ("0", "1368"):
            rows = result["train_only_diagnostics"][step]["windows"]
            values = [float(row["tail62_rear_cl_rms_absolute_error"]) for row in rows]
            if len(values) != 6 or not all(math.isfinite(value) and value >= 0 for value in values):
                return {"verified": False}
            means.append(sum(values) / len(values))
        return {"verified": True, "rms_before": means[0], "rms_after": means[1], "admission": False,
                "formal_result": _fcp011_formal_result(base, scope)}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified": False}


def _parse_fcp011_formal(output: str, scope: str) -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    try:
        pid = int(fields.get("MainPID", "0"))
    except ValueError:
        pid = 0
    state = fields.get("ActiveState", "unknown")
    command = fields.get("ExecStart", "")
    bound = f"FCP_POSTEVAL_PROFILE=p011_{scope.replace('-', '_')} " in command and "run_fcp008_posteval_spark.sh --execute" in command
    return {"service_state": state, "pid": pid, "running": state == "active" and pid > 0 and bound, "admission": False}


FCP013_TRAINING_APPROVAL_SHA = "1bdcfcf71d1581bbae66fc6551dde61a500bd95a464d9f61d510cbac1721a120"


def _parse_fcp013_live(output: str, now: float, unit: str = "fluid-control-fcp013-training-20261005.service") -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line and not line.startswith("{"))
    state = fields.get("ActiveState", "unknown")
    try:
        pid = int(fields.get("MainPID", "0"))
    except ValueError:
        pid = 0
    command = fields.get("ExecStart", "")
    invocation = fields.get("InvocationID", "")
    bound = any(name in command for name in ("run_fcp013_training_spark.sh", "run_fcp013_training_recovery_spark.sh", "train_fcp013_independent_force_fno.py")) and "--resource-probe" not in command
    latest = None
    for line in output.splitlines():
        try:
            record = json.loads(line)
            if invocation:
                if record.get("_SYSTEMD_INVOCATION_ID") != invocation or record.get("_SYSTEMD_USER_UNIT") != unit:
                    continue
            elif pid and str(record.get("_PID")) != str(pid):
                continue
            row = json.loads(record["MESSAGE"])
            step = row["step"]
            values = {key: float(row[key]) for key in ("total", "h1_balanced", "ar_balanced")}
            stamp = int(record["__REALTIME_TIMESTAMP"]) / 1e6
            if type(step) is not int or not 1 <= step <= 1368 or not all(math.isfinite(value) and value >= 0 for value in values.values()):
                continue
            if row["identity"].get("split") != "train" or row["identity"].get("rollout_steps") != 100:
                continue
            if not math.isfinite(stamp) or stamp > now + 5:
                continue
            if latest is None or stamp > latest["timestamp"]:
                latest = {"step": step, **values, "timestamp": stamp}
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    running = state == "active" and pid > 0 and bound
    return {"service_state": state, "pid": pid, "running": running,
            "step": latest["step"] if latest else None,
            **{key: latest[key] if latest else None for key in ("total", "h1_balanced", "ar_balanced")},
            "progress_fresh": bool(running and latest and 0 <= now - latest["timestamp"] <= 300),
            "admission": False}


FCP013_TERMINAL_FILES = {
    "completion_receipt.json": "3c53a7fb94d5ad5f29bed6522317389f02842d078d157d63e20b90b6948e2d90",
    "candidate_audit.json": "1c280b291ae7ded1f8e63ccc46ba40a7e085e7d7b6fa69f0dddff18ffac704a7",
    "fixed_six_diagnostics/result.json": "8e0255c955c1b26fdff240a0854fc0a92d3bd247cc38ed6c268fc1d397cec873",
}


def _parse_fcp013_posteval_live(output: str) -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    try:
        pid = int(fields.get("MainPID", "0"))
    except ValueError:
        pid = 0
    command = fields.get("ExecStart", "")
    bound = (fields.get("InvocationID") == "7235b2f06282435a89b84964e384c60f"
             and "FCP_POSTEVAL_PROFILE=p013" in command
             and "p013_posteval_chain_f95048c3a786_immutable/scripts/run_fcp008_posteval_spark.sh --execute" in command)
    return {"state": fields.get("ActiveState", "unknown"),
            "running": bool(bound and fields.get("ActiveState") == "active"
                            and fields.get("SubState") == "running" and pid > 0),
            "admission": False}


def _fcp013_terminal_progress(root: Path) -> dict:
    base = root / "artifacts/fcp013_independent_force_fno_training_r2_20261005"
    try:
        evidence = {}
        for name, digest in FCP013_TERMINAL_FILES.items():
            raw = (base / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                return {"verified": False}
            evidence[name] = json.loads(raw)
        completion, audit = evidence["completion_receipt.json"], evidence["candidate_audit.json"]
        result = evidence["fixed_six_diagnostics/result.json"]
        if (completion.get("status") != "FC_P013_TRAINING_COMPLETE_NOT_ADMISSION"
                or completion.get("candidate_audit_sha256") != FCP013_TERMINAL_FILES["candidate_audit.json"]
                or completion.get("dual_manifest_sha256") != audit.get("dual_manifest_sha256")
                or result.get("dual_manifest_sha256") != audit.get("dual_manifest_sha256")
                or result.get("status") != "FC_P013_FIXED_TRAIN_DIAGNOSTICS_COMPLETE_NOT_ADMISSION"
                or result.get("tensor_sha256_before") != result.get("tensor_sha256_after")):
            return {"verified": False}
        parent = result["panels"]["p009_parent"]["windows"]
        candidate = result["panels"]["p013_terminal"]["windows"]
        if len(parent) != 6 or len(candidate) != 6 or any(a["identity"] != b["identity"] for a, b in zip(parent, candidate)):
            return {"verified": False}
        rows = [{"h1_parent": a["true_state_h1_force_mae"][3],
                 "h1_candidate": b["true_state_h1_force_mae"][3],
                 "ar_parent": a["free_ar_force_mae"][3],
                 "ar_candidate": b["free_ar_force_mae"][3]} for a, b in zip(parent, candidate)]
        formal = subprocess.run(["systemctl", "--user", "show", "fluid-control-fcp013-posteval-r2-20261005.service",
                                 "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "ExecStart", "-p", "InvocationID"],
                                capture_output=True, text=True, timeout=2, check=False)
        return {"verified": True, "rows": rows,
                "h1_regressions": sum(r["h1_candidate"] > r["h1_parent"] for r in rows),
                "ar_regressions": sum(r["ar_candidate"] > r["ar_parent"] for r in rows),
                "fields_unchanged": all(a[k] == b[k] for a, b in zip(parent, candidate)
                                        for k in ("true_state_h1_field_relative_l2_uvp", "free_ar_field_relative_l2_uvp")),
                "formal": _parse_fcp013_posteval_live(formal.stdout), "admission": False}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        return {"verified": False}


FCP016_INVOCATION = "a95370a65c2b47e0b0e2926261937e33"
FCP016_APPROVAL = "f61980e2b98bfa5eb84a0a6d8e93b90924632802e28d1e9d79fbe605b712af53"
FCP016_LAUNCHER = "artifacts/fcp016_fixed_panel_source_20261005_immutable/scripts/run_fcp016_fixed_panel_fit_spark.sh"


def _parse_fcp016_live(output: str, log: str, log_age: float, result: dict | None = None) -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if (fields.get("InvocationID") != FCP016_INVOCATION
            or FCP016_LAUNCHER + " --execute" not in fields.get("ExecStart", "")
            or "FCP016_APPROVAL_SHA256=" + FCP016_APPROVAL not in fields.get("ExecStart", "")):
        return {"verified": False, "admission": False}
    try:
        pid = int(fields.get("MainPID", ""))
    except ValueError:
        return {"verified": False, "admission": False}
    progress = None
    for line in log.splitlines():
        try:
            row = json.loads(line)
            if (row.get("event") == "panel_window" and type(row.get("update")) is int
                    and row["update"] in (0, 8, 16, 32)
                    and row.get("global_index") in (160, 816, 923, 975, 1077, 1233)):
                progress = max(progress or 0, row["update"])
        except (ValueError, TypeError, AttributeError):
            continue
    running = fields.get("ActiveState") == "active" and fields.get("SubState") == "running" and pid > 0
    exited = (fields.get("ActiveState") == "active" and fields.get("SubState") == "exited" and pid == 0
              and fields.get("Result") == "success" and fields.get("ExecMainCode") == "1"
              and fields.get("ExecMainStatus") == "0")
    result = result or {}
    terminal = (exited and result.get("status") == "FC_P016_FIXED_PANEL_FIT_COMPLETE_NOT_ADMISSION"
                and result.get("training_experiment") == "FC-P016" and result.get("optimizer_steps") == 32
                and result.get("probe_sha256") == "18b210077a93bbd21a327ae6578a73fb371bf0d2f48d0541f6ca8bd87aa42bbb"
                and result.get("candidate_saved") is False and result.get("validation_accessed") is False
                and result.get("frozen_test_accessed") is False and result.get("ppo_executed") is False
                and [x.get("update") for x in result.get("panels", [])] == [0,8,16,32])
    return {"verified": True, "running": running, "terminal": terminal,
            "minimum_completed_updates": progress, "progress_fresh": 0 <= log_age <= 300,
            "log_age_seconds": log_age, "admission": False, "independent_terminal_audit": False}


def _fcp016_probe(root: Path) -> dict:
    base = root / "artifacts/fcp016_fixed_panel_fit_20261005"
    try:
        for path, expected in {
            root / "docs/FC_P016_RUNNING_EXECUTION_20261005.json": "55711bedfcb27df86f31bdb2785f3206288a8332c2cc8d79dc06f013c1f2b630",
            base / "execution_approval.json": FCP016_APPROVAL,
            base / "immutable_launcher.sh": "496d7cc91598c3f440aa1c12f180409b47b4c371ce7410f1bba7f09cb7acfbe9",
            root / FCP016_LAUNCHER: "496d7cc91598c3f440aa1c12f180409b47b4c371ce7410f1bba7f09cb7acfbe9",
        }.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return {"verified": False}
        state = subprocess.run(["systemctl", "--user", "show", "fluid-control-fcp016-fixed-panel-fit-20261005.service",
            "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "InvocationID", "-p", "ExecStart",
            "-p", "Result", "-p", "ExecMainCode", "-p", "ExecMainStatus"],capture_output=True,text=True,timeout=2,check=False)
        if state.returncode != 0: return {"verified": False}
        result = _read_json(base / "result.json", {})
        return {**_parse_fcp016_live(state.stdout, _tail_text(base / "run.log"),
                  time.time() - (base / "run.log").stat().st_mtime, result),
                "sampled_at_utc": datetime.now(UTC).isoformat()}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError):
        return {"verified": False}


def _fcp015_formal_result(root: Path) -> dict:
    base = root / "artifacts/fcp015_window_accumulation_training_20261005/posteval_fc_p015"
    try:
        raw = (base / "receipt.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != "353004afa2aa5a35b912205751a100bc6d37d0572ce5431108dca3ed2ea95a5b":
            return {"verified": False}
        gate_raw = (base / "development_gate.json").read_bytes()
        if hashlib.sha256(gate_raw).hexdigest() != json.loads(raw)["sha256"]["development_gate.json"]:
            return {"verified": False}
        gate = json.loads(gate_raw)
        branches = gate["window_gate"]["branches"]
        if gate["status"] != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL" or len(branches) != 6:
            return {"verified": False}
        return {"verified": True, "admission": False, "joint_pass":sum(x["joint_pass"] is True for x in branches),
                **{label:sum(x["metric_pass"][key] is True for x in branches) for label,key in
                   (("cd_pass","total_cd"),("rms_pass","rear_cl_fluctuation_rms"),("mean_pass","rear_cl_mean"))}}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified": False}


FCP018_APPROVAL = "eea5bd5c6a3fe585ae1104600421e50b335f61299c4014c5e3136722af3d4d39"
FCP018_PROTOCOL = "310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d"
FCP018_LAUNCHER = "artifacts/fcp018_reduced_rate_source_20261005_immutable/scripts/run_fcp018_reduced_rate_spark.sh"


def _parse_fcp023_live(state: dict, log: str, process_matches: bool) -> dict:
    if state.get("InvocationID") != "39aec740a9914226bb1f74c2d29e7917":
        return {"verified": False}
    pid = state.get("MainPID", "0")
    if pid != "0" and not process_matches:
        return {"verified": False}
    running = state.get("ActiveState") == "active" and state.get("SubState") == "running" and pid != "0" and process_matches
    terminal = (state.get("ActiveState") == "active" and state.get("SubState") == "exited"
                and pid == "0" and state.get("Result") == "success"
                and state.get("ExecMainCode") == "1" and state.get("ExecMainStatus") == "0")
    updates = {"LOW": set(), "HIGH": set()}
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("event") != "arm_update_complete":
            continue
        arm, step = row.get("arm"), row.get("update")
        if arm not in updates or type(step) is not int or not 1 <= step <= 16 or step in updates[arm]:
            return {"verified": False}
        if arm == "HIGH" and len(updates["LOW"]) != 16:
            return {"verified": False}
        updates[arm].add(step)
    if any(steps != set(range(1, len(steps)+1)) for steps in updates.values()):
        return {"verified": False}
    return {"verified": True, "running": running, "exited_success": terminal,
            "updates": {key: len(value) for key, value in updates.items()}, "admission": False}


def _parse_registered_progress(state: dict, log: str, matches: bool, registration: dict) -> dict:
    if state.get("InvocationID") != registration["invocation"]:
        return {"verified": False}
    pid = state.get("MainPID", "0")
    if pid != "0" and not matches:
        return {"verified": False}
    plan = registration["planned_updates"]
    if not isinstance(plan, dict) or not plan or any(type(n) is not int or not 1 <= n <= 10000 for n in plan.values()):
        return {"verified": False}
    counts = {key: set() for key in plan}
    kind = registration.get("progress_kind", "optimizer_updates")
    p029_modes = {"p029_scales": "scales", "p029_probe": "resource-probe", "p029_train": "train"}
    if kind in p029_modes:
        target = 1 if kind == "p029_probe" else 171
        if plan != {"FNO": target}:
            return {"verified": False}
        seen, groups = set(), set()
        for line in log.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") not in ("window_complete", "group_complete"):
                continue
            if row.get("mode") != p029_modes[kind]:
                return {"verified": False}
            if row["event"] == "window_complete":
                case, start, dataset = row.get("case"), row.get("start"), row.get("dataset_index")
                if not isinstance(case, str) or not case or type(start) is not int or start < 0 or type(dataset) is not int or dataset not in (0, 1, 2) or row.get("split") != "train" or row.get("rollout_steps") != 100:
                    return {"verified": False}
                key = (dataset, case, start)
                if key in seen or len(seen) >= (1 if kind == "p029_probe" else 1368):
                    return {"verified": False}
                seen.add(key)
            else:
                group = row.get("group")
                if kind == "p029_probe" or type(group) is not int or group != len(groups) + 1 or group > 171 or len(seen) < 8 * group:
                    return {"verified": False}
                groups.add(group)
        running = (state.get("ActiveState"), state.get("SubState")) in (("active", "running"), ("activating", "start")) and pid != "0" and matches
        terminal = state.get("ActiveState") == "active" and state.get("SubState") == "exited" and pid == "0" and state.get("Result") == "success" and state.get("ExecMainCode") == "1" and state.get("ExecMainStatus") == "0"
        return {"verified": True, "running": running, "exited_success": terminal,
                "updates": {"FNO": len(seen) if kind == "p029_probe" else len(groups)}, "planned_updates": plan,
                "windows": {"FNO": len(seen)}, "label": registration["label"], "description": registration["description"],
                "progress_unit": "次参数更新" if kind == "p029_train" else "组计算（不更新参数）", "admission": False}
    if kind not in ("optimizer_updates", "resource_arms", "history_training", "flow_training", "diagnostic_origins"):
        return {"verified": False}
    if kind == "flow_training" and plan != {"FNO": 171}:
        return {"verified": False}
    if kind == "diagnostic_origins" and plan != {"CFD": 44}:
        return {"verified": False}
    cases = set()
    windows = {key: set() for key in plan}
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if kind == "flow_training" and isinstance(row, dict) and row.get("event") == "training_window_complete":
            case, start, dataset = row.get("case"), row.get("start"), row.get("dataset_index")
            if not isinstance(case, str) or not case or type(start) is not int or start < 0 or type(dataset) is not int or dataset not in (0, 1, 2) or row.get("split") != "train":
                return {"verified": False}
            key = (dataset, case, start)
            if key in windows["FNO"] or len(windows["FNO"]) >= 1368:
                return {"verified": False}
            windows["FNO"].add(key)
            continue
        if kind == "history_training" and isinstance(row, dict) and row.get("event") == "training_window_complete":
            k, consumed = row.get("history_k"), row.get("consumed")
            arm = "K" + str(k)
            if type(k) is not int or k not in (1, 4) or arm not in plan or type(consumed) is not int or not 1 <= consumed <= plan[arm] * 8 or consumed in windows[arm]:
                return {"verified": False}
            windows[arm].add(consumed)
            continue
        event = {"resource_arms": "arm_complete", "history_training": "accumulation_update_complete", "flow_training": "accumulation_update_complete", "diagnostic_origins": "origin_complete"}.get(kind, "arm_update_complete")
        if not isinstance(row, dict) or row.get("event") != event:
            continue
        if kind == "diagnostic_origins":
            case = row.get("case")
            if not isinstance(case, str) or not case or case in cases:
                return {"verified": False}
            cases.add(case)
            arm, step = "CFD", row.get("count")
        elif kind == "flow_training":
            arm, step = "FNO", row.get("update")
        elif kind == "resource_arms":
            if type(row.get("k")) is not int or row["k"] not in (1, 4):
                return {"verified": False}
            arm, step = "K" + str(row["k"]), 1
        elif kind == "history_training":
            k = row.get("history_k")
            if type(k) is not int or k not in (1, 4):
                return {"verified": False}
            arm, step = "K" + str(k), row.get("update")
        else:
            arm, step = row.get("arm"), row.get("update")
        if arm not in plan or type(step) is not int or not 1 <= step <= plan[arm] or step in counts[arm]:
            return {"verified": False}
        counts[arm].add(step)
    if any(steps != set(range(1, len(steps)+1)) for steps in counts.values()):
        return {"verified": False}
    if kind == "history_training" and any(windows[k] != set(range(1, len(windows[k]) + 1)) or len(windows[k]) < 8 * len(counts[k]) for k in plan):
        return {"verified": False}
    if kind == "flow_training" and len(windows["FNO"]) < 8 * len(counts["FNO"]):
        return {"verified": False}
    running = (state.get("ActiveState"), state.get("SubState")) in (("active", "running"), ("activating", "start")) and pid != "0" and matches
    terminal = state.get("ActiveState") == "active" and state.get("SubState") == "exited" and pid == "0" and state.get("Result") == "success" and state.get("ExecMainCode") == "1" and state.get("ExecMainStatus") == "0"
    return {"verified": True, "running": running, "exited_success": terminal,
            "updates": {k: len(v) for k, v in counts.items()}, "planned_updates": plan,
            "label": registration["label"], "description": registration["description"],
            **({"windows": {k: len(v) for k, v in windows.items()}} if kind in ("history_training", "flow_training") else {}),
            "progress_unit": "个诊断工况（无参数更新）" if kind == "diagnostic_origins" else "项无更新计算" if kind == "resource_arms" else "次更新", "admission": False}


def _registered_terminal_review(root: Path, registration: dict, progress: dict) -> dict:
    """Display a bound engineering/rejection review, never infer admission."""
    try:
        review = registration["review"]
        if not progress.get("verified") or not progress.get("exited_success"):
            return {"verified": False}
        if progress["updates"] != progress["planned_updates"]:
            return {"verified": False}
        payloads = {}
        for key in ("report", "result"):
            path = Path(review[key])
            resolved = (root / path).resolve()
            if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
                return {"verified": False}
            raw = resolved.read_bytes()
            if hashlib.sha256(raw).hexdigest() != review[key + "_sha256"]:
                return {"verified": False}
            payloads[key] = raw
        result = json.loads(payloads["result"])
        if review.get("kind") == "p027_diagnostic":
            rows = result.get("rows", [])
            if (result.get("status") != "P027_OFFLINE_DIAGNOSTIC_COMPLETE_NOT_ADMISSION"
                    or len(rows) != 44 or len({r["case"] for r in rows}) != 44
                    or result.get("flow_transitions") != 440
                    or result.get("aerodynamic_state_evaluations") != 1760
                    or any(result.get(k) is not False for k in (
                        "scientific_admission", "optimizer_created", "model_saved",
                        "validation_accessed", "frozen_test_accessed"))):
                return {"verified": False}
            return {"verified": True, "diagnostic_completed": True, "admission": False,
                    "summary": review["summary"], "next_action": review["next_action"]}
        if review.get("kind") == "history_resource":
            if (result.get("status") != "FC_P026_HISTORY_RESOURCE_COMPLETE_NOT_ADMISSION"
                    or result.get("optimizer_steps") != 0
                    or result.get("candidate_saved") is not False
                    or result.get("scientific_admission") is not False
                    or [arm.get("k") for arm in result.get("arms", [])] != [1, 4]):
                return {"verified": False}
            return {"verified": True, "engineering_pass": True, "admission": False,
                    "summary": review["summary"], "next_action": review["next_action"]}
        if result["comparison"]["local_support"] is not False:
            return {"verified": False}
        return {"verified": True, "local_support": False, "admission": False,
                "summary": review["summary"], "next_action": review["next_action"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified": False}


FORMAL_STAGE_LABELS = {
    "validation10": "10个验证工况：流场与受力预测",
    "validation_diagnostic": "核对10个验证工况的误差统计",
    "endpoint_gate": "检查长时预测的误差指标",
    "dynamic6": "6个动态控制工况：连续流场预测",
    "dynamic_diagnostic": "核对动态工况的误差统计",
    "force_window": "检查阻力、平均升力与升力波动",
    "development_gate": "汇总控制相关预测指标",
}


def _registered_formal_progress(root: Path, registration: dict, state: dict, matches: bool) -> dict:
    """Display actual completed commands, never interpret them as metric passes."""
    def confined(name):
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
            raise ValueError("formal progress path outside project")
        return path

    if state.get("InvocationID") != registration["invocation"]:
        return {"verified": False}
    pid = state.get("MainPID", "0")
    if pid != "0" and not matches:
        return {"verified": False}
    approval_raw = confined(registration["approval"]).read_bytes()
    if hashlib.sha256(approval_raw).hexdigest() != registration["approval_sha256"]:
        raise ValueError("formal approval differs")
    approval = json.loads(approval_raw)
    if (approval.get("formal_evaluation_authorized") is not True
            or approval.get("ppo_auto_launch") is not False):
        raise ValueError("not an approved formal evaluation")
    if approval.get("status") == "FC_P026_APPROVED_ORIGINAL_FORMAL_EVALUATION":
        if type(approval.get("history_k")) is not int or approval["history_k"] not in (1, 4):
            raise ValueError("not an approved formal evaluation")
        arm = f"K{approval['history_k']}"
    elif approval.get("status") in ("FC_P028_APPROVED_ORIGINAL_FORMAL_EVALUATION", "FC_P029_APPROVED_ORIGINAL_FORMAL_EVALUATION"):
        if (approval.get("reviewed_by_lead") is not True
                or approval.get("independent_terminal_audit", {}).get("reviewed_by_lead") is not True
                or approval.get("official_dual_reload", {}).get("reviewed_by_lead") is not True):
            raise ValueError("not an approved formal evaluation")
        arm = {"FC_P028_APPROVED_ORIGINAL_FORMAL_EVALUATION": "P028",
               "FC_P029_APPROVED_ORIGINAL_FORMAL_EVALUATION": "P029"}[approval["status"]]
    else:
        raise ValueError("not an approved formal evaluation")
    if registration["planned_updates"] != {arm: len(FORMAL_STAGE_LABELS)}:
        raise ValueError("formal step plan differs")
    output = confined(approval["output_relative_directory"])
    if hashlib.sha256((output / "evidence/formal_approval.json").read_bytes()).hexdigest() != registration["approval_sha256"]:
        raise ValueError("actual formal approval copy differs")
    completed = 0
    for index, name in enumerate(FORMAL_STAGE_LABELS):
        terminal_path = output / "evidence" / f"{name}_container_terminal.json"
        if not terminal_path.exists():
            continue
        if completed != index:
            raise ValueError("formal step completion is not contiguous")
        initial = json.loads((output / "evidence" / f"{name}_container.json").read_text())
        terminal = json.loads(terminal_path.read_text())
        if not initial.get("Id") or initial["Id"] != terminal.get("Id"):
            raise ValueError("formal container identity differs")
        for proof in (initial, terminal):
            if (proof.get("Image") != approval["official_image_id"]
                    or not any(m.get("Source") == str(output)
                               and m.get("Destination") == "/workspace/output"
                               and m.get("RW") is True for m in proof.get("Mounts", []))):
                raise ValueError("formal container output/image differs")
        status = terminal["State"]
        if (status.get("Status") != "exited" or status.get("Running") is not False
                or status.get("OOMKilled") is not False
                or type(status.get("ExitCode")) is not int or status["ExitCode"] != 0):
            raise ValueError("formal step did not exit successfully")
        completed += 1
    running = ((state.get("ActiveState"), state.get("SubState")) in
               (("active", "running"), ("activating", "start")) and pid != "0" and matches)
    exited = (state.get("ActiveState") == "active" and state.get("SubState") == "exited"
              and pid == "0" and state.get("Result") == "success"
              and state.get("ExecMainCode") == "1" and state.get("ExecMainStatus") == "0")
    if exited and completed != len(FORMAL_STAGE_LABELS):
        raise ValueError("formal unit ended without all step evidence")
    current = None
    lines = confined(registration["log"]).read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index == len(lines) - 1:
                continue  # A concurrent writer can leave its final line incomplete.
            raise
        if not isinstance(row, dict):
            raise ValueError("invalid formal memory sample")
        current = row
    detail = "计算结束，预测精度结果待复核" if exited else "阶段采样待更新"
    if running and current:
        sampled = current.get("time_unix")
        age = datetime.now(UTC).timestamp() - sampled if type(sampled) in (int, float) else float("inf")
        if 0 <= age < 60:
            step = current.get("step")
            if step not in FORMAL_STAGE_LABELS and step != "precision":
                raise ValueError("unknown formal step")
            detail = "当前：" + FORMAL_STAGE_LABELS.get(step, "检查数值计算设置")
    return {"verified": True, "running": running, "exited_success": exited,
            "updates": {arm: completed}, "planned_updates": registration["planned_updates"],
            "label": registration["label"], "description": registration["description"],
            "progress_unit": "项评估步骤", "progress_detail": detail, "admission": False}


def _two_segment_bridge(root: Path, registration: dict, state: dict) -> dict:
    """Exact completed CPU bridge evidence, never a model/control admission."""
    try:
        if (registration["unit"] != "fluid-control-two-segment-frame-cpu-r4-20261006.service"
                or registration["invocation"] != "99e5c019c08240668241f7ac036320f3"
                or state.get("InvocationID") != registration["invocation"]
                or any(state.get(k) != v for k, v in {
                    "MainPID":"0", "ActiveState":"active", "SubState":"exited",
                    "Result":"success", "ExecMainCode":"1", "ExecMainStatus":"0"}.items())):
            return {"verified": False}
        base = root / "artifacts/online_two_segment_cpu_20261006_r4"
        files = {
            root / "docs/TWO_SEGMENT_CURRENT_FRAME_R4_APPROVAL_20261006.json": "9e268e48f2201eeefbbcb67fefcd3f3184d82877339a429b3f847a2134a9ddd6",
            base / "result.json": "be3c57e00003d7092b116058604a47d2ea2b2c1f033551adb39188f1c91f7584",
            base / "cleanup.json": "176c4419be678560ac81af6c1b98b88649ac95f7b739f8a441cd1cb24459bdee",
            base / "container_terminal.json": "4f9948cf37959af9d9a50c32d2a924cf139561116d55c384deca87f84ea24229",
        }
        docs = {}
        for path, expected in files.items():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                return {"verified":False}
            docs[path.name] = json.loads(raw)
        result, cleanup, terminal = docs["result.json"], docs["cleanup.json"], docs["container_terminal.json"]
        rows = result.get("rows", [])
        if (result.get("status") != "TWO_SEGMENT_ENGINEERING_ONLY_NOT_CONTROL_ADMISSION"
                or any(result.get(k) is not False for k in ("model_loaded", "policy_loaded", "scientific_admission"))
                or result.get("source_unchanged") is not True
                or [(r.get("step"),r.get("start"),r.get("end")) for r in rows] != [(1,148.0,148.1),(2,148.1,148.2)]
                or any(r.get("applied_now") != 0 or r.get("applied_next") != 0
                       or r.get("health",{}).get("solver_ended_cleanly") is not True
                       or r.get("health",{}).get("steps") != 20 for r in rows)
                or cleanup.get("errors") != [] or cleanup.get("cid") != terminal.get("Id")
                or terminal.get("State",{}).get("Running") is not False
                or terminal.get("State",{}).get("OOMKilled") is not False
                or terminal.get("State",{}).get("Status") != "exited"):
            return {"verified":False}
        # The retained solver container is intentionally stopped after its exec
        # segments; container exit137 is not asserted to be solver exit0.
        return {"verified":True,"bridge_completed":True,"running":False,
                "machine_compute":False,"scientific_admission":False,
                "completed_segments":2,"planned_segments":2,
                "integration_status":"software_preparation_not_compute",
                "container_exit_code":terminal["State"]["ExitCode"]}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified":False}


_EXPLORATORY_MPC_PROFILES = {
    "exploratory_h2_feedback": {
        "unit": "fluid-control-exploratory-short-h2-real-cfd-20261006.service",
        "invocation": "e3b9eb7b58724a1c9ec4e64d63ac7bbe",
        "base": "artifacts/exploratory_paired_h2_real_cfd_20261006",
        "running_status": "EXPLORATORY_REAL_CFD_SHORT_H2_FEEDBACK_RUNNING_NOT_ADMISSION",
        "terminal_review": "instantaneous_h2",
    },
    "exploratory_causal_history_h2_feedback": {
        "unit": "fluid-control-exploratory-causal-h2-real-cfd-20261006.service",
        "invocation": "6f554f10e87e4b9f9d6b6ed8b555c548",
        "base": "artifacts/exploratory_causal_history_h2_real_cfd_20261006",
        "running_status": "EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H2_RUNNING_NOT_ADMISSION",
        "terminal_review": "causal_history_h2",
    },
    "exploratory_causal_history_h5_feedback": {
        "unit": "fluid-control-exploratory-causal-h5-real-cfd-20261006.service",
        "invocation": "6adc59fae65344d2b49b57cbe5b30f70",
        "base": "artifacts/exploratory_causal_history_h5_real_cfd_20261006",
        "running_status": "EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H5_RUNNING_NOT_ADMISSION",
        "terminal_review": None,
        "planned_cycles": 10,
    },
    "exploratory_accelerated_long_h5_feedback": {
        "unit": "fluid-control-accelerated-long-h5-20261006.service",
        "invocation": "a601eec2da7649b4af6f9354a4deb470",
        "base": "artifacts/exploratory_accelerated_long_h5_real_cfd_20261006",
        "running_status": "EXPLORATORY_ACCELERATED_LONG_H5_RUNNING_NOT_ADMISSION",
        "terminal_review": None,
        "planned_cycles": 124,
    },
}

_CURRENT_FIELD_CACHE = {"sha256": None, "png": None}
_CURRENT_FIELD_LOCK = threading.Lock()
_FINAL_PPO_FIELD_CACHE = {"sha256": None, "png": None}
_FINAL_PPO_FIELD_LOCK = threading.Lock()
_LONG_H5_RECOVERED_SHA = "1605604dc27f106acd05e6a721f26c4ba24527ac53996d6e65fbc70c601fa2b1"
_LONG_H5_REVIEW_SHA = "aa343c826b683eff840826d3bbcd1db02e2bdbfa1321a33522a965dde5d1f687"
_H5_PPO_APPROVAL_SHA = "8aa44f5f177d6c7d831a1efc55abf9d8db4b700d640cb640c5fcbb709cc44569"
_H5_PPO_RESULT_SHA = "138a7b192eef1a6454cefa47cda7803c9b362937641a645c00889ac5a5d7a0c4"
_H5_PPO_SUPERVISOR_SHA = "87d9f0be7dfb49565c0ea335691ad598f644f411b22297e6cce7fac4e8ab384c"
_DIVERSE_H5_PPO_APPROVAL_SHA = "760e1f9e81494bdd8c3742cd0ce77e168b0df2bf288bd04546412096721f41e2"
_DIVERSE_H5_PPO_RESULT_SHA = "cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640"
_DIVERSE_H5_PPO_SUPERVISOR_SHA = "0b3e2556a5f34136dee3f2749a5803dc9cffffe2852b25b2d7bd59b61182f1a1"
_DIVERSE_H5_PPO_DRIVER_SHA = "aae8c9a4311112439251b695001c7601ffd3d7cd3010937f86bd9a0bfebf3040"
_DIVERSE_32768_PPO_R2_APPROVAL_SHA = "1cd1d5182fd7e7a3eed11060e7a6ffe9fad840c776f021f239f059525a515132"
_DIVERSE_32768_PPO_DRIVER_SHA = "4d681771736b63b628712d3b62fcdde831601e80221aef6f1fd78a4b6840ff01"
_DIVERSE_32768_PPO_R1_FAILURE_SHA = "45a8fbb0f7e62962fed21b143ecab88b91cf5b33bd41a0fb5701a089cf1aab51"
_DIVERSE_PPO_CFD_APPROVAL_SHA = "67fda1a404f844d89b986442a4a9000561b02757417d5a8d366fdf9f9e6033db"
_DIVERSE_PPO_CFD_DRIVER_SHA = "89e0d8bea92440babd3d647eed31758db9cfc2a43e31ed6e9d9bdf5047b77b6e"
_DIVERSE_PPO_CFD_RESULT_SHA = "8c909aa4bd0b73e3cf570dd55cb2a1abd7346a9c424695a5e0056b4e5e833bdc"
_DIVERSE_PPO_CFD_REVIEW_SHA = "31a338bfc81e4ece0adeee943c074686e7065e57048d5783697d062856aa4e53"
_DIVERSE_32768_LONG_CFD_APPROVAL_SHA = "e103288a0558c10784a43a199a3c4d731ffc0e6509753646da7bb6930cb4dc12"
_DIVERSE_32768_LONG_CFD_DRIVER_SHA = "17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa"
_FINAL_PPO_CFD_APPROVAL_SHA = "7ace192519a08795fe9217473fae33941fc5edbb1075daeeb3701e672c521cb3"
_FINAL_PPO_CFD_DRIVER_SHA = "44b488a97a2882e1325da8871d3ac4905cdae2a6f2cbb17202ced91afc58b91a"
_FINAL_PPO_CFD_RESULT_SHA = "4007493f22de5855cbd0574e0ec006ca715941b8396f4e48af6527dc11e03d47"
_FINAL_PPO_CFD_REVIEW_SHA = "33c15b7ca14e9d65ef158cbec763dcbc0f16d162ac30f129d3b13ddb765eb301"
_FINAL_PPO_FIELD_RESULT_SHA = "7b8930ef7b928799e8982bd9c4d7507c1e11d34cc6f1f466820358ff13124d69"
_FINAL_PPO_FIELD_NPZ_SHA = "967fdb76efc5124630e2cc5b738b06521a036fe348fe9d550908c31e1b0cc448"
_FINAL_PPO_FIELD_UNIT_EVIDENCE_SHA = "e1f5fa599b35b3e4603800608e892ee605e619d6ac9029691705ce314d2b709b"
_FINAL_PPO_FIELD_MANIFEST_SHA = "f6328c753d2db54556d502c0101f4a13b41a3113bdf659af49b00d40f2c255b8"


def _long_h5_recovered_metrics(root: Path) -> dict | None:
    """Return compact, independently reviewed recovery evidence; preserve exit1."""
    base = root / "artifacts/exploratory_accelerated_long_h5_real_cfd_20261006"
    metrics_path = base / "recovered_metrics.json"
    review_path = root / "docs/EXPLORATORY_ACCELERATED_LONG_H5_TERMINAL_REVIEW_20261006.md"
    try:
        if (hashlib.sha256(metrics_path.read_bytes()).hexdigest() != _LONG_H5_RECOVERED_SHA
                or hashlib.sha256(review_path.read_bytes()).hexdigest() != _LONG_H5_REVIEW_SHA):
            return None
        payload = json.loads(metrics_path.read_text())
        unit = payload["original_unit"]
        if (payload.get("status") != "OFFLINE_METRICS_RECOVERED_FROM_FAILED_POSTPROCESSING_NOT_ADMISSION"
                or payload.get("cycles") != 124 or payload.get("scientific_admission") is not False
                or payload.get("new_cfd_or_model_execution") is not False
                or payload.get("original_result_written") is not False
                or unit.get("InvocationID") != "a601eec2da7649b4af6f9354a4deb470"
                or unit.get("Result") != "exit-code" or unit.get("ExecMainStatus") != "1"
                or payload.get("original_restart_rehashed_unchanged") is not True):
            return None
        expected = {
            "full": ([148.0, 160.4], 2480),
            "first_6p2": ([148.0, 154.2], 1240),
            "trailing_6p2": ([154.2, 160.4], 1240),
        }
        windows = []
        for name, (interval, samples) in expected.items():
            row = payload["windows"][name]; mpc, zero = row["branches"]["mpc"], row["branches"]["zero"]
            values = [row["paired_drag_reduction"], row["paired_rear_cl_fluctuation_rms_ratio"],
                      row["absolute_mean_rear_cl_over_paired_zero_rms"],
                      mpc["total_cd_mean"], zero["total_cd_mean"], mpc["rear_cl_mean"],
                      mpc["rear_cl_fluctuation_rms"], zero["rear_cl_fluctuation_rms"]]
            if (row.get("interval_open_left_closed_right") != interval
                    or mpc.get("samples") != samples or zero.get("samples") != samples
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in values)):
                return None
            windows.append({"name": name, "interval": interval,
                            "mpc_total_cd_mean": mpc["total_cd_mean"],
                            "zero_total_cd_mean": zero["total_cd_mean"],
                            "drag_reduction_percent": 100 * row["paired_drag_reduction"],
                            "mpc_rear_cl_mean": mpc["rear_cl_mean"],
                            "mpc_rear_cl_rms": mpc["rear_cl_fluctuation_rms"],
                            "zero_rear_cl_rms": zero["rear_cl_fluctuation_rms"],
                            "rear_cl_rms_change_percent": 100 * (
                                row["paired_rear_cl_fluctuation_rms_ratio"] - 1),
                            "mean_bias_over_zero_rms_percent": 100 * row[
                                "absolute_mean_rear_cl_over_paired_zero_rms"]})
        return {"verified": True, "metrics_sha256": _LONG_H5_RECOVERED_SHA,
                "review_sha256": _LONG_H5_REVIEW_SHA, "original_unit_exit_status": 1,
                "original_result_written": False, "scientific_admission": False,
                "windows": windows}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _complete_jsonl(path: Path):
    """Yield only newline-terminated JSON objects; reject malformed complete rows."""
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("JSONL row is not an object")
            yield row


def _exploratory_h5_ppo_training(root: Path) -> dict:
    """Bind the completed exploratory PPO training; never imply CFD benefit."""
    base = root / "artifacts/exploratory_h5_ppo_training_20261006"
    payload = base / "payload"
    approval = root / "docs/EXPLORATORY_H5_PPO_APPROVAL_20261006.json"
    result_path = payload / "result.json"
    supervisor_path = base / "supervisor_result.json"
    try:
        if (hashlib.sha256(approval.read_bytes()).hexdigest() != _H5_PPO_APPROVAL_SHA
                or hashlib.sha256(result_path.read_bytes()).hexdigest() != _H5_PPO_RESULT_SHA
                or hashlib.sha256(supervisor_path.read_bytes()).hexdigest()
                != _H5_PPO_SUPERVISOR_SHA):
            raise ValueError("PPO receipt identity mismatch")
        result = json.loads(result_path.read_text())
        supervisor = json.loads(supervisor_path.read_text())
        artifacts = result["artifacts"]
        expected_artifacts = {
            "ppo_final.zip": artifacts["ppo_final.zip"],
            "vecnormalize.pkl": artifacts["vecnormalize.pkl"],
            "transitions.jsonl": artifacts["transitions.jsonl"],
            "source_spec.json": artifacts["source_spec.json"],
            "progress.json": artifacts["progress.json"],
        }
        for name, expected_sha in expected_artifacts.items():
            if hashlib.sha256((payload / name).read_bytes()).hexdigest() != expected_sha:
                raise ValueError("PPO artifact identity mismatch")
        if (result.get("status") != "EXPLORATORY_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION"
                or result.get("timesteps") != 4096
                or result.get("scientific_admission") is not False
                or result.get("cfd_executed") is not False
                or result.get("fno_tensors_unchanged") is not True
                or result.get("policy_tensor_sha256_before")
                == result.get("policy_tensor_sha256_after")
                or supervisor.get("returncode") != 0
                or supervisor.get("result_sha256") != _H5_PPO_RESULT_SHA
                or supervisor.get("scientific_admission") is not False):
            raise ValueError("unexpected PPO terminal contract")
        progress = {}
        progress_rows = 0
        for row in _complete_jsonl(payload / "progress.json"):
            progress.update(row)
            progress_rows += 1
        env_counts = {index: 0 for index in range(4)}
        transition_count = 0
        truncated = 0
        for row in _complete_jsonl(payload / "transitions.jsonl"):
            index = row.get("env_index")
            if index not in env_counts or row.get("scientific_admission") is not False:
                raise ValueError("unexpected transition identity")
            env_counts[index] += 1
            transition_count += 1
            truncated += int(row.get("TimeLimit.truncated") is True)
        diagnostics = result["diagnostics"]
        if (progress_rows != 9 or transition_count != 4096
                or env_counts != {0: 1024, 1: 1024, 2: 1024, 3: 1024}
                or truncated != 816 or diagnostics.get("episodes_completed") != 816
                or progress.get("time/total_timesteps") != 4096
                or len(result.get("optimizer_steps", [])) != 64
                or result.get("ppo_n_updates") != 32):
            raise ValueError("incomplete PPO records")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result",
                  "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show",
            "fluid-control-exploratory-h5-ppo-20261006.service",
            *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if (state.get("InvocationID") != "21cb82da66214924b38f120eb30723e5"
                or state.get("MainPID") != "0" or state.get("ActiveState") != "active"
                or state.get("SubState") != "exited" or state.get("Result") != "success"
                or state.get("ExecMainCode") != "1" or state.get("ExecMainStatus") != "0"):
            raise ValueError("PPO unit is not the reviewed terminal execution")
        return {"verified": True, "running": False, "training_complete": True,
                "timesteps": transition_count, "target_timesteps": 4096,
                "environments": 4, "environment_counts": env_counts,
                "timeouts": truncated, "ppo_updates": result["ppo_n_updates"],
                "optimizer_steps": len(result["optimizer_steps"]),
                "value_loss": progress["train/value_loss"],
                "approx_kl": progress["train/approx_kl"],
                "episode_reward_mean": diagnostics["episode_return"]["mean"],
                "policy_changed": True, "fno_tensors_unchanged": True,
                "cfd_executed": False, "scientific_admission": False,
                "minimum_available_gib": supervisor["minimum_available_bytes"] / 2**30,
                "result_sha256": _H5_PPO_RESULT_SHA,
                "approval_sha256": _H5_PPO_APPROVAL_SHA}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return {"verified": False, "running": False, "training_complete": False}


def _exploratory_diverse_h5_ppo_training(root: Path) -> dict:
    """Bind the terminal 24-reset PPO training without implying CFD benefit."""
    base = root / "artifacts/exploratory_diverse_h5_ppo_training_20261006"
    payload = base / "payload"
    approval = root / "docs/EXPLORATORY_DIVERSE_H5_PPO_APPROVAL_20261006.json"
    result_path = payload / "result.json"
    supervisor_path = base / "supervisor_result.json"
    driver = (root / "artifacts/exploratory_diverse_h5_ppo_source_20261006_immutable"
              / "scripts/train_exploratory_diverse_h5_ppo.py")
    try:
        if (hashlib.sha256(approval.read_bytes()).hexdigest()
                != _DIVERSE_H5_PPO_APPROVAL_SHA
                or hashlib.sha256(result_path.read_bytes()).hexdigest()
                != _DIVERSE_H5_PPO_RESULT_SHA
                or hashlib.sha256(supervisor_path.read_bytes()).hexdigest()
                != _DIVERSE_H5_PPO_SUPERVISOR_SHA
                or hashlib.sha256(driver.read_bytes()).hexdigest()
                != _DIVERSE_H5_PPO_DRIVER_SHA):
            raise ValueError("diverse PPO receipt identity mismatch")
        approval_payload = json.loads(approval.read_text())
        result = json.loads(result_path.read_text())
        supervisor = json.loads(supervisor_path.read_text())
        artifacts = result["artifacts"]
        for name, expected_sha in artifacts.items():
            if hashlib.sha256((payload / name).read_bytes()).hexdigest() != expected_sha:
                raise ValueError("diverse PPO artifact identity mismatch")
        counts = result["reset_counts_by_phase"]
        expected_counts = [35, 34, 34, 34, 34, 34]
        if (approval_payload.get("status") != "EXPLORATORY_DIVERSE_H5_PPO_EXECUTION_APPROVED"
                or approval_payload.get("execution_authorized") is not True
                or result.get("status")
                != "EXPLORATORY_DIVERSE_H5_PPO_TRAINING_COMPLETE_NOT_ADMISSION"
                or result.get("timesteps") != 4096
                or result.get("protocol", {}).get("reset_count") != 24
                or result.get("protocol", {}).get("cfd_execution") is not False
                or result.get("scientific_admission") is not False
                or result.get("fno_tensors_unchanged") is not True
                or result.get("policy_tensor_sha256_before")
                == result.get("policy_tensor_sha256_after")
                or result.get("ppo_n_updates") != 32
                or len(result.get("optimizer_steps", [])) != 64
                or set(counts) != {"00", "02", "04", "06"}
                or any(counts[phase] != expected_counts for phase in counts)
                or supervisor.get("returncode") != 0
                or supervisor.get("result_sha256") != _DIVERSE_H5_PPO_RESULT_SHA
                or supervisor.get("scientific_admission") is not False):
            raise ValueError("unexpected diverse PPO terminal contract")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result",
                  "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show",
            "fluid-control-exploratory-diverse-h5-ppo-20261006.service",
            *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if (state.get("InvocationID") != "3a34c4d621244e4bbacdf1816b5b1374"
                or state.get("MainPID") != "0" or state.get("ActiveState") != "active"
                or state.get("SubState") != "exited" or state.get("Result") != "success"
                or state.get("ExecMainCode") != "1" or state.get("ExecMainStatus") != "0"):
            raise ValueError("diverse PPO unit is not the reviewed terminal execution")
        diagnostics = result["diagnostics"]
        return {"verified": True, "running": False, "training_complete": True,
                "timesteps": 4096, "reset_count": 24,
                "phase_reset_counts": [sum(counts[p]) for p in ("00", "02", "04", "06")],
                "ppo_updates": 32, "optimizer_steps": 64,
                "episode_reward_mean": diagnostics["episode_return"]["mean"],
                "applied_omega_rms": diagnostics["applied_omega"]["rms"],
                "minimum_available_gib": supervisor["minimum_available_bytes"] / 2**30,
                "policy_changed": True, "fno_tensors_unchanged": True,
                "cfd_executed": False, "scientific_admission": False,
                "approval_sha256": _DIVERSE_H5_PPO_APPROVAL_SHA,
                "result_sha256": _DIVERSE_H5_PPO_RESULT_SHA}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return {"verified": False, "running": False, "training_complete": False}


def _exploratory_diverse_h5_32768_ppo_training(root: Path) -> dict:
    """Expose R2 only after actual transitions; preserve R1 as pre-training failure."""
    base = root / "artifacts/exploratory_diverse_h5_32768_ppo_training_20261006_r2"
    approval_path = root / "docs/EXPLORATORY_DIVERSE_H5_32768_PPO_R2_APPROVAL_20261006.json"
    driver_path = (root / "artifacts/exploratory_diverse_h5_32768_source_20261006_immutable"
                   / "train_exploratory_diverse_h5_32768_ppo.py")
    r1 = (root / "artifacts/exploratory_diverse_h5_32768_ppo_training_20261006"
          / "supervisor_result.json")
    try:
        if (hashlib.sha256(approval_path.read_bytes()).hexdigest()
                != _DIVERSE_32768_PPO_R2_APPROVAL_SHA
                or hashlib.sha256(driver_path.read_bytes()).hexdigest()
                != _DIVERSE_32768_PPO_DRIVER_SHA
                or hashlib.sha256(r1.read_bytes()).hexdigest()
                != _DIVERSE_32768_PPO_R1_FAILURE_SHA):
            raise ValueError("32768 PPO identity mismatch")
        approval = json.loads(approval_path.read_text())
        r1_result = json.loads(r1.read_text())
        protocol = approval.get("protocol", {})
        if (approval.get("status") != "EXPLORATORY_DIVERSE_H5_32768_PPO_EXECUTION_APPROVED"
                or approval.get("execution_authorized") is not True
                or approval.get("reviewed_by_lead") is not True
                or protocol.get("timesteps") != 32768 or protocol.get("reset_count") != 24
                or protocol.get("device") != "cuda:0"
                or protocol.get("cfd_execution") is not False
                or protocol.get("scientific_admission") is not False
                or r1_result.get("returncode") != 1 or r1_result.get("result_sha256") is not None
                or r1_result.get("scientific_admission") is not False):
            raise ValueError("unexpected 32768 PPO approval/failure evidence")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result",
                  "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show",
            "fluid-control-exploratory-diverse-h5-32768-ppo-r2-20261006.service",
            *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if state.get("InvocationID") != "a19900b2bfa64d8d8372b67bc0564139":
            raise ValueError("unexpected 32768 PPO R2 invocation")
        running = (state.get("MainPID", "0").isdigit() and int(state["MainPID"]) > 0
                   and state.get("ActiveState") == "active"
                   and state.get("SubState") in ("running", "start"))
        progress = {}
        rows = 0
        for item in _complete_jsonl(base / "payload/progress.json"):
            progress.update(item); rows += 1
        timesteps = progress.get("time/total_timesteps")
        if (rows < 1 or type(timesteps) is not int or not 0 < timesteps <= 32768
                or timesteps % 512 != 0):
            raise ValueError("no actual 32768 PPO R2 transitions observed")
        memory = list(_complete_jsonl(base / "memory.jsonl"))
        available = [row["MemAvailable"] for row in memory
                     if type(row.get("MemAvailable")) in (int, float)
                     and math.isfinite(row["MemAvailable"])]
        if not available:
            raise ValueError("missing 32768 PPO R2 resources")
        return {"verified": True, "running": running, "timesteps": timesteps,
                "target_timesteps": 32768,
                "ppo_updates": int(progress.get("train/n_updates", 0)),
                "episode_reward_mean": progress.get("rollout/ep_rew_mean"),
                "current_available_gib": available[-1] / 2**30,
                "minimum_available_gib": min(available) / 2**30,
                "reset_count": 24, "fno_tensors_frozen": True,
                "gpu_policy_training": True, "cfd_executed": False,
                "scientific_admission": False, "r1_pretraining_failure_preserved": True,
                "approval_sha256": _DIVERSE_32768_PPO_R2_APPROVAL_SHA}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return {"verified": False, "running": False, "scientific_admission": False}


def _exploratory_diverse_ppo_real_cfd(root: Path) -> dict:
    """Read the live new-policy/zero CFD pair without borrowing old field evidence."""
    base = root / "artifacts/exploratory_diverse_ppo_real_cfd_20261006"
    approval_path = root / "docs/EXPLORATORY_DIVERSE_PPO_CFD_APPROVAL_20261006.json"
    driver_path = (root / "artifacts/exploratory_diverse_ppo_cfd_source_20261006_immutable"
                   / "run_exploratory_diverse_ppo_real_cfd.py")
    try:
        if (hashlib.sha256(approval_path.read_bytes()).hexdigest()
                != _DIVERSE_PPO_CFD_APPROVAL_SHA
                or hashlib.sha256(driver_path.read_bytes()).hexdigest()
                != _DIVERSE_PPO_CFD_DRIVER_SHA):
            raise ValueError("diverse PPO CFD identity mismatch")
        approval = json.loads(approval_path.read_text())
        if (approval.get("status") != "EXPLORATORY_DIVERSE_PPO_REAL_CFD_EXECUTION_APPROVED"
                or approval.get("execution_authorized") is not True
                or approval.get("steps") != 124 or approval.get("inference_device") != "cpu"
                or approval.get("scientific_admission") is not False
                or approval.get("driver_sha256") != _DIVERSE_PPO_CFD_DRIVER_SHA):
            raise ValueError("unexpected diverse PPO CFD approval")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result",
                  "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show",
            "fluid-control-exploratory-diverse-ppo-cfd-20261006.service",
            *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if state.get("InvocationID") != "464de68ee1114eea8e8ae214d18dc045":
            raise ValueError("unexpected diverse PPO CFD invocation")
        running = (state.get("MainPID", "0").isdigit() and int(state["MainPID"]) > 0
                   and state.get("ActiveState") == "active"
                   and state.get("SubState") in ("running", "start"))
        document = json.loads((base / "progress.json").read_text())
        rows = document.get("rows", []); completed = document.get("completed_cycles")
        if (type(completed) is not int or completed != len(rows) or not 0 < completed <= 124):
            raise ValueError("invalid diverse PPO CFD progress count")
        series = []
        for index, row in enumerate(rows, 1):
            ppo, zero = row["output_observation"], row["zero_observation"]
            values = [row["start_time"], row["end_time"], row["requested_omega"],
                      row["applied_omega"], ppo[64], ppo[66], ppo[67],
                      zero[64], zero[66], zero[67]]
            if (row.get("step") != index or len(ppo) != 69 or len(zero) != 69
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in values)
                    or abs(row["end_time"] - row["start_time"] - .1) > 1e-8
                    or row.get("solver_health", {}).get("ppo", {}).get("steps") != 20
                    or row.get("solver_health", {}).get("zero", {}).get("steps") != 20
                    or row["solver_health"]["ppo"].get("solver_ended_cleanly") is not True
                    or row["solver_health"]["zero"].get("solver_ended_cleanly") is not True):
                raise ValueError("invalid diverse PPO CFD row")
            series.append({"force_time": row["end_time"],
                           "requested_omega": row["requested_omega"],
                           "omega": row["applied_omega"],
                           "ppo_total_cd": ppo[64] + ppo[66],
                           "zero_total_cd": zero[64] + zero[66],
                           "ppo_rear_cl": ppo[67], "zero_rear_cl": zero[67]})
        available = []
        for resource in _complete_jsonl(base / "resources.jsonl"):
            value = resource.get("MemAvailable")
            if type(value) in (int, float) and math.isfinite(value):
                available.append(value)
        if not available:
            raise ValueError("missing diverse PPO CFD resources")
        terminal = None
        result_path = base / "result.json"
        review_path = root / "docs/EXPLORATORY_DIVERSE_PPO_CFD_TERMINAL_REVIEW_20261006.md"
        if (completed == 124 and not running and result_path.is_file()
                and hashlib.sha256(result_path.read_bytes()).hexdigest()
                == _DIVERSE_PPO_CFD_RESULT_SHA and review_path.is_file()
                and hashlib.sha256(review_path.read_bytes()).hexdigest()
                == _DIVERSE_PPO_CFD_REVIEW_SHA):
            result = json.loads(result_path.read_text())
            if (result.get("status") != "EXPLORATORY_DIVERSE_PPO_REAL_CFD_COMPLETE_NOT_ADMISSION"
                    or result.get("cycles") != 124
                    or result.get("scientific_admission") is not False
                    or result.get("approval_sha256") != _DIVERSE_PPO_CFD_APPROVAL_SHA
                    or result.get("source_restart_unchanged") is not True
                    or result.get("owned_containers_cleaned") is not True
                    or result.get("fno_inference") is not False
                    or result.get("mpc_action_selection") is not False):
                raise ValueError("unexpected diverse PPO CFD terminal result")
            expected_windows = {"full": ([148.0, 160.4], 2480),
                                "first_6p2": ([148.0, 154.2], 1240),
                                "trailing_6p2": ([154.2, 160.4], 1240)}
            windows = []
            for name, (interval, samples) in expected_windows.items():
                item = result["windows"][name]; ppo = item["branches"]["ppo"]
                zero = item["branches"]["zero"]
                values = [item["paired_drag_reduction"],
                          item["paired_rear_cl_fluctuation_rms_ratio"],
                          item["absolute_mean_rear_cl_over_paired_zero_rms"],
                          ppo["total_cd_mean"], zero["total_cd_mean"]]
                if (item.get("interval_open_left_closed_right") != interval
                        or ppo.get("samples") != samples or zero.get("samples") != samples
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               for value in values)):
                    raise ValueError("invalid diverse PPO CFD window")
                windows.append({"name": name,
                    "drag_reduction_percent": 100 * values[0],
                    "rear_cl_rms_change_percent": 100 * (values[1] - 1),
                    "mean_bias_percent": 100 * values[2],
                    "ppo_total_cd_mean": values[3], "zero_total_cd_mean": values[4]})
            action = result["action_summary"]
            terminal = {"verified": True, "result_sha256": _DIVERSE_PPO_CFD_RESULT_SHA,
                        "review_sha256": _DIVERSE_PPO_CFD_REVIEW_SHA,
                        "physical_success": False, "windows": windows,
                        "max_abs_applied_omega": action["max_abs_applied_omega"],
                        "saturated_endpoints": action["saturated_endpoints"]}
        return {"verified": True, "running": running, "completed_cycles": completed,
                "planned_cycles": 124, "actual_timeseries": series,
                "latest": series[-1], "minimum_available_gib": min(available) / 2**30,
                "policy_training_complete": True, "fno_training": False,
                "inference_device": "cpu", "scientific_admission": False,
                "control_success_verified": False, "terminal_result": terminal,
                "approval_sha256": _DIVERSE_PPO_CFD_APPROVAL_SHA}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return {"verified": False, "running": False,
                "scientific_admission": False, "control_success_verified": False}


def _exploratory_diverse_32768_long_cfd(root: Path) -> dict:
    """Read the one actual 800-cycle CFD pair only after a completed cycle exists."""
    base = root / "artifacts/exploratory_diverse_32768_ppo_long_cfd_20261006"
    approval_path = root / "docs/EXPLORATORY_DIVERSE_32768_PPO_LONG_CFD_APPROVAL_20261006.json"
    driver_path = (root / "artifacts/exploratory_diverse_32768_ppo_long_cfd_source_20261006_immutable"
                   / "run_exploratory_diverse_32768_ppo_long_cfd.py")
    try:
        if (hashlib.sha256(approval_path.read_bytes()).hexdigest()
                != _DIVERSE_32768_LONG_CFD_APPROVAL_SHA
                or hashlib.sha256(driver_path.read_bytes()).hexdigest()
                != _DIVERSE_32768_LONG_CFD_DRIVER_SHA):
            raise ValueError("long CFD identity mismatch")
        approval = json.loads(approval_path.read_text())
        if (approval.get("status")
                != "EXPLORATORY_DIVERSE_32768_PPO_LONG_CFD_EXECUTION_APPROVED"
                or approval.get("execution_authorized") is not True
                or approval.get("steps") != 800 or approval.get("inference_device") != "cpu"
                or approval.get("scientific_admission") is not False
                or approval.get("driver_sha256") != _DIVERSE_32768_LONG_CFD_DRIVER_SHA):
            raise ValueError("unexpected long CFD approval")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result",
                  "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show",
            "fluid-control-exploratory-diverse-32768-ppo-long-cfd-20261006.service",
            *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if state.get("InvocationID") != "285bea88ff234cd5acfb9cb03c2b3cf3":
            raise ValueError("unexpected long CFD invocation")
        running = (state.get("MainPID", "0").isdigit() and int(state["MainPID"]) > 0
                   and state.get("ActiveState") == "active"
                   and state.get("SubState") in ("running", "start"))
        document = json.loads((base / "progress.json").read_text())
        rows = document.get("rows", []); completed = document.get("completed_cycles")
        if (type(completed) is not int or completed != len(rows) or not 0 < completed <= 800):
            raise ValueError("invalid long CFD progress count")
        series = []
        for index, row in enumerate(rows, 1):
            ppo, zero = row["output_observation"], row["zero_observation"]
            values = [row["start_time"], row["end_time"], row["requested_omega"],
                      row["applied_omega"], ppo[64], ppo[66], ppo[67],
                      zero[64], zero[66], zero[67]]
            if (row.get("step") != index or len(ppo) != 69 or len(zero) != 69
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in values)
                    or abs(row["start_time"] - (148.0 + .1 * (index - 1))) > 1e-8
                    or abs(row["end_time"] - (148.0 + .1 * index)) > 1e-8
                    or row.get("solver_health", {}).get("ppo", {}).get("steps") != 20
                    or row.get("solver_health", {}).get("zero", {}).get("steps") != 20
                    or row["solver_health"]["ppo"].get("solver_ended_cleanly") is not True
                    or row["solver_health"]["zero"].get("solver_ended_cleanly") is not True):
                raise ValueError("invalid long CFD row")
            series.append({"force_time": row["end_time"],
                           "requested_omega": row["requested_omega"],
                           "omega": row["applied_omega"],
                           "ppo_total_cd": ppo[64] + ppo[66],
                           "zero_total_cd": zero[64] + zero[66],
                           "ppo_rear_cl": ppo[67], "zero_rear_cl": zero[67]})
        available = [row["MemAvailable"] for row in _complete_jsonl(base / "resources.jsonl")
                     if type(row.get("MemAvailable")) in (int, float)
                     and math.isfinite(row["MemAvailable"])]
        if not available:
            raise ValueError("missing long CFD resources")
        return {"verified": True, "running": running, "completed_cycles": completed,
                "planned_cycles": 800, "actual_timeseries": series,
                "latest": series[-1], "current_available_gib": available[-1] / 2**30,
                "minimum_available_gib": min(available) / 2**30,
                "policy_training_complete": True, "gpu_training": False,
                "inference_device": "cpu", "primary_window": [168.0, 228.0],
                "discarded_warmup_cycles": 200, "early_comparison_cycles": 124,
                "scientific_admission": False, "control_success_verified": False,
                "approval_sha256": _DIVERSE_32768_LONG_CFD_APPROVAL_SHA}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return {"verified": False, "running": False,
                "scientific_admission": False, "control_success_verified": False}


def _exploratory_final_ppo_real_cfd(root: Path) -> dict:
    """Read the fixed final-policy/zero real-CFD pair without borrowing MPC fields."""
    base = root / "artifacts/exploratory_final_ppo_real_cfd_20261006"
    approval_path = root / "docs/EXPLORATORY_FINAL_PPO_CFD_APPROVAL_20261006.json"
    driver_path = (root / "artifacts/exploratory_final_ppo_cfd_source_20261006_immutable"
                   / "run_exploratory_ppo_real_cfd.py")
    try:
        if (hashlib.sha256(approval_path.read_bytes()).hexdigest()
                != _FINAL_PPO_CFD_APPROVAL_SHA
                or hashlib.sha256(driver_path.read_bytes()).hexdigest()
                != _FINAL_PPO_CFD_DRIVER_SHA):
            raise ValueError("final PPO CFD identity mismatch")
        approval = json.loads(approval_path.read_text())
        if (approval.get("status") != "EXPLORATORY_FINAL_PPO_REAL_CFD_EXECUTION_APPROVED"
                or approval.get("execution_authorized") is not True
                or approval.get("steps") != 124
                or approval.get("inference_device") != "cpu"
                or approval.get("scientific_admission") is not False
                or approval.get("driver_sha256") != _FINAL_PPO_CFD_DRIVER_SHA):
            raise ValueError("unexpected final PPO CFD approval")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result",
                  "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show",
            "fluid-control-exploratory-final-ppo-cfd-20261006.service",
            *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        if state.get("InvocationID") != "fd6d92f7ea9946b49c91c07e21f1d74b":
            raise ValueError("unexpected final PPO CFD invocation")
        running = (state.get("ActiveState") == "active"
                   and state.get("SubState") == "running"
                   and state.get("MainPID", "0").isdigit()
                   and state.get("MainPID") != "0")
        progress = json.loads((base / "progress.json").read_text())
        rows = progress.get("rows")
        completed = progress.get("completed_cycles")
        if (type(completed) is not int or not 0 <= completed <= 124
                or not isinstance(rows, list) or len(rows) != completed):
            raise ValueError("invalid final PPO CFD progress")
        actual_timeseries = []
        latest = None
        for expected_step, row in enumerate(rows, 1):
            if row.get("step") != expected_step:
                raise ValueError("nonsequential final PPO CFD progress")
            start, end = row.get("start_time"), row.get("end_time")
            requested, omega, delta = (row.get("requested_omega"),
                                       row.get("applied_omega"),
                                       row.get("applied_delta_omega"))
            ppo_obs, zero_obs = row.get("output_observation"), row.get("zero_observation")
            values = [start, end, requested, omega, delta]
            if (any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in values)
                    or abs(end - start - .1) > 1e-8
                    or abs(omega) > .75 + 1e-12 or abs(delta) > .1 + 1e-12
                    or not isinstance(ppo_obs, list) or len(ppo_obs) != 69
                    or not isinstance(zero_obs, list) or len(zero_obs) != 69
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in [*ppo_obs, *zero_obs])):
                raise ValueError("invalid final PPO CFD row")
            health = row.get("solver_health", {})
            if any(health.get(branch, {}).get("solver_ended_cleanly") is not True
                   or health.get(branch, {}).get("steps") != 20
                   for branch in ("ppo", "zero")):
                raise ValueError("invalid solver evidence")
            item = {"step": expected_step, "force_time": float(end),
                    "omega": float(omega),
                    "ppo_total_cd": float(ppo_obs[64] + ppo_obs[66]),
                    "zero_total_cd": float(zero_obs[64] + zero_obs[66]),
                    "ppo_rear_cl": float(ppo_obs[67]),
                    "zero_rear_cl": float(zero_obs[67])}
            actual_timeseries.append(item)
            latest = item
        resources = list(_complete_jsonl(base / "resources.jsonl"))
        available = [row.get("MemAvailable") for row in resources]
        if not available or any(type(value) is not int or value <= 0 for value in available):
            raise ValueError("missing final PPO CFD resources")
        terminal = None
        result_path = base / "result.json"
        review_path = root / "docs/EXPLORATORY_FINAL_PPO_CFD_TERMINAL_REVIEW_20261006.md"
        if (completed == 124 and not running and result_path.is_file()
                and review_path.is_file()
                and hashlib.sha256(result_path.read_bytes()).hexdigest()
                == _FINAL_PPO_CFD_RESULT_SHA
                and hashlib.sha256(review_path.read_bytes()).hexdigest()
                == _FINAL_PPO_CFD_REVIEW_SHA):
            result = json.loads(result_path.read_text())
            if (result.get("status") != "EXPLORATORY_FINAL_PPO_REAL_CFD_COMPLETE_NOT_ADMISSION"
                    or result.get("cycles") != 124
                    or result.get("scientific_admission") is not False
                    or result.get("approval_sha256") != _FINAL_PPO_CFD_APPROVAL_SHA
                    or result.get("source_restart_unchanged") is not True
                    or result.get("owned_containers_cleaned") is not True
                    or result.get("fno_inference") is not False
                    or result.get("mpc_action_selection") is not False
                    or len(result.get("rows", [])) != 124
                    or any(row.get("requested_omega") != .75
                           for row in result["rows"])
                    or result.get("action_summary", {}).get("saturated_endpoints") != 117
                    or result.get("action_summary", {}).get("rate_limited_endpoints") != 7):
                raise ValueError("unexpected final PPO CFD terminal result")
            expected_windows = {
                "full": ([148.0, 160.4], 2480),
                "first_6p2": ([148.0, 154.2], 1240),
                "trailing_6p2": ([154.2, 160.4], 1240),
            }
            fixed_zero_rms = 1.1826535012844825
            terminal_windows = []
            for name, (interval, samples) in expected_windows.items():
                window = result["windows"][name]
                ppo = window["branches"]["ppo"]
                zero = window["branches"]["zero"]
                drag = window["paired_drag_reduction"]
                rms_ratio = window["paired_rear_cl_fluctuation_rms_ratio"]
                bias = window["absolute_mean_rear_cl_over_paired_zero_rms"]
                values = [drag, rms_ratio, bias, ppo["total_cd_mean"],
                          zero["total_cd_mean"], ppo["rear_cl_mean"],
                          ppo["rear_cl_fluctuation_rms"],
                          zero["rear_cl_fluctuation_rms"]]
                if (window.get("interval_open_left_closed_right") != interval
                        or ppo.get("samples") != samples or zero.get("samples") != samples
                        or any(type(value) not in (int, float) or not math.isfinite(value)
                               for value in values)):
                    raise ValueError("invalid final PPO CFD terminal window")
                terminal_windows.append({"name": name,
                    "drag_reduction_percent": 100 * drag,
                    "rear_cl_rms_change_percent": 100 * (rms_ratio - 1),
                    "mean_bias_percent": 100 * bias,
                    "fixed_reference_mean_bias_percent":
                        100 * abs(ppo["rear_cl_mean"]) / fixed_zero_rms,
                    "passes_original_10_percent_mean_bias": bias <= .1,
                    "passes_sensitivity_20_percent_mean_bias": bias <= .2,
                    "ppo_total_cd_mean": ppo["total_cd_mean"],
                    "zero_total_cd_mean": zero["total_cd_mean"],
                    "ppo_rear_cl_mean": ppo["rear_cl_mean"],
                    "ppo_rear_cl_rms": ppo["rear_cl_fluctuation_rms"],
                    "zero_rear_cl_rms": zero["rear_cl_fluctuation_rms"]})
            terminal = {"verified": True, "result_sha256": _FINAL_PPO_CFD_RESULT_SHA,
                        "review_sha256": _FINAL_PPO_CFD_REVIEW_SHA,
                        "requested_positive_limit_count": 124,
                        "saturated_endpoints": 117, "rate_limited_endpoints": 7,
                        "physical_success": False, "windows": terminal_windows}
        field = _final_ppo_field_evidence(root) if terminal is not None else None
        return {"verified": True, "running": running,
                "completed_cycles": completed, "planned_cycles": 124,
                "latest": latest, "actual_timeseries": actual_timeseries,
                "minimum_available_gib": min(available) / 2**30,
                "policy_training_complete": True, "inference_device": "cpu",
                "gpu_training": False, "online_fno": False, "mpc": False,
                "scientific_admission": False, "control_success_verified": False,
                "approval_sha256": _FINAL_PPO_CFD_APPROVAL_SHA,
                "actual_cfd_field": field,
                "terminal_review": terminal}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError):
        return {"verified": False, "running": False,
                "scientific_admission": False, "control_success_verified": False}


def _current_trial_field_evidence(root: Path, profile: dict, rows: list[dict],
                                  requested_sha256: str | None = None) -> dict | None:
    """Validate the fixed current-trial NPZ; never accept a caller path."""
    if (profile.get("base") != "artifacts/exploratory_accelerated_long_h5_real_cfd_20261006"
            or not rows or len(rows) > 124):
        return None
    if requested_sha256 is None:
        row = rows[-1]
    else:
        if not re.fullmatch(r"[0-9a-f]{64}", requested_sha256):
            return None
        matches = [row for row in rows
                   if row.get("current_sample_sha256", {}).get("mpc") == requested_sha256]
        if len(matches) != 1:
            return None
        row = matches[0]
    step, start, end = row.get("step"), row.get("start_time"), row.get("end_time")
    if (type(step) is not int or type(start) not in (int, float)
            or type(end) not in (int, float) or abs(end - start - .1) > 1e-8):
        return None
    base = root / profile["base"]
    path = base / f"current_mpc_{float(start):.1f}.npz"
    try:
        expected = row["current_sample_sha256"]["mpc"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            return None
        import numpy as np
        with np.load(path, allow_pickle=False) as packet:
            if set(packet.files) != {"state", "mask", "time", "x", "y"}:
                return None
            state, mask = packet["state"], packet["mask"]
            sample_time, x, y = packet["time"], packet["x"], packet["y"]
            time_tolerance = max(
                2 * float(abs(np.spacing(np.float32(abs(sample_time[0]))))), 1e-7)
            if (state.shape != (3, 128, 256) or state.dtype != np.float32
                    or mask.shape != (1, 128, 256) or mask.dtype != np.uint8
                    or sample_time.shape != (1,) or x.shape != (256,) or y.shape != (128,)
                    or x.dtype != np.float32 or y.dtype != np.float32
                    or not np.isfinite(state).all() or not np.isfinite(sample_time).all()
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.all((mask == 0) | (mask == 1))
                    or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)
                    or abs(float(sample_time[0]) - float(start)) > time_tolerance):
                return None
        return {"verified": True, "sha256": digest, "field_time": float(start),
                "stored_sample_time": float(sample_time[0]),
                "time_tolerance": time_tolerance,
                "force_time": float(end), "step": step,
                "roi": {"x": [float(x[0]), float(x[-1])],
                        "y": [float(y[0]), float(y[-1])]},
                "quantity_semantics": {
                    "speed": "CFD solver units",
                    "pressure": "CFD pressure with ROI mean removed (solver units)",
                }}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _current_trial_field_png(root: Path, profile: dict, rows: list[dict],
                             requested_sha256: str) -> bytes | None:
    """Render only the SHA-bound actual CFD start state; no predicted field."""
    evidence = _current_trial_field_evidence(root, profile, rows, requested_sha256)
    if evidence is None:
        return None
    with _CURRENT_FIELD_LOCK:
        if _CURRENT_FIELD_CACHE["sha256"] == evidence["sha256"]:
            return _CURRENT_FIELD_CACHE["png"]
        try:
            import io
            import numpy as np
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            path = (root / profile["base"]
                    / f"current_mpc_{evidence['field_time']:.1f}.npz")
            with np.load(path, allow_pickle=False) as packet:
                state, valid = packet["state"], packet["mask"][0].astype(bool)
                x, y = packet["x"], packet["y"]
            speed = np.where(valid, np.sqrt(state[0] ** 2 + state[1] ** 2), np.nan)
            pressure = np.where(valid, state[2], np.nan)
            figure = Figure(figsize=(11, 3.8), dpi=120, layout="constrained")
            canvas = FigureCanvasAgg(figure)
            for axis, values, title, cmap in (
                    (figure.add_subplot(1, 2, 1), speed,
                     "Actual CFD speed |U| (solver units)", "viridis"),
                    (figure.add_subplot(1, 2, 2), pressure,
                     "Actual CFD pressure, ROI mean removed (solver units)", "coolwarm")):
                image = axis.imshow(values, origin="lower", aspect="equal",
                                    extent=(x[0], x[-1], y[0], y[-1]), cmap=cmap)
                axis.set(xlabel="x (solver coordinates)", ylabel="y (solver coordinates)", title=title)
                figure.colorbar(image, ax=axis, shrink=.82)
            figure.suptitle(
                f"Actual OpenFOAM sampled ROI at cycle start t={evidence['field_time']:.1f}; "
                f"not an FNO-predicted field | sha256:{evidence['sha256'][:12]}…",
                fontsize=10)
            stream = io.BytesIO(); canvas.print_png(stream); payload = stream.getvalue()
            _CURRENT_FIELD_CACHE.update(sha256=evidence["sha256"], png=payload)
            return payload
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            return None


def _final_ppo_field_evidence(root: Path) -> dict | None:
    """Validate the one fixed final-PPO actual-CFD field; never accept a path."""
    base = root / "artifacts/exploratory_final_ppo_field_preview_20261006"
    result_path = base / "result.json"
    npz_path = base / "final_ppo_actual_cfd_160.4.npz"
    unit_path = base / "unit_evidence.json"
    manifest_path = base / "manifest.sha256.json"
    try:
        if (hashlib.sha256(result_path.read_bytes()).hexdigest()
                != _FINAL_PPO_FIELD_RESULT_SHA
                or hashlib.sha256(npz_path.read_bytes()).hexdigest()
                != _FINAL_PPO_FIELD_NPZ_SHA
                or hashlib.sha256(unit_path.read_bytes()).hexdigest()
                != _FINAL_PPO_FIELD_UNIT_EVIDENCE_SHA
                or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                != _FINAL_PPO_FIELD_MANIFEST_SHA):
            return None
        result = json.loads(result_path.read_text())
        unit = json.loads(unit_path.read_text())
        if (result.get("status") != "FINAL_PPO_ACTUAL_CFD_FIELD_PREVIEW_COMPLETE_NOT_ADMISSION"
                or result.get("actual_cfd") is not True
                or result.get("model_prediction") is not False
                or result.get("cfd_rerun") is not False
                or result.get("scientific_admission") is not False
                or result.get("npz_sha256") != _FINAL_PPO_FIELD_NPZ_SHA
                or result.get("result_sha256") != _FINAL_PPO_CFD_RESULT_SHA
                or result.get("review_sha256") != _FINAL_PPO_CFD_REVIEW_SHA
                or unit.get("owned_container_ids_after_cleanup") != []):
            return None
        for name in ("export_unit", "curator_unit"):
            state = unit[name]
            if (state.get("MainPID") != "0" or state.get("ActiveState") != "active"
                    or state.get("SubState") != "exited" or state.get("Result") != "success"
                    or state.get("ExecMainCode") != "1" or state.get("ExecMainStatus") != "0"
                    or state.get("MemoryMax") != "4294967296"
                    or state.get("MemorySwapMax") != "0"):
                return None
        import numpy as np
        with np.load(npz_path, allow_pickle=False) as packet:
            if set(packet.files) != {"state", "mask", "time", "x", "y"}:
                return None
            state, mask = packet["state"], packet["mask"]
            sample_time, x, y = packet["time"], packet["x"], packet["y"]
            evidence = result["evidence"]
            intended = float(evidence["intended_time"])
            tolerance = float(evidence["time_tolerance"])
            if (state.shape != (3, 128, 256) or state.dtype != np.float32
                    or mask.shape != (1, 128, 256) or mask.dtype != np.uint8
                    or sample_time.shape != (1,) or x.shape != (256,) or y.shape != (128,)
                    or x.dtype != np.float32 or y.dtype != np.float32
                    or not np.isfinite(state).all() or not np.isfinite(sample_time).all()
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not np.all((mask == 0) | (mask == 1))
                    or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)
                    or intended != 160.4 or tolerance != 3.0517578125e-05
                    or abs(float(sample_time[0]) - intended) > tolerance
                    or float(sample_time[0]) != float(evidence["stored_time"])
                    or list(state.shape) != evidence["state_shape"]
                    or abs(float(mask.mean()) - float(evidence["valid_fraction"])) > 1e-12):
                return None
        return {"verified": True, "sha256": _FINAL_PPO_FIELD_NPZ_SHA,
                "result_sha256": _FINAL_PPO_FIELD_RESULT_SHA,
                "unit_evidence_sha256": _FINAL_PPO_FIELD_UNIT_EVIDENCE_SHA,
                "manifest_sha256": _FINAL_PPO_FIELD_MANIFEST_SHA,
                "intended_time": intended, "stored_sample_time": float(sample_time[0]),
                "time_tolerance": tolerance, "actual_cfd": True,
                "model_prediction": False, "cfd_rerun": False,
                "roi": {"x": [float(x[0]), float(x[-1])],
                        "y": [float(y[0]), float(y[-1])]},
                "quantity_semantics": {
                    "speed": "CFD solver units",
                    "pressure": "CFD pressure with ROI mean removed (solver units)",
                }}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ImportError):
        return None


def _final_ppo_field_png(root: Path, requested_sha256: str) -> bytes | None:
    """Render the exact SHA-bound final-PPO actual CFD field, never an MPC field."""
    if requested_sha256 != _FINAL_PPO_FIELD_NPZ_SHA:
        return None
    evidence = _final_ppo_field_evidence(root)
    if evidence is None:
        return None
    with _FINAL_PPO_FIELD_LOCK:
        if _FINAL_PPO_FIELD_CACHE["sha256"] == requested_sha256:
            return _FINAL_PPO_FIELD_CACHE["png"]
        try:
            import io
            import numpy as np
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            path = (root / "artifacts/exploratory_final_ppo_field_preview_20261006"
                    / "final_ppo_actual_cfd_160.4.npz")
            with np.load(path, allow_pickle=False) as packet:
                state, valid = packet["state"], packet["mask"][0].astype(bool)
                x, y = packet["x"], packet["y"]
            speed = np.where(valid, np.sqrt(state[0] ** 2 + state[1] ** 2), np.nan)
            pressure = np.where(valid, state[2], np.nan)
            figure = Figure(figsize=(11, 3.8), dpi=120, layout="constrained")
            canvas = FigureCanvasAgg(figure)
            for axis, values, title, cmap in (
                    (figure.add_subplot(1, 2, 1), speed,
                     "PPO CFD: speed |U|", "viridis"),
                    (figure.add_subplot(1, 2, 2), pressure,
                     "PPO CFD: pressure (ROI mean removed)",
                     "coolwarm")):
                image = axis.imshow(values, origin="lower", aspect="equal",
                                    extent=(x[0], x[-1], y[0], y[-1]), cmap=cmap)
                axis.set(xlabel="x (solver coordinates)", ylabel="y (solver coordinates)",
                         title=title)
                colorbar = figure.colorbar(image, ax=axis, shrink=.82)
                colorbar.set_label("solver units")
            figure.suptitle(
                "Actual OpenFOAM sampled ROI for final PPO branch at t=160.4; "
                f"not a model prediction | sha256:{requested_sha256[:12]}…", fontsize=10)
            stream = io.BytesIO(); canvas.print_png(stream); payload = stream.getvalue()
            _FINAL_PPO_FIELD_CACHE.update(sha256=requested_sha256, png=payload)
            return payload
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            return None


def _exploratory_mpc_profile(registration: dict) -> dict:
    kind = registration.get("progress_kind", "exploratory_h2_feedback")
    return _EXPLORATORY_MPC_PROFILES[kind]


def _exploratory_mpc_terminal_review(root: Path, review_kind: str = "instantaneous_h2") -> bool:
    """Bind the independent short-window review, never confer admission."""
    if review_kind == "causal_history_h2":
        base = root / "artifacts/exploratory_causal_history_h2_real_cfd_20261006"
        files = {
            root / "docs/EXPLORATORY_CAUSAL_HISTORY_H2_TERMINAL_REVIEW_20261006.md": "9ceb4d58a66b549faa86834d15d0444d57c8fdcc2f2635f18859440a679d2098",
            base / "result.json": "74a28d45dce9b84ec5044700fe470390cde899a2fcf40a0b893c1b28817d99ca",
            base / "container_terminal_e59efa04750c.json": "6ab3d0319b4e4f04bb3498b0e302533f6628aec5603d4b2bb3fe010b5831ccd5",
            base / "container_terminal_e610fafa3753.json": "b0a8bb934ef1c222bb318e4a92ac8ff50b39e96ab6648b3ca0f67d148cc12b97",
        }
        try:
            for path, expected in files.items():
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    return False
            result = json.loads((base / "result.json").read_text())
            return (result.get("status") == "EXPLORATORY_REAL_CFD_CANONICAL_HISTORY_H2_COMPLETE_NOT_ADMISSION"
                    and result.get("selector_mode") == "canonical_causal_history_h2_v1"
                    and result.get("cycles") == 10
                    and result.get("physical_duration_D_over_U") == 1.0
                    and [row.get("selected_omega") for row in result.get("rows", [])] == [0.0] * 10
                    and all(result.get(key) is False for key in
                            ("scientific_admission", "ppo_executed", "hydrogym_solver_used",
                             "original_long_ar_gate_passed"))
                    and result.get("source_restart_unchanged") is True)
        except (OSError, ValueError, TypeError):
            return False
    if review_kind != "instantaneous_h2":
        return False
    base = root / "artifacts/exploratory_paired_h2_real_cfd_20261006"
    files = {
        root / "docs/EXPLORATORY_PAIRED_H2_TERMINAL_REVIEW_20261006.md": "8c600368d836e8c34c09e4ac3be1129ffcfb67fd3587e48e4e4feba33d711dcd",
        base / "result.json": "45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb",
        base / "container_terminal_257e7829da44.json": "d0036b3f7fd0fbc66b28ee8c7111a0951f25d045be41e842fb894e90e7f5fc52",
        base / "container_terminal_e49a8a5a3542.json": "cfe5e053c8bd8fac787607844d18ded16d4b823b5ead3106a6197080fab64819",
    }
    try:
        for path, expected in files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        result=json.loads((base/"result.json").read_text())
        return (result.get("status")=="EXPLORATORY_REAL_CFD_SHORT_H2_FEEDBACK_COMPLETE_NOT_ADMISSION"
                and result.get("cycles")==10 and result.get("physical_duration_D_over_U")==1.0
                and all(result.get(k) is False for k in ("scientific_admission","ppo_executed","hydrogym_solver_used","original_long_ar_gate_passed"))
                and result.get("source_restart_unchanged") is True)
    except (OSError,ValueError,TypeError):
        return False


def _exploratory_mpc_progress(root: Path, registration: dict, state: dict, matches: bool) -> dict:
    try:
        profile = _exploratory_mpc_profile(registration)
        if (registration.get("unit") != profile["unit"]
                or registration.get("invocation") != profile["invocation"]
                or state.get("InvocationID") != profile["invocation"]
                or registration.get("invocation") != state.get("InvocationID")):
            return {"verified":False}
        running = state.get("ActiveState") in ("active","activating") and state.get("SubState") in ("running","start") and state.get("MainPID") != "0" and matches
        if state.get("MainPID") != "0" and not matches:
            return {"verified":False}
        exited = all(state.get(k)==v for k,v in {"ActiveState":"active","SubState":"exited","MainPID":"0","Result":"success","ExecMainCode":"1","ExecMainStatus":"0"}.items())
        base=root/profile["base"]
        path=base/"progress.json"
        document=json.loads(path.read_text()) if path.exists() else {"rows":[],"completed_cycles":0}
        rows=document.get("rows",[]); count=document.get("completed_cycles")
        planned = profile.get("planned_cycles", 10)
        if type(count) is not int or not 0<=count<=planned or len(rows)!=count:
            return {"verified":False}
        if rows and (document.get("status") != profile["running_status"]
                     or document.get("identity",{}).get("k1_manifest_sha256") != "7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7"):
            return {"verified":False}
        latest=None
        actual_timeseries=[]
        for step,row in enumerate(rows,1):
            if row.get("step")!=step or abs(row.get("start_time",0)-(148+(step-1)*.1))>1e-8 or abs(row.get("end_time",0)-(148+step*.1))>1e-8:
                return {"verified":False}
            omega=row["selected_omega"]; previous=row["previous_omega"]
            predicted=row["selected_predicted_next_forces"]; actual=row["actual_endpoint_forces"]["mpc"]; zero=row["actual_endpoint_forces"]["zero"]
            values=[omega,previous,*predicted.values(),*actual.values(),*zero.values()]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values) or abs(omega)>.75+1e-12 or abs(omega-previous)>.1+1e-12:
                return {"verified":False}
            latest={"step":step,"omega":omega,"predicted_cd":predicted["front_cd"]+predicted["rear_cd"],"predicted_cl":predicted["rear_cl"],"actual_cd":actual["front_cd"]+actual["rear_cd"],"actual_cl":actual["rear_cl"],"zero_cd":zero["front_cd"]+zero["rear_cd"],"zero_cl":zero["rear_cl"]}
            actual_timeseries.append({"step":step,"field_time":float(row["start_time"]),
                                      "force_time":float(row["end_time"]),"omega":float(omega),
                                      "mpc_total_cd":float(actual["front_cd"]+actual["rear_cd"]),
                                      "zero_total_cd":float(zero["front_cd"]+zero["rear_cd"]),
                                      "mpc_rear_cl":float(actual["rear_cl"]),
                                      "zero_rear_cl":float(zero["rear_cl"])})
        current_field = _current_trial_field_evidence(root, profile, rows)
        recovered_metrics = (_long_h5_recovered_metrics(root)
                             if planned == 124 else None)
        reviewed=(exited and count==planned
                  and _exploratory_mpc_terminal_review(root, profile["terminal_review"]))
        postprocessing_failed=(count==planned and state.get("ActiveState")=="failed"
                               and state.get("SubState")=="failed"
                               and state.get("Result")=="exit-code")
        return {"verified":True,"mpc_trial":True,"running":running,"exited_success":exited,
                "completed_cycles":count,"latest":latest,"scientific_admission":False,
                "planned_cycles":planned,"planned_duration_D_over_U":planned*.1,
                "solver_cycles_complete":count==planned,
                "postprocessing_failed":postprocessing_failed,
                "current_cfd_field":current_field,"actual_timeseries":actual_timeseries,
                "recovered_metrics":recovered_metrics,
                "progress_kind":registration.get("progress_kind", "exploratory_h2_feedback"),
                "terminal_review_verified":reviewed,
                "terminal_review_pending":exited and not reviewed,"control_success_verified":False}
    except (OSError,ValueError,KeyError,TypeError):
        return {"verified":False}


def _registered_experiment_live(root: Path) -> dict:
    sampled = datetime.now(UTC).isoformat()
    try:
        registration = json.loads((root / "docs/LIVE_EXPERIMENT.json").read_text())
        if registration.get("progress_kind") == "two_segment_bridge":
            raw = subprocess.check_output(["systemctl", "--user", "show",
                "fluid-control-two-segment-frame-cpu-r4-20261006.service",
                *[arg for key in ("InvocationID","MainPID","ActiveState","SubState","Result","ExecMainCode","ExecMainStatus") for arg in ("-p",key)]], text=True, timeout=5)
            state = dict(line.split("=",1) for line in raw.splitlines() if "=" in line)
            return {**_two_segment_bridge(root, registration, state), "sampled_at_utc":sampled}
        def confined(name):
            p = Path(name)
            if p.is_absolute() or not (root / p).resolve().is_relative_to(root.resolve()):
                raise ValueError("registration path outside project")
            return root / p
        unit = registration["unit"]
        if not unit.startswith("fluid-control-") or not unit.endswith(".service") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-." for c in unit):
            raise ValueError("unexpected unit")
        launcher = confined(registration["launcher"])
        for key in ("approval", "launcher"):
            if hashlib.sha256(confined(registration[key]).read_bytes()).hexdigest() != registration[key+"_sha256"]:
                raise ValueError("registration identity mismatch")
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show", unit,
             *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        pid = state.get("MainPID", "0")
        matches = pid.isdigit() and pid != "0" and str(launcher).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if registration.get("progress_kind") in _EXPLORATORY_MPC_PROFILES:
            return {**_exploratory_mpc_progress(root,registration,state,matches),"sampled_at_utc":sampled}
        if registration.get("progress_kind") == "formal_evaluation":
            progress = _registered_formal_progress(root, registration, state, matches)
        else:
            log = confined(registration["log"]).read_text()
            progress = _parse_registered_progress(state, log, matches, registration)
        return {**progress, "review": _registered_terminal_review(root, registration, progress),
                "sampled_at_utc": sampled}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return {"verified": False, "sampled_at_utc": sampled}


def _fcp024_live(root: Path) -> dict:
    sampled = datetime.now(UTC).isoformat()
    try:
        approval = root / "docs/FC_P024_EXECUTION_APPROVAL_20261005.json"
        launcher = root / "artifacts/fcp024_response_scale_source_20261005_immutable/scripts/run_fcp024_response_scale_spark.sh"
        if hashlib.sha256(approval.read_bytes()).hexdigest() != "53926e4d2defe71f24f7ddb99b2a51706855997c94a112de2d50f462c6fda4cd" or hashlib.sha256(launcher.read_bytes()).hexdigest() != "683482e6f77bd81498f12ee9b5dd8ab6fdc0755c2b47da23a020f021f48ff705":
            return {"verified": False, "sampled_at_utc": sampled}
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show", "fluid-control-fcp024-response-scale-20261005.service",
             *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        pid = state.get("MainPID", "0")
        if state.get("InvocationID") != "b85dcf9d70e443fe92ae4ef5d72d3c76":
            return {"verified": False, "sampled_at_utc": sampled}
        matches = pid.isdigit() and pid != "0" and str(launcher).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if pid != "0" and not matches:
            return {"verified": False, "sampled_at_utc": sampled}
        running = state.get("ActiveState") == "active" and state.get("SubState") == "running" and matches
        terminal = state.get("ActiveState") == "active" and state.get("SubState") == "exited" and pid == "0" and state.get("Result") == "success" and state.get("ExecMainCode") == "1" and state.get("ExecMainStatus") == "0"
        return {"verified": True, "running": running, "exited_success": terminal, "sampled_at_utc": sampled, "admission": False}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"verified": False, "sampled_at_utc": sampled}


def _fcp023_terminal_review(root: Path) -> dict:
    try:
        result = root / "artifacts/fcp023_input_block_20261005/result.json"
        report = root / "docs/FC_P023_TERMINAL_REVIEW_20261005.md"
        raw = result.read_bytes()
        if hashlib.sha256(raw).hexdigest() != "adfdd9a86cedf75019b655fe360b64b09d1aa51ce3de2f9166d0d80e296007cb":
            return {"verified": False}
        if hashlib.sha256(report.read_bytes()).hexdigest() != "2b883509ed5dd62deb737ed586603080ba1c6a2719c42c09f443b20878c5bb7b":
            return {"verified": False}
        result_data = json.loads(raw)
        return {"verified": True, "local_support": result_data["comparison"]["local_support"],
                "scientific_admission": False, "experiment_id": "FC-E033"}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified": False}


def _fcp023_live(root: Path) -> dict:
    sampled = datetime.now(UTC).isoformat()
    try:
        approval = root / "docs/FC_P023_EXECUTION_APPROVAL_20261005.json"
        launcher = root / "artifacts/fcp023_input_block_source_20261005_immutable/scripts/run_fcp023_input_block_spark.sh"
        if hashlib.sha256(approval.read_bytes()).hexdigest() != "714db1f9d09b0ee7037953d6b807b3341cf7a736889d5c5fa98e5ae8c0116cf0":
            return {"verified": False, "sampled_at_utc": sampled}
        if hashlib.sha256(launcher.read_bytes()).hexdigest() != "1215d6e107639c3e902944e26a4c228729d102ee5b79d3b75297009d5571e924":
            return {"verified": False, "sampled_at_utc": sampled}
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show", "fluid-control-fcp023-input-block-20261005.service",
             *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        pid = state.get("MainPID", "0")
        matches = pid.isdigit() and pid != "0" and str(launcher).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        log = (root / "artifacts/fcp023_input_block_20261005/run.log").read_text()
        parsed = _parse_fcp023_live(state, log, matches)
        if parsed.get("exited_success"):
            parsed["terminal_review"] = _fcp023_terminal_review(root)
        return {**parsed, "sampled_at_utc": sampled}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"verified": False, "sampled_at_utc": sampled}


def _parse_fcp022_live(state: dict, log: str, process_matches: bool) -> dict:
    if state.get("InvocationID") != "f1f3f7b31e70440693da2661a12cfc04":
        return {"verified": False}
    pid = state.get("MainPID", "0")
    if pid != "0" and not process_matches:
        return {"verified": False}
    running = state.get("ActiveState") == "active" and state.get("SubState") == "running" and pid != "0" and process_matches
    terminal = (state.get("ActiveState") == "active" and state.get("SubState") == "exited"
                and pid == "0" and state.get("Result") == "success"
                and state.get("ExecMainCode") == "1" and state.get("ExecMainStatus") == "0")
    updates = {"A_zero": set(), "B_causal": set()}
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("event") != "arm_update_complete":
            continue
        arm, step = row.get("arm"), row.get("update")
        if arm not in updates or type(step) is not int or not 1 <= step <= 16 or step in updates[arm]:
            return {"verified": False}
        if arm == "B_causal" and len(updates["A_zero"]) != 16:
            return {"verified": False}
        updates[arm].add(step)
    if any(steps != set(range(1, len(steps)+1)) for steps in updates.values()):
        return {"verified": False}
    return {"verified": True, "running": running, "exited_success": terminal,
            "updates": {key: len(value) for key, value in updates.items()}, "admission": False}


def _fcp022_live(root: Path) -> dict:
    sampled = datetime.now(UTC).isoformat()
    try:
        approval = root / "docs/FC_P022_EXECUTION_APPROVAL_20261005.json"
        launcher = root / "artifacts/fcp022_causal_conditioning_source_20261005_immutable/scripts/run_fcp022_causal_conditioning_spark.sh"
        if hashlib.sha256(approval.read_bytes()).hexdigest() != "273fe63f097049fe28f9d3f6f241308a95eefb6fab23ca2dbe0d23559043cf78":
            return {"verified": False, "sampled_at_utc": sampled}
        if hashlib.sha256(launcher.read_bytes()).hexdigest() != "257382621faa554888c0eca5501bb11e86a7d529b6649d4d8665a705c8502f88":
            return {"verified": False, "sampled_at_utc": sampled}
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show", "fluid-control-fcp022-causal-conditioning-20261005.service",
             *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        pid = state.get("MainPID", "0")
        matches = pid.isdigit() and pid != "0" and str(launcher).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        log = (root / "artifacts/fcp022_causal_conditioning_20261005/run.log").read_text()
        return {**_parse_fcp022_live(state, log, matches), "sampled_at_utc": sampled}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"verified": False, "sampled_at_utc": sampled}


def _parse_fcp020_live(state: dict, log: str, process_matches: bool) -> dict:
    if state.get("InvocationID") != "4567f6d393414bba8baf2239d16960a7":
        return {"verified": False}
    running = (state.get("SubState") in {"start", "running"}
               and state.get("MainPID", "0") != "0" and process_matches)
    terminal = (state.get("ActiveState") == "active" and state.get("SubState") == "exited"
                and state.get("MainPID") == "0" and state.get("Result") == "success"
                and state.get("ExecMainStatus") == "0" and state.get("ExecMainCode") == "1")
    if state.get("MainPID", "0") != "0" and not process_matches:
        return {"verified": False}
    updates = {"A_original": set(), "B_symmetric_tail": set()}
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("event") != "arm_update_complete":
            continue
        arm, step = row.get("arm"), row.get("update")
        if arm not in updates or type(step) is not int or not 1 <= step <= 16 or step in updates[arm]:
            return {"verified": False}
        updates[arm].add(step)
    if any(steps != set(range(1, len(steps)+1)) for steps in updates.values()):
        return {"verified": False}
    return {"verified": True, "running": running, "exited_success": terminal,
            "updates": {k: len(v) for k, v in updates.items()}, "admission": False}


def _fcp020_live(root: Path) -> dict:
    try:
        approval = root / "docs/FC_P020_EXECUTION_APPROVAL_20261005.json"
        if hashlib.sha256(approval.read_bytes()).hexdigest() != "ab69dfc9559a7e5458aba08c85b57d8d4aa41529ac7980e28f739ef667fab30a":
            return {"verified": False}
        launcher = root / "artifacts/fcp020_symmetric_statistics_source_20261005_immutable/scripts/run_fcp020_symmetric_statistics_spark.sh"
        if hashlib.sha256(launcher.read_bytes()).hexdigest() != "f2d2dba8fc7668beeb346ab89e935fae402e42406b5b50898fbbd32dc2067e11":
            return {"verified": False}
        fields = ("InvocationID", "MainPID", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus")
        raw = subprocess.check_output(["systemctl", "--user", "show", "fluid-control-fcp020-symmetric-statistics-20261005.service",
             *[arg for key in fields for arg in ("-p", key)]], text=True, timeout=5)
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        pid = state.get("MainPID", "0")
        matches = pid.isdigit() and pid != "0" and str(launcher).encode() in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        log = (root / "artifacts/fcp020_symmetric_statistics_20261005/run.log").read_text()
        return {**_parse_fcp020_live(state, log, matches), "sampled_at_utc": datetime.now(UTC).isoformat(),
                "terminal_result": _fcp020_result(root)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"verified": False}


def _fcp020_result(root: Path) -> dict:
    try:
        raw = (root / "artifacts/fcp020_symmetric_statistics_20261005/result.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != "a7c0d0c41b35391e22d07fb223a5ed243891ccdd4759815e9bf08b82670b5042":
            return {"verified": False}
        result = json.loads(raw)
        checks = result["comparison"]
        if checks["conclusion"] != "LOCAL_CONDITIONS_NOT_MET" or checks["local_support"] is not False:
            return {"verified": False}
        return {"verified": True, "local_support": False, "admission": False,
                "h1_bias_delta": checks["checks"]["h1"]["statistics"]["bias_mse"]["delta_initial"]}
    except (OSError, ValueError, TypeError, KeyError):
        return {"verified": False}


def _fcp018_formal_result(root: Path) -> dict:
    base = root / "artifacts/fcp018_reduced_rate_training_20261005/posteval_fc_p018"
    try:
        raw = (base / "receipt.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != "d4d3f85a79e31d50866bb8dbd23ec90e0453cde39ef6b314e3db80344d33869c":
            return {"verified": False}
        gate_raw = (base / "development_gate.json").read_bytes()
        if hashlib.sha256(gate_raw).hexdigest() != json.loads(raw)["sha256"]["development_gate.json"]:
            return {"verified": False}
        gate = json.loads(gate_raw)
        branches = gate["window_gate"]["branches"]
        if gate["status"] != "DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL" or len(branches) != 6:
            return {"verified": False}
        return {"verified": True, "admission": False, "joint_pass": sum(x["joint_pass"] is True for x in branches),
                **{label: sum(x["metric_pass"][key] is True for x in branches) for label, key in
                   (("cd_pass", "total_cd"), ("rms_pass", "rear_cl_fluctuation_rms"), ("mean_pass", "rear_cl_mean"))}}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified": False}


def _fcp018_formal_live(root: Path) -> dict:
    try:
        import watch_training_evaluation_state as monitor
        task = monitor.p018_formal_authority(
            root, monitor.unit_state(monitor.P018_FORMAL_UNIT))
        return {"observed": True, "sampled_at_utc": task["observed_utc"],
                "task": task, "admission": False}
    except (ImportError, AttributeError, OSError, subprocess.SubprocessError, ValueError, TypeError):
        return {"observed": False, "admission": False}


def _parse_fcp018_live(output: str, log: str, log_age: float) -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if (fields.get("InvocationID") != "1ca4654aab074278bb2efdfff8dbc1eb"
            or FCP018_LAUNCHER + " --execute" not in fields.get("ExecStart", "")
            or "FCP018_APPROVAL_SHA256=" + FCP018_APPROVAL not in fields.get("ExecStart", "")):
        return {"verified": False, "admission": False}
    try:
        pid = int(fields.get("MainPID", ""))
    except ValueError:
        return {"verified": False, "admission": False}
    updates = 0
    for line in log.splitlines():
        try:
            row = json.loads(line)
            if (row.get("event") == "accumulation_update" and type(row.get("update")) is int
                    and 1 <= row["update"] <= 171 and row.get("consumed_windows") == 8 * row["update"]
                    and row.get("actual_learning_rate") == 1.5625e-7
                    and row.get("training_protocol_sha256") == FCP018_PROTOCOL):
                updates = max(updates, row["update"])
        except (ValueError, TypeError, AttributeError):
            continue
    return {"verified": True, "running": fields.get("ActiveState") == "active"
            and fields.get("SubState") == "running" and pid > 0,
            "updates": updates, "consumed_windows": updates * 8,
            "progress_fresh": 0 <= log_age <= 300, "log_age_seconds": log_age,
            "admission": False, "independent_terminal_audit": False}


def _fcp018_training(root: Path) -> dict:
    base = root / "artifacts/fcp018_reduced_rate_training_20261005"
    try:
        for path, expected in {
            root / "docs/FC_P018_RUNNING_EXECUTION_20261005.json": "c040eae25fa31a98164e08e10dc4f007eb6a3ef38329ad0dcfaddd734c7eb53c",
            base / "execution_approval.json": FCP018_APPROVAL,
            base / "training_protocol.json": FCP018_PROTOCOL,
            base / "immutable_launcher.sh": "102b8c6efe54b2e868e4973315260b6ed73b8e24e770af4fa2f04876af36c731",
            root / FCP018_LAUNCHER: "102b8c6efe54b2e868e4973315260b6ed73b8e24e770af4fa2f04876af36c731",
        }.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return {"verified": False}
        state = subprocess.run(["systemctl", "--user", "show", "fluid-control-fcp018-reduced-rate-20261005.service",
            "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "InvocationID", "-p", "ExecStart"],
            capture_output=True, text=True, timeout=2, check=False)
        if state.returncode != 0: return {"verified": False}
        return {**_parse_fcp018_live(state.stdout, _tail_text(base / "run.log"),
                    time.time() - (base / "run.log").stat().st_mtime),
                "sampled_at_utc": datetime.now(UTC).isoformat()}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return {"verified": False}


def _parse_fcp015_live(output: str, log: str) -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    try:
        pid = int(fields.get("MainPID", "0"))
    except ValueError:
        pid = 0
    bound = (fields.get("InvocationID") == "7842742926284d0c94b0383163d5dc0b"
             and "fcp015_window_accumulation_source_20261005_immutable/scripts/run_fcp015_window_accumulation_spark.sh --execute" in fields.get("ExecStart", ""))
    updates = 0
    for line in log.splitlines():
        try:
            row = json.loads(line)
            if (row.get("event") == "accumulation_update" and type(row.get("update")) is int
                    and 1 <= row["update"] <= 171 and row.get("consumed_windows") == 8 * row["update"]):
                updates = max(updates, row["update"])
        except (ValueError, TypeError, AttributeError):
            continue
    return {"running": bool(bound and fields.get("ActiveState") == "active"
                            and fields.get("SubState") == "running" and pid > 0),
            "updates": updates, "consumed_windows": updates * 8, "admission": False}


def _parse_fcp015_posteval(output: str) -> dict:
    fields = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    bound = (fields.get("InvocationID") == "c41e61fcaa5f49e3be9e0092c1e19bc3"
             and "fcp015_formal_supervisor_bf3668d852f8_immutable/supervise_fcp015_formal.py" in fields.get("ExecStart", ""))
    try:
        pid = int(fields.get("MainPID", "0"))
    except ValueError:
        return {"verified": False}
    if not bound:
        return {"verified": False}
    return {"verified": True, "running": fields.get("ActiveState") == "active"
            and fields.get("SubState") == "running" and pid > 0, "admission": False}


def _fcp015_posteval(root: Path) -> dict:
    base = root / "artifacts/fcp015_window_accumulation_training_20261005"
    try:
        for path, expected in {
            base / "completion_receipt.json": "9c27e5ebe104a8988f274b9c3e8cd0d6728838c1d6aa34daff9d25d310605fdf",
            base / "dual_reload_receipt.json": "925a7dc0c1a1be0157afd5838018b1a3bccd74f0083b2d70718e8a65e8c2b18c",
            root / "docs/FC_P015_FORMAL_EVALUATION_APPROVAL_20261005.json": "991d3e4e5d5ac439824b9f8cfcfd1327be11460f79901053e57c83919df7730e",
        }.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return {"verified": False}
        state = subprocess.run(["systemctl", "--user", "show", "fluid-control-fcp015-posteval-20261005.service",
                                "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "InvocationID", "-p", "ExecStart"],
                               capture_output=True, text=True, timeout=2, check=False)
        return _parse_fcp015_posteval(state.stdout) if state.returncode == 0 else {"verified": False}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"verified": False}


def _fcp015_training(root: Path) -> dict:
    base = root / "artifacts/fcp015_window_accumulation_training_20261005"
    try:
        for name, expected in {
            "execution_approval.json": "5f42527e4b032b0c3aff4180aa34a6570ec1fcb3945ec7302934d8386c4702d6",
            "running_execution_evidence.json": "320dda2539952d602d2e5c21f21d731c2c6e468063cfb85d9db5215e6ce16836",
        }.items():
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
                return {"verified": False}
        state = subprocess.run(["systemctl", "--user", "show", "fluid-control-fcp015-window-accumulation-20261005.service",
                                "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "InvocationID", "-p", "ExecStart"],
                               capture_output=True, text=True, timeout=2, check=False)
        log_path = base / "run.log"
        log = log_path.read_bytes()[-262144:].decode(errors="replace")
        return {"verified": True, "sampled_at_utc": datetime.now(UTC).isoformat(),
                "log_age_seconds": max(0.0, time.time() - log_path.stat().st_mtime),
                **_parse_fcp015_live(state.stdout, log), "evaluation": _fcp015_posteval(root)}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"verified": False}


def _fcp013_training(root: Path) -> dict:
    if FCP013_TRAINING_APPROVAL_SHA is None:
        return {"ready": False}
    try:
        raw = (root / "docs/FC_P013_TRAINING_EXECUTION_APPROVAL_20261005.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != FCP013_TRAINING_APPROVAL_SHA:
            return {"ready": False}
        recovery = (root / "docs/FC_P013_RECOVERY_APPROVAL_20261005.md").read_bytes()
        if hashlib.sha256(recovery).hexdigest() != "6579dfbf97d6a7fbe6b74fa537addc52fe7b4cf266368772ce52965d9d973095":
            return {"ready": False}
    except OSError:
        return {"ready": False}
    unit = "fluid-control-fcp013-training-r2-20261005.service"
    try:
        state = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID", "-p", "ExecStart", "-p", "InvocationID"], capture_output=True, text=True, timeout=2, check=False)
        journal = subprocess.run(["journalctl", "--user", "-u", unit, "-n", "40", "-o", "json", "--no-pager"], capture_output=True, text=True, timeout=2, check=False)
        status = _parse_fcp013_live(state.stdout + "\n" + journal.stdout, time.time(), unit)
    except (OSError, subprocess.SubprocessError):
        status = _parse_fcp013_live("", time.time())
    return {"ready": True, "attempt": 2, "unit": unit, "sampled_at_utc": datetime.now(UTC).isoformat(),
            "terminal": _fcp013_terminal_progress(root), **status}


def _fcp012_diagnostic(root: Path) -> dict:
    try:
        raw = (root / "artifacts/fcp012_decoder_gradient_diagnostic_20261005/diagnostic/result.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != "4142cdc5c67ff04d50f2921887e01034a9ea7d09a2865ac286b52d1c911d814d":
            return {"verified": False}
        result = json.loads(raw)
        counts = {}
        for model in ("p009_parent", "fcp011_decoder_tail"):
            rows = [row for row in result["rows"] if row["model"] == model and row["global_index"] != 160]
            pairs = [row["gradient_groups"]["complete"]["field_vs_weighted_force"] for row in rows]
            if len(pairs) != 5:
                return {"verified": False}
            counts[model] = {"scale_signal_windows": sum(p["left_to_right_norm_ratio"] > 10 for p in pairs),
                             "conflict_signal_windows": sum(p["cosine"] < -0.2 for p in pairs)}
        return {"verified": True, "window_count": 12, "models": counts, "admission": False}
    except (OSError, ValueError, KeyError, TypeError):
        return {"verified": False}


def _fcp011_training(root: Path) -> dict:
    approval = root / "docs/FC_P011_TRAINING_EXECUTION_APPROVAL_20261005.json"
    try:
        if hashlib.sha256(approval.read_bytes()).hexdigest() != "1cd0dce1bb4ea4c2e580e147a0a440f2d528ea16e914479a629eb8185709338a":
            return {"ready": False}
    except OSError:
        return {"ready": False}
    arms = []
    for worker, scope, label in ((False, "head-only", "主节点 A：只训练升力输出"),
                                  (True, "decoder-tail", "辅助节点 B：增加解码器训练")):
        unit = f"fluid-control-fcp011-{scope}-training-20261005.service"
        manager = "systemctl" if worker else "systemctl --user"
        journal = "journalctl" if worker else "journalctl --user"
        script = f"{manager} show {unit} -p ActiveState -p MainPID -p ExecStart; {journal} -u {unit} -n 24 -o json --no-pager"
        script += f"; echo FCP011_FORMAL_STATUS; {manager} show fluid-control-fcp011-{scope}-posteval-20261005.service -p ActiveState -p MainPID -p ExecStart"
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=2", "USER@WORKER_HOST", script] if worker else ["sh", "-c", script]
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=4, check=False)
            training_output, _, formal_output = response.stdout.partition("FCP011_FORMAL_STATUS\n")
            status = _parse_fcp011_live(training_output, time.time())
            formal = _parse_fcp011_formal(formal_output, scope)
        except (OSError, subprocess.SubprocessError):
            status = _parse_fcp011_live("", time.time())
            formal = _parse_fcp011_formal("", scope)
        arms.append({"label": label, **status, "terminal": _fcp011_terminal(root, scope), "formal": formal})
    return {"ready": True, "sampled_at_utc": datetime.now(UTC).isoformat(), "arms": arms,
            "admission": False, "fixed_window_diagnostics_persisted": all(a["terminal"]["verified"] for a in arms)}


def _fcp009_formal_status(root: Path) -> dict:
    base = root / "artifacts/fcp009_joint_force_row_candidate_20261005/posteval_fc_p009"
    model = "dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31"
    stage = "validation10"
    for step, following in (("validation10", "dynamic6"), ("dynamic6", "force_window")):
        receipt = _read_json(base / "step_receipts" / (step + ".json"), {})
        if isinstance(receipt, dict) and receipt.get("status") == "FC_P009_POSTEVAL_STEP_COMPLETE" and receipt.get("checkpoint_sha256") == model and receipt.get("step") == step:
            stage = following
        else:
            break
    complete = _read_json(base / "receipt.json", {})
    verified_fail = False
    try:
        verified_fail = (
            hashlib.sha256((base / "receipt.json").read_bytes()).hexdigest()
            == "ac5c0dd047c90fddba884b77cd82bbe4f5147123d0f2f4455fb1b938607e231c"
            and hashlib.sha256((base / "development_gate.json").read_bytes()).hexdigest()
            == "1a7526e705b4ed3c56c4facdb6a8ee2e9b26f9f4fec3060df5823a4104499d7e"
        )
    except OSError:
        pass
    return {"service_state": _service_state("fluid-control-fcp009-posteval-20261005.service"),
            "verified_fail": verified_fail,
            "stage": stage,
            "complete_recorded": isinstance(complete, dict) and complete.get("status") == "FC_P009_POSTEVAL_COMPLETE" and complete.get("checkpoint_sha256") == model,
            "sampled_at_utc": datetime.now(UTC).isoformat(), "admission": False}


def _joint_readout_diagnostic(root: Path) -> dict:
    """Show measured diagnostic results only, with immutable identity binding."""
    base = root / "artifacts/fcp009_free_ar_force_readout_cache_20261005"
    try:
        path = base / "cpu_joint_50_50_analysis.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != FCP009_JOINT_SHA:
            return {"ready": False}
        result = json.loads(path.read_text())
        if result.get("status") != "FC_P009_FIXED_50_50_JOINT_READOUT_DIAGNOSTIC_COMPLETE" or result.get("source_result_sha256") != "1321c30a1e12172b85b269405c208981390f952bdf6f03a7fe2dd21f3bb91daf":
            return {"ready": False}
        rows = []
        for horizon, label in (("H1", "单步"), ("H100", "连续第100步")):
            for channel, channel_label in (("total_cd", "总阻力 Cd"), ("rear_cl", "后柱升力 Cl")):
                before = result["reports"]["parent_predict_free_ar"]["pooled"]["relative_horizons"][horizon]["physical"][channel]["mae"]
                after = result["reports"]["joint_fit_predict_free_ar"]["pooled"]["relative_horizons"][horizon]["physical"][channel]["mae"]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (before, after)):
                    return {"ready": False}
                rows.append({"label": label + " / " + channel_label, "before": before, "after": after})
        return {"ready": True, "rows": rows, "admission": False, "candidate_verified": False,
                "formal": _fcp009_formal_status(root)}
    except (OSError, ValueError, KeyError, TypeError):
        return {"ready": False}


def _free_ar_diagnostic(root: Path) -> dict:
    """Bounded progress observation; process liveness is never inferred from logs."""
    path = root / "artifacts/fcp009_free_ar_force_readout_cache_20261005/run.log"
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 65536))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if not isinstance(item, dict) or item.get("event") != "fcp009_extract_progress":
                continue
            windows, rows = item.get("windows"), item.get("rows")
            if type(windows) is not int or not 0 <= windows <= 1368 or type(rows) is not int or rows != windows * 100:
                continue
            return {"ready": True, "windows": windows, "rows": rows,
                    "finite": item.get("finite") is True,
                    "service_state": _service_state("fluid-control-fcp009-free-ar-cache-20261005.service"),
                    "sampled_at_utc": datetime.now(UTC).isoformat(),
                    "progress_age_seconds": max(0, datetime.now(UTC).timestamp() - path.stat().st_mtime),
                    "admission": False}
    except OSError:
        pass
    return {"ready": False}


def _fcp008_service_state() -> str:
    """Observe all recovery generations without mistaking old failures for live work."""
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "list-units", "--all", "--plain", "--no-legend",
             "--no-pager", "--type=service", "fluid-control-fcp008-posteval*.service"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode != 0:
            return "unknown"
        states = []
        for line in proc.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 4 and re.fullmatch(r"fluid-control-fcp008-posteval[-a-z0-9]*\.service", fields[0]):
                states.append(fields[2])
        if "active" in states:
            return "active"
        return "failed" if "failed" in states else "inactive"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _full_train_calibration(root: Path) -> dict:
    """Latest calibrated candidate; process state is sampled, not inferred from files."""
    base = root / FCP008
    try:
        path = base / "candidate_build/result.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != FCP008_RESULT_SHA:
            return {"ready": False}
        result = json.loads(path.read_text())
        rows = []
        for channel in ("rear_cd", "rear_cl"):
            before = result["full_train_parent_native_metrics_physical"][channel]["mae"]
            after = result["full_train_native_metrics_physical"][channel]["mae"]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in (before, after)):
                return {"ready": False}
            rows.append({"channel": channel, "before": before, "after": after})
        out = base / "posteval_fc_p008"
        stage = "validation10"
        if (out / "step_receipts/validation10.json").is_file():
            stage = "dynamic6"
        if (out / "step_receipts/dynamic6.json").is_file():
            stage = "force_window"
        receipt = _read_json(out / "receipt.json", {})
        complete = receipt.get("status") == "FC_P008_POSTEVAL_COMPLETE" and receipt.get("checkpoint_sha256") == result["candidate_model_sha256"]
        validation = None
        gate_path = out / "validation10/endpoint_gate.json"
        step = _read_json(out / "step_receipts/validation10.json", {})
        if gate_path.is_file() and step.get("sha256", {}).get("validation10/endpoint_gate.json") == hashlib.sha256(gate_path.read_bytes()).hexdigest():
            gate = _read_json(gate_path, {})
            delta = gate.get("h100_start0_action_difference", {})
            value, maximum = delta.get("pairwise_delta_cd_mae"), delta.get("predeclared_delta_cd_mae_maximum")
            if gate.get("checkpoint_sha256") == result["candidate_model_sha256"] and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in (value, maximum)):
                validation = {"status": gate.get("status"), "delta_cd_mae": value, "maximum": maximum}
        return {"ready": True, "rows": rows, "stage": stage,
                "formal_complete": complete, "admission": False,
                "validation": validation,
                "service_state": _fcp008_service_state(),
                "sampled_at_utc": datetime.now(UTC).isoformat()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"ready": False}


def _fixed_feature_readout(root: Path) -> dict:
    """Expose only completed, hash-bound diagnostic metrics, never admission."""
    base = root / FIXED_READOUT
    try:
        receipt = json.loads((base / "completion_receipt.json").read_text())
        result_path = base / "result_bundle/result.json"
        cache_path = base / "result_bundle/fixed_features.npz"
        for path, key in ((result_path, "result_sha256"), (cache_path, "cache_sha256")):
            if not path.resolve().is_relative_to(base.resolve()):
                return {"ready": False}
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get(key):
                return {"ready": False}
        result = json.loads(result_path.read_text())
        if receipt.get("status") != "FCP003C_FIXED_FEATURE_FORCE_READOUT_V2_EXECUTION_COMPLETE_NOT_ADMISSION":
            return {"ready": False}
        if result.get("status") != "FCP003C_FIXED_FEATURE_FORCE_READOUT_DIAGNOSTIC_COMPLETE" or result.get("model_sha256") != C_FINAL_SHA:
            return {"ready": False}
        for item in (receipt, result):
            if item.get("optimizer_steps") != 0 or any(item.get(k) is not False for k in ("candidate_saved", "validation_accessed", "frozen_test_accessed", "ppo_executed")):
                return {"ready": False}
        precision = result["numerical_protocol"]["precision_effective"]
        if precision["float32_matmul_precision"] != "highest" or precision["cuda_matmul_allow_tf32"] is not False or precision["cudnn_allow_tf32"] is not False:
            return {"ready": False}
        if result["model_tensor_state_sha256_before"] != result["model_tensor_state_sha256_after"] or result["field_repeat_bitwise_identical"] is not True:
            return {"ready": False}
        rows = []
        for panel in ("prefix_targets_1_100", "late_targets_101_200"):
            row = {"panel": panel}
            for channel in ("rear_cd", "rear_cl"):
                for side, key in (("before", "original_readout_metrics_physical"), ("after", "fitted_readout_metrics_physical")):
                    metric = result[key][panel]["action"][channel]
                    value = metric["mae"]
                    if metric["count"] != 800 or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        return {"ready": False}
                    row[f"{channel}_{side}"] = value
            rows.append(row)
        return {"ready": True, "rows": rows, "path": str(FIXED_READOUT), "admission": False}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"ready": False}


def _latest_evidence(root: Path) -> dict:
    """Load only finalized evidence; missing/incomplete artifacts remain explicit."""
    return {
        "v4_validation_decision": _read_json(root / V4_VALIDATION_DECISION, None),
        "v4_window_mean_cd": _read_json(root / V4_WINDOW_MEAN_CD, None),
        "v3_parent_window_mean_cd": _read_json(
            root / V3_PARENT_WINDOW_MEAN_CD, None
        ),
        "two_phase_alternating": _read_json(root / TWO_PHASE_ALTERNATING, None),
        "long_dwell075": _read_json(root / LONG_DWELL075, None),
        "low_action_phase94_canonical": _read_json(
            root / LOW_ACTION_PHASE94_CANONICAL, None
        ),
    }


def _pooled_terminal_nrmse(rows: object) -> float | None:
    if not isinstance(rows, list) or not rows:
        return None
    try:
        numerator = sum(
            int(row["segments"]) * float(row["total_drag_rmse"]) ** 2
            for row in rows
        )
        denominator = sum(
            int(row["segments"]) * float(row["total_drag_target_rms"]) ** 2
            for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return None
    if denominator <= 0 or not math.isfinite(numerator + denominator):
        return None
    return math.sqrt(numerator / denominator)


def _low_action_fno_summary(root: Path) -> dict | None:
    pairwise = _read_json(root / LOW_ACTION_FNO_PAIRWISE, None)
    if pairwise is None:
        return None
    if pairwise.get("status") != "LOW_ACTION_PAIRWISE_FNO_H100_AUDIT_COMPLETE":
        return None
    models = {}
    for result_key, directory in (
        ("v3_parent", "v3_h20_parent"),
        ("v4_candidate", "v4_h20_candidate"),
    ):
        evaluation = _read_json(
            root / LOW_ACTION_FNO_ROOT / directory / "evaluation.json", None
        )
        pooled = _read_json(
            root / LOW_ACTION_FNO_ROOT / directory / "pooled_audit.json", None
        )
        pair = pairwise.get("models", {}).get(result_key)
        if (
            not isinstance(evaluation, dict)
            or evaluation.get("split") != "validation"
            or evaluation.get("action_mode") != "observed"
            or not isinstance(pooled, dict)
            or pooled.get("split") != "validation"
            or not isinstance(pair, dict)
        ):
            return None
        try:
            endpoint = pooled["horizons"]["100"]["pooled"]
            summary = evaluation["summary"]["100"]
            model_mae = float(summary["total_drag_mae"])
            persistence_mae = float(summary["persistence_total_drag_mae"])
            values = {
                "h100_pooled_total_drag_nrmse": float(
                    endpoint["total_drag_nrmse_pooled"]
                ),
                "h100_total_drag_mae": model_mae,
                "h100_persistence_total_drag_mae": persistence_mae,
                "h100_mae_minus_persistence": model_mae - persistence_mae,
                "strict_start0_pairwise_delta_absolute_error": float(
                    pair["strict_common_initial_pairwise_absolute_error"]
                ),
                "strict_start0_ranking_correct": pair[
                    "strict_common_initial_ranking_correct"
                ],
            }
        except (KeyError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            for key, value in values.items()
            if key != "strict_start0_ranking_correct"
        ):
            return None
        models[result_key] = values
    return {
        "status": pairwise["status"],
        "split": "validation",
        "models": models,
        "scope": "H100 endpoint validation diagnostic; not control success",
    }


def _host_output(worker: bool) -> str:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "USER@WORKER_HOST", HOST_COMMAND] if worker else ["sh", "-c", HOST_COMMAND]
    result = subprocess.run(command, capture_output=True, text=True, timeout=6, check=True)
    return result.stdout


def _primary_cuda_free_gib() -> float | None:
    """Sample actual immediate CUDA headroom; GB10 MemAvailable can differ."""
    try:
        result = subprocess.run(
            ["python3", "-c", "import torch; print(torch.cuda.mem_get_info(0)[0]/1024**3)"],
            capture_output=True, text=True, timeout=8, check=True,
        )
        return float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _matched_start_progress(lines: list[str]) -> list[dict]:
    """Parse worker solver tails; transfer is audited separately on Spark."""
    observed: dict[str, tuple[float | None, bool, bool]] = {}
    try:
        start = lines.index("__MATCHED_START__") + 1
    except ValueError:
        start = len(lines)
    for line in lines[start:]:
        fields = line.split("|")
        if len(fields) != 4 or fields[0] not in MATCHED_START_CASES:
            continue
        try:
            last_time = float(fields[1]) if fields[1] else None
        except ValueError:
            last_time = None
        observed[fields[0]] = (last_time, fields[2] == "1", fields[3] == "1")

    rows = []
    for case, (label, restart_time) in MATCHED_START_CASES.items():
        last_time, active, ended = observed.get(case, (None, False, False))
        iteration = 0
        if last_time is not None:
            iteration = round((last_time - restart_time) / MATCHED_START_DT)
            iteration = max(0, min(MATCHED_START_ITERATIONS, iteration))
        if ended:
            status = "complete"
            iteration = MATCHED_START_ITERATIONS
        elif active:
            status = "running"
        elif last_time is None:
            status = "pending"
        else:
            status = "stopped_incomplete"
        rows.append(
            {
                "case": case,
                "label": label,
                "status": status,
                "iteration": iteration,
                "iterations_total": MATCHED_START_ITERATIONS,
                "last_solver_time": last_time,
                "solver_finished": ended,
            }
        )
    return rows


def _matched_start_transfer_receipts(root: Path) -> list[dict]:
    """Fail-closed validation of Spark-local raw-transfer receipts."""
    rows = []
    for case, (label, _) in MATCHED_START_CASES.items():
        path = root / MATCHED_START_RECEIPTS / f"{case}.json"
        receipt = _read_json(path, None)
        checks = {
            "status": isinstance(receipt, dict)
            and receipt.get("status") == "RAW_TRANSFER_VERIFIED",
            "case": isinstance(receipt, dict) and receipt.get("case") == case,
            "phase_manifest_sha256": isinstance(receipt, dict)
            and receipt.get("phase_manifest_sha256")
            == MATCHED_START_PHASE_MANIFEST_SHA256,
        }
        rows.append(
            {
                "case": case,
                "label": label,
                "raw_transfer_verified": all(checks.values()),
                "checks": checks,
                "receipt_path": str(MATCHED_START_RECEIPTS / f"{case}.json"),
            }
        )
    return rows


def _matched_start_pipeline_status(root: Path) -> list[dict]:
    """Read small markers and path metadata without opening VTK or HDF5 data."""
    rows = []
    for case, (label, _) in MATCHED_START_CASES.items():
        vtk_path = root / MATCHED_START_VTK_READY / f"{case}.json"
        vtk = _read_json(vtk_path, None)
        vtk_checks = {
            "status": isinstance(vtk, dict) and vtk.get("status") == "VTK_READY",
            "case": isinstance(vtk, dict) and vtk.get("case") == case,
            "frames": isinstance(vtk, dict) and vtk.get("frames") == 801,
        }
        staging = root / MATCHED_START_STAGING / case / "train"
        hdf = staging / f"{case}.h5"
        temporary = staging / f"{case}.h5.tmp"
        log = root / MATCHED_START_CURATOR_LOGS / f"{case}.log"
        hdf_exists = hdf.is_file()
        temporary_exists = temporary.exists()
        if hdf_exists and temporary_exists:
            hdf_status = "conflict"
        elif hdf_exists:
            hdf_status = "complete"
        elif temporary_exists:
            hdf_status = "writing"
        elif log.is_file():
            hdf_status = "started"
        else:
            hdf_status = "pending"
        rows.append(
            {
                "case": case,
                "label": label,
                "vtk_ready": all(vtk_checks.values()),
                "vtk_checks": vtk_checks,
                "vtk_marker_path": str(MATCHED_START_VTK_READY / f"{case}.json"),
                "hdf_staging_status": hdf_status,
                "hdf_staging_path": str(
                    MATCHED_START_STAGING / case / "train" / f"{case}.h5"
                ),
                "curator_log_exists": log.is_file(),
            }
        )
    return rows


def _matched_start_physics_summary(root: Path) -> dict | None:
    """Reduce the finalized nine-case JSON without exposing branch time series."""
    data = _read_json(root / MATCHED_START_PHYSICS_SUMMARY, None)
    if not isinstance(data, dict) or data.get("status") != (
        "MATCHED_START_9_CASE_TRAIN_COMMISSIONING_PHYSICS_SUMMARY"
    ):
        return None
    phases = data.get("phases")
    if not isinstance(phases, dict) or any(
        not isinstance(row, dict) or row.get("split") != "train"
        for row in phases.values()
    ):
        return None
    actions = {}
    comparison_count = 0
    joint_pass_count = 0
    for action in ("m075", "p075"):
        comparisons = [
            phase.get("same_phase_zero_comparisons", {}).get(action)
            for phase in phases.values()
        ]
        if not comparisons or any(not isinstance(row, dict) for row in comparisons):
            return None
        drag = [row.get("total_drag_reduction_fraction_positive_is_better") for row in comparisons]
        bias = [row.get("abs_mean_rear_cl_over_zero_fluctuation_rms") for row in comparisons]
        if any(not isinstance(value, (int, float)) for value in (*drag, *bias)):
            return None
        actions[action] = {
            "mean_drag_reduction": sum(drag) / len(drag),
            "mean_lift_bias_ratio": sum(bias) / len(bias),
        }
        comparison_count += len(comparisons)
        joint_pass_count += sum(
            row.get("canonical_joint_diagnostic_pass") is True
            for row in comparisons
        )
    return {
        "status": data["status"],
        "scope": "train_phase_open_loop_commissioning_only",
        "actions": actions,
        "comparison_count": comparison_count,
        "joint_pass_count": joint_pass_count,
    }


def _full40_extension_status(root: Path) -> dict:
    """Read the fixed scheduler/receipt JSONs; never inspect CFD or HDF data."""
    scheduler = _read_json(root / FULL40_SCHEDULER_STATE, None)
    snapshot = scheduler.get("snapshot", {}) if isinstance(scheduler, dict) else {}
    observed = snapshot.get("cases", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(observed, dict):
        observed = {}
    rows = []
    for case, (split, phase_bin, action_target) in FULL40_CASES.items():
        state = observed.get(case, {})
        if not isinstance(state, dict):
            state = {}
        receipt = _read_json(root / FULL40_RECEIPTS / f"{case}.json", None)
        checks = {
            "status": isinstance(receipt, dict)
            and receipt.get("status") == "FULL40_RAW_TRANSFER_VERIFIED",
            "case": isinstance(receipt, dict) and receipt.get("case") == case,
            "split": isinstance(receipt, dict) and receipt.get("split") == split,
            "phase_bin": isinstance(receipt, dict)
            and receipt.get("phase_bin") == phase_bin,
            "action_target": isinstance(receipt, dict)
            and receipt.get("action_target") == action_target,
            "predeclaration": isinstance(receipt, dict)
            and receipt.get("full40_predeclaration_sha256")
            == FULL40_PREDECLARATION_SHA256,
            "authorization": isinstance(receipt, dict)
            and receipt.get("full40_extension_authorization_sha256")
            == FULL40_AUTHORIZATION_SHA256,
            "raw_manifest": isinstance(receipt, dict)
            and isinstance(receipt.get("worker_raw_manifest_sha256"), str)
            and len(receipt["worker_raw_manifest_sha256"]) == 64
            and isinstance(receipt.get("raw_file_count"), int)
            and receipt["raw_file_count"] > 0,
        }
        verified = all(checks.values())
        status = state.get("status", "UNKNOWN")
        rows.append(
            {
                "case": case,
                "split": split,
                "phase_bin": phase_bin,
                "action_target": action_target,
                "scheduler_status": status,
                "alive": state.get("alive") is True,
                "solver_completed": status == "COMPLETED" or verified,
                "raw_qc_verified": verified,
                "receipt_checks": checks,
            }
        )
    return {
        "status": scheduler.get("status") if isinstance(scheduler, dict) else None,
        "scheduler_receipt_count": scheduler.get("receipt_count")
        if isinstance(scheduler, dict)
        else None,
        "cases": rows,
    }


def _small_file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _latest_artifact_dir(parent: Path, prefix: str) -> Path | None:
    try:
        candidates = [
            path for path in parent.glob(f"{prefix}*") if path.is_dir()
        ]
        return max(candidates, key=lambda path: path.stat().st_mtime, default=None)
    except OSError:
        return None


def _training_stage(root: Path, stage: str, expected_epochs: int) -> dict:
    prefix = f"{DEV30_QUICKSCREEN_PREFIX}{stage}_"
    run = _latest_artifact_dir(root / "artifacts", prefix)
    if run is None:
        return {"status": "NOT_STARTED", "epoch": 0, "expected_epochs": expected_epochs}
    history = _read_json(run / "training_history.json", [])
    if not isinstance(history, list):
        history = []
    rows = [row for row in history if isinstance(row, dict)]
    epoch = rows[-1].get("epoch", 0) if rows else 0
    return {
        "status": "COMPLETE" if epoch == expected_epochs else "RUNNING_OR_INCOMPLETE",
        "run_id": run.name.removeprefix(prefix),
        "epoch": epoch if isinstance(epoch, int) else 0,
        "expected_epochs": expected_epochs,
        "last_metrics": rows[-1] if rows else None,
    }


def _service_state(unit: str) -> str:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _fixed_training_stage(
    run: Path, expected_epochs: int, unit: str, *, worker_sync: bool = False
) -> dict:
    """Read one declared run only; never discover candidate or frozen artifacts."""
    history = _read_json(run / "training_history.json", [])
    if not isinstance(history, list):
        history = []
    rows = [row for row in history if isinstance(row, dict)]
    epoch = rows[-1].get("epoch", 0) if rows else 0
    if not isinstance(epoch, int):
        epoch = 0
    service_state = _service_state(unit)
    status = (
        "COMPLETE"
        if epoch == expected_epochs
        else "FAILED"
        if service_state == "failed"
        else "RUNNING"
        if service_state == "active"
        else "NOT_STARTED"
        if not run.exists()
        else "INCOMPLETE"
    )
    result = {
        "status": status,
        "service_state": service_state,
        "epoch": epoch,
        "expected_epochs": expected_epochs,
        "last_metrics": rows[-1] if rows else None,
        "frozen_hdf_opened_or_enumerated": False,
    }
    if worker_sync:
        result["synced_epochs"] = sum(
            (run / f"worker_epoch_{index:02d}_sync.json").is_file()
            for index in range(1, expected_epochs + 1)
        )
        result["final_sync_complete"] = (run / "worker_final_sync.json").is_file()
    return result


def _local_training_process_active() -> bool:
    try:
        lines = subprocess.run(
            ["ps", "-eo", "comm=,args="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return False
    return any(
        line.split(None, 1)[0] in {"python", "python3"}
        and "train_tandem_fno_rollout.py" in line
        for line in lines
        if line.split(None, 1)
    )


def _local_h20_posteval_active() -> bool:
    """Identify the fixed A-run validation evaluator, not an arbitrary GPU process."""
    try:
        lines = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return False
    return any(
        "evaluate_tandem_fno.py" in line
        and "tandem_fno_full40_free_ar_h20_ar20_20261003" in line
        and "--split validation" in line
        for line in lines
    )


def _h20_posteval_stage(run: Path, active: bool) -> dict:
    evaluation = _read_json(run / "validation10/evaluation.json", {})
    summary = evaluation.get("summary", {}) if isinstance(evaluation, dict) else {}
    complete = all(str(horizon) in summary for horizon in (1, 10, 50, 100))
    return {
        "status": "COMPLETE" if complete else "RETRY_RUNNING" if active else "PENDING",
        "complete": complete,
        "active": active,
        "initial_failure": "read-only visualization output path",
        "model_training_complete": True,
    }


def _last_jsonl_event(path: Path, event: str, limit: int = 262_144) -> dict | None:
    """Read only a bounded tail and return the newest matching JSON event."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("event") == event:
            return row
    return None


def _direct_cfd_ppo_status(root: Path) -> dict:
    """Read only the declared direct-CFD PPO run; FNO is not in its reward path."""
    run = root / "artifacts/direct_cfd/directppo2048_v1"
    latest_steps = [
        _last_jsonl_event(run / f"worker_env{index}.jsonl", "step")
        for index in (0, 1)
    ]
    live_collection_steps = sum(
        row.get("step", 0)
        for row in latest_steps
        if isinstance(row, dict) and isinstance(row.get("step"), int)
    )
    progress = _read_json(run / "training/progress.json", {})
    if not isinstance(progress, dict):
        progress = {}
    checkpoints = progress.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        checkpoints = []
    completed = progress.get("completed_transitions", 0)
    if not isinstance(completed, int):
        completed = 0
    training_result = _read_json(run / "training/result.json", {})
    if not isinstance(training_result, dict):
        training_result = {}
    training_complete = training_result.get("status") in {
        "DIRECT_REAL_CFD_PPO_TRAINING_COMPLETE",
        "DIRECT_REAL_CFD_PPO_CONTINUATION_COMPLETE",
    }
    running = (
        run.is_dir()
        and any(row is not None for row in latest_steps)
        and not training_complete
    )
    pair = root / "artifacts/direct_cfd/directppo2048_b00_eval80_v1"
    pair_progress = _read_json(pair / "rollout/progress.json", {})
    if not isinstance(pair_progress, dict):
        pair_progress = {}
    physical_result = _read_json(pair / "physical_result.json", {})
    if not isinstance(physical_result, dict):
        physical_result = {}
    independent = root / "artifacts/direct_cfd/directppo2048_b01_eval80_v1"
    independent_result = _read_json(independent / "physical_result.json", {})
    if not isinstance(independent_result, dict):
        independent_result = {}
    replay_audit = _read_json(
        root
        / "artifacts/direct_cfd/b00seq_b01_openloop_v1/audit_receipt.json",
        {},
    )
    if not isinstance(replay_audit, dict):
        replay_audit = {}
    pair_figure = pair / "raw_pair_timeseries.png"
    return {
        "status": progress.get("status", "DIRECT_REAL_CFD_PPO_RUNNING")
        if running
        else "IMPLEMENTATION_AND_TESTING",
        "running": running,
        "training_complete": training_complete,
        "completed_transitions": completed,
        "target_transitions": 2048,
        "live_collection_steps": live_collection_steps,
        "collection_steps_per_update": 256,
        "ppo_update_count": len(checkpoints),
        "last_checkpoint": checkpoints[-1] if checkpoints else None,
        "environment_latest_steps": latest_steps,
        "method": "official HydroGym + direct OpenFOAM feedback",
        "surrogate_dependency": False,
        "fno_used_for_reward": False,
        "physical_result_available": False,
        "final_physical_gate": "pending paired 80-D/U OpenFOAM evaluation",
        "paired_80d_evaluation": {
            "service_state": _service_state(
                "directppo2048-b00-eval80-v1.service"
            ),
            "completed_steps_per_branch": pair_progress.get(
                "completed_steps_per_branch", 0
            ),
            "physical_result": physical_result,
            "figure": {
                "available": pair_figure.is_file(),
                "path": "/direct-cfd-pair.png",
                "version": int(pair_figure.stat().st_mtime)
                if pair_figure.is_file()
                else 0,
            },
            "scope": "b00 train-phase preliminary paired physical validation",
        },
        "independent_b01_evaluation": {
            "physical_result": independent_result,
            "scope": (
                "b01 held-out-start-time validation; 18 D/U from b00 is about "
                "three shedding periods and statistical independence is unproven"
            ),
        },
        "b00_sequence_b01_replay": replay_audit,
        "frozen_test_accessed": False,
    }


def _free_ar_ablation(root: Path) -> dict:
    """Expose only the two predeclared free-AR runs and fixed Dynamic6 evidence."""
    h20 = root / "artifacts/tandem_fno_full40_free_ar_h20_ar20_20261003"
    h50 = root / "artifacts/tandem_fno_full40_free_ar_h50_ar50_20261003"
    parent_dynamic6 = _read_json(
        root
        / "artifacts/tandem_cylinders/full40_dynamic6_fno_e5_20261003/diagnostic.json",
        None,
    )
    h50_dynamic6 = _read_json(
        root
        / "artifacts/tandem_cylinders/full40_dynamic6_fno_h50_e1_20261003/diagnostic.json",
        None,
    )
    h50_stage = _fixed_training_stage(
        h50,
        8,
        "fluid-control-sync-worker-h50-20261003.service",
        worker_sync=True,
    )
    reallocation = _read_json(
        root / "artifacts/worker_audit/B_E5_REALLOCATION_RECEIPT.json", {}
    )
    if (
        isinstance(reallocation, dict)
        and reallocation.get("status") == "B_E5_FROZEN_FOR_DYNAMIC_H100_REALLOCATION"
    ):
        h50_stage["status"] = "INTENTIONALLY_REALLOCATED_AT_EPOCH5"
        h50_stage["reallocation"] = reallocation
    h50_validation = _read_json(
        root
        / "artifacts/tandem_fno_full40_free_ar_h50_epoch1_eval_20261003/validation10/diagnostic.json",
        None,
    )
    h50_stage["validation10"] = (
        h50_validation if isinstance(h50_validation, dict) else {}
    )
    dynamic_h100 = _fixed_training_stage(
        root
        / "artifacts/tandem_fno_dynamic_train8_h100_worker_h100_e5_2ep_r2_20261004",
        2,
        "fluid-control-sync-worker-dynamic-h100-r2-20261004.service",
        worker_sync=True,
    )
    control_train16_h100 = _fixed_training_stage(
        root / "artifacts/tandem_fno_control_train16_h100_20261004",
        2,
        "fluid-control-train16-h100-full-v1-20261004.service",
    )
    control_train16_h100["technical_probe_formal_candidate"] = False
    control_train16_h100["technical_probe_excluded_path"] = (
        "artifacts/tandem_fno_control_train16_h100_probe_20261004"
    )
    h20_stage = _fixed_training_stage(
        h20,
        8,
        "fluid-control-free-ar-h20-resume-20261003.service",
    )
    h20_stage["previous_failed_service_state"] = _service_state(
        "fluid-control-free-ar-h20-20261003.service"
    )
    h20_stage["local_training_process_active"] = _local_training_process_active()
    h20_stage["post_evaluation"] = _h20_posteval_stage(
        h20, _local_h20_posteval_active()
    )
    try:
        active_log = (
            h20 / "train.resume.log"
            if (h20 / "train.resume.log").is_file()
            else h20 / "train.log"
        )
        h20_stage["train_log_mtime_utc"] = datetime.fromtimestamp(
            active_log.stat().st_mtime, UTC
        ).isoformat(timespec="seconds")
        h20_stage["active_log"] = active_log.name
    except OSError:
        h20_stage["train_log_mtime_utc"] = None
        h20_stage["active_log"] = None
    return {
        "h20": h20_stage,
        "h50": h50_stage,
        "dynamic_h100": dynamic_h100,
        "control_train16_h100": control_train16_h100,
        "dynamic_h100_posteval": _read_json(
            root
            / "artifacts/tandem_fno_dynamic_train8_h100_worker_h100_e5_2ep_r2_20261004/canonical_posteval_receipt.json",
            {},
        ),
        "dynamic6_fno": h50_dynamic6
        if isinstance(h50_dynamic6, dict)
        else parent_dynamic6
        if isinstance(parent_dynamic6, dict)
        else {},
        "b5_dynamic6_start0": _read_json(
            root / "artifacts/fno_b5_dynamic6_start0_20261004/summary.json", {}
        ),
        "dynamic6_service_state": _service_state(
            "fluid-control-dynamic6-h50-e1-20261003.service"
        ),
        "direct_cfd_ppo": _direct_cfd_ppo_status(root),
        "frozen_hdf_opened_or_enumerated": False,
    }


def _tail_text(path: Path, limit: int = 262_144) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _dynamic6_runtime(root: Path) -> dict:
    cases = [
        "full40_dynamic_validation_b01_zero",
        "full40_dynamic_validation_b01_minus",
        "full40_dynamic_validation_b01_plus",
        "full40_dynamic_validation_b05_zero",
        "full40_dynamic_validation_b05_minus",
        "full40_dynamic_validation_b05_plus",
    ]
    case_root = root / "cfd/tandem_cylinders/cases"
    rows = []
    for name in cases:
        directory = case_root / name
        config = _read_json(directory / "case_config.json", {})
        start = config.get("start_time") if isinstance(config, dict) else None
        log = directory / "log.pimpleFoam.full40_dynamic_validation"
        marker = directory / "solver_complete.full40_dynamic_validation.json"
        times = re.findall(r"^Time = ([-+0-9.eE]+)\s*$", _tail_text(log), re.MULTILINE)
        latest = float(times[-1]) if times else None
        steps = (
            max(0, min(4000, round((latest - float(start)) / 0.005)))
            if latest is not None and isinstance(start, (int, float))
            else 0
        )
        rows.append(
            {
                "case": name,
                "phase_bin": config.get("phase_bin")
                if isinstance(config, dict)
                else None,
                "profile": config.get("profile")
                if isinstance(config, dict)
                else None,
                "steps": steps,
                "latest_time": latest,
                "complete": marker.is_file(),
                "log_exists": log.is_file(),
            }
        )
    completed = sum(row["complete"] for row in rows)
    current = next(
        (row for row in rows if row["log_exists"] and not row["complete"]), None
    )
    try:
        service = subprocess.run(
            [
                "systemctl",
                "--user",
                "is-active",
                "fluid-control-dynamic6-serial-r2-20261003.service",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        service_state = service.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        service_state = "unknown"
    try:
        processes = subprocess.run(
            ["ps", "-eo", "comm=,args="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.splitlines()
        ai_training_processes = sum(
            line.split(None, 1)[0] in {"python", "python3"}
            and "spark_gpu_guard.py" not in line
            and (
                "/train_tandem_fno.py" in line
                or "/train_tandem_fno_rollout.py" in line
            )
            for line in processes
            if line.split(None, 1)
        )
        cfd_solver_processes = sum(
            line.split(None, 1)[0] == "pimpleFoam"
            for line in processes
            if line.split(None, 1)
        )
    except (OSError, subprocess.SubprocessError):
        ai_training_processes = None
        cfd_solver_processes = None
    status = (
        "SOLVER_COMPLETE_PENDING_PANEL_QC"
        if completed == 6
        else "FAILED"
        if service_state == "failed"
        else "RUNNING"
        if service_state == "active"
        else "INCOMPLETE"
    )
    return {
        "status": status,
        "service_state": service_state,
        "completed_cases": completed,
        "case_count": 6,
        "current_case": current,
        "cases": rows,
        "ai_training_processes": ai_training_processes,
        "cfd_solver_processes": cfd_solver_processes,
        "frozen_hdf_opened_or_enumerated": False,
    }


def _full40_development_chain(root: Path, full40: dict) -> dict:
    """Read fixed metadata paths only; never enumerate or open frozen HDF data."""
    extension_rows = full40.get("cases", []) if isinstance(full40, dict) else []
    full40_verified = sum(
        row.get("raw_qc_verified") is True
        for row in extension_rows
        if isinstance(row, dict)
    )
    nine_qc = _read_json(root / MATCHED9_FINAL / "commissioning_qc.json", None)
    nine_pass = (
        isinstance(nine_qc, dict)
        and nine_qc.get("status") == "MATCHED_START_COMMISSIONING_NINE_CASE_QC_OK"
    )

    dev30_root = root / DEV30_RELEASE
    dev30_manifest_path = dev30_root / "manifest.json"
    dev30_manifest = _read_json(dev30_manifest_path, None)
    published = (
        isinstance(dev30_manifest, dict)
        and dev30_manifest.get("profile") == "matched_start_full40_v1"
        and dev30_manifest.get("release_kind")
        == "immutable_development_train20_validation10"
        and dev30_manifest.get("materialized_trajectory_counts")
        == {"train": 20, "validation": 10}
        and dev30_manifest.get("frozen_test_materialized") is False
    )

    train_names = list(MATCHED_START_CASES)
    train_names.extend(
        case for case, row in FULL40_CASES.items() if row[0] == "train"
    )
    validation_names = [
        case for case, row in FULL40_CASES.items() if row[0] == "validation"
    ]
    if published:
        train_ready = sum(
            (dev30_root / "train" / f"{case}.h5").is_file()
            for case in train_names
        )
        validation_ready = sum(
            (dev30_root / "validation" / f"{case}.h5").is_file()
            for case in validation_names
        )
    else:
        train_ready = sum(
            (root / MATCHED9_FINAL / "train" / f"{case}.h5").is_file()
            for case in MATCHED_START_CASES
        )
        train_ready += sum(
            (
                root
                / FULL40_STAGING
                / case
                / "train"
                / f"{case}.h5"
            ).is_file()
            for case, row in FULL40_CASES.items()
            if row[0] == "train"
        )
        validation_ready = sum(
            (
                root
                / FULL40_STAGING
                / case
                / "validation"
                / f"{case}.h5"
            ).is_file()
            for case in validation_names
        )

    diagnostic_dir = _latest_artifact_dir(
        root / "artifacts/tandem_cylinders", DEV30_DIAGNOSTIC_PREFIX
    )
    diagnostic = (
        _read_json(diagnostic_dir / "diagnostic.json", None)
        if diagnostic_dir is not None
        else None
    )
    if not isinstance(diagnostic, dict) or diagnostic.get("formal_gate") is not False:
        diagnostic = None

    ppo_run = _latest_artifact_dir(root / CANONICAL_PPO_ROOT, "run_")
    ppo_preflight = _read_json(root / CANONICAL_PPO_ROOT / "preflight.json", None)
    ppo_result = None
    if ppo_run is not None:
        for name in ("result.json", "summary.json", "training_summary.json"):
            ppo_result = _read_json(ppo_run / name, None)
            if isinstance(ppo_result, dict):
                break
    ppo_status = (
        ppo_result.get("status")
        if isinstance(ppo_result, dict)
        else ppo_preflight.get("status")
        if isinstance(ppo_preflight, dict)
        else "BLOCKED_NOT_STARTED"
    )
    return {
        "frozen_hdf_enumerated_or_opened": False,
        "raw_qc": {
            "full40_verified": full40_verified,
            "full40_target": 31,
            "commissioning_qc_pass": nine_pass,
        },
        "development_hdf": {
            "train_ready": train_ready,
            "train_target": 20,
            "validation_ready": validation_ready,
            "validation_target": 10,
        },
        "dev30_release": {
            "published": published,
            "reason": None if published else "IMMUTABLE_DEV30_NOT_PUBLISHED",
            "manifest_sha256": _small_file_sha256(dev30_manifest_path),
        },
        "quickscreen": {
            "onestep": _training_stage(root, "onestep", 10),
            "h20": _training_stage(root, "h20", 5),
        },
        "execution_incident": _read_json(
            root
            / "docs/results/full40_dev30_quickscreen_qs1_interrupt_recovery.json",
            None,
        ),
        "validation_diagnostic": diagnostic,
        "canonical_ppo": {
            "started": ppo_run is not None,
            "status": ppo_status,
        },
    }


def _parse_host(output: str, previous: tuple[int, int] | None):
    lines = output.strip().splitlines()
    cpu = [int(x) for x in lines[0].split()[1:]]
    total, idle = sum(cpu), cpu[3] + cpu[4]
    usage = None if previous is None or total == previous[0] else 100 * (1 - (idle - previous[1]) / (total - previous[0]))
    memory = {parts[0].rstrip(":"): int(parts[1]) for line in lines[1:3] if (parts := line.split())}
    gpu = [float(x.strip()) for x in lines[3].split(",")]
    task_start = lines.index("__TASKS__")
    epoch_start = lines.index("__EPOCH__")
    v3_start = lines.index("__V3_WORKER_EPOCH__")
    rollout_start = lines.index("__V3_ROLLOUT_EPOCH__")
    h20_start = lines.index("__V3_H20_EPOCH__")
    primary_h20_start = lines.index("__V3_PRIMARY_SEED_H20_EPOCH__")
    h20_rear_drag_start = lines.index("__V3_H20_REAR_DRAG_EPOCH__")
    v4_start = lines.index("__V4_EPOCH__")
    v4_single_validation_start = lines.index("__V4_SINGLE_VALIDATION__")
    active = []
    matched_start_curator_tasks = 0
    for line in lines[task_start + 1 : epoch_start]:
        fields = line.split(None, 1)
        if len(fields) != 2:
            continue
        command, args = fields
        if command == "pimpleFoam":
            active.append("OpenFOAM CFD")
        elif command == "foamToVTK":
            active.append("OpenFOAM 流场导出")
        elif command == "bash" and "build_control_gap_v4_spark.sh" in args:
            active.append("v4 训练数据 Curator 整理与校验")
        elif command == "bash" and "finalize_tandem_multistep_worker.sh" in args:
            active.append("模型训练完成后自动回传与验收")
        elif command in ("python", "python3") and "spark_gpu_guard.py" not in args:
            if "train_tandem_fno_paired_stats.py" in args:
                active.append("PhysicsNeMo FNO 动态配对训练" if "dynamic_paired" in args else "PhysicsNeMo FNO 配对监督训练")
            elif "train_tandem_fno_rollout.py" in args:
                active.append("PhysicsNeMo FNO 训练")
            elif "train_tandem_fno.py" in args:
                active.append("PhysicsNeMo FNO 新数据训练")
            elif "evaluate_tandem_fno.py" in args:
                active.append("PhysicsNeMo FNO 推理评估")
            elif "screen_tandem_cem_mpc.py" in args:
                active.append("CEM 控制筛选")
            elif "curate_tandem_cfd.py" in args:
                active.append("PhysicsNeMo Curator 数据整理")
            elif "curate_low_action_phase94_validation.py" in args and (
                "matched_start_commissioning_train9_v1" in args
            ):
                active.append("Matched-start Curator 数据整理")
                matched_start_curator_tasks += 1
    try:
        second_seed = json.loads(lines[epoch_start + 1])
        if not isinstance(second_seed, dict) or not isinstance(second_seed.get("epoch"), int):
            second_seed = {}
    except (IndexError, ValueError):
        second_seed = {}
    v3_epoch = int(lines[v3_start + 1]) if len(lines) > v3_start + 1 and lines[v3_start + 1].isdigit() else 0
    try:
        rollout = json.loads(lines[rollout_start + 1])
        rollout_epoch = int(rollout["epoch"])
    except (IndexError, ValueError, KeyError, TypeError):
        rollout_epoch = 0
    h20_epoch = int(lines[h20_start + 1]) if len(lines) > h20_start + 1 and lines[h20_start + 1].isdigit() else 0
    primary_h20_epoch = int(lines[primary_h20_start + 1]) if len(lines) > primary_h20_start + 1 and lines[primary_h20_start + 1].isdigit() else 0
    h20_rear_drag_epoch = int(lines[h20_rear_drag_start + 1]) if len(lines) > h20_rear_drag_start + 1 and lines[h20_rear_drag_start + 1].isdigit() else 0
    v4_epoch = int(lines[v4_start + 1]) if len(lines) > v4_start + 1 and lines[v4_start + 1].isdigit() else 0
    try:
        v4_single_rows = json.loads(lines[v4_single_validation_start + 1])
    except (IndexError, ValueError):
        v4_single_rows = None
    return {"time": datetime.now(UTC).isoformat(timespec="seconds"), "cpu": usage, "gpu": gpu[0], "temp_c": gpu[1], "power_w": gpu[2], "mem_available_gib": memory["MemAvailable"] / 1024**2, "mem_total_gib": memory["MemTotal"] / 1024**2, "tasks": sorted(set(active)), "task_count": len(active), "matched_start_curator_tasks": matched_start_curator_tasks, "second_seed_epoch": second_seed.get("epoch", 0), "second_seed_force_mae": second_seed.get("rollout_force_mae"), "v3_worker_epoch": v3_epoch, "v3_rollout_epoch": rollout_epoch, "v3_h20_epoch": h20_epoch, "v3_primary_seed_h20_epoch": primary_h20_epoch, "v3_h20_rear_drag_epoch": h20_rear_drag_epoch, "v4_epoch": v4_epoch, "v4_single_validation_pooled_nrmse": _pooled_terminal_nrmse(v4_single_rows), "matched_start": _matched_start_progress(lines)}, (total, idle)


def _cfd_progress(root: Path) -> list[dict]:
    rows = []
    for name in ("landscape_long_val_zero_20261003", "landscape_long_val_p100_20261003",
                 "landscape_long_val_m100_20261003", "landscape_step_val_p100_20261003",
                 "landscape_step_val_m100_20261003", "periodic_val_p10_20261003",
                 "periodic_val_p20_20261003", "train_signed_pulse_p_v4_20261003",
                 "train_signed_pulse_m_v4_20261003"):
        path = root / "cfd/tandem_cylinders/cases" / name / "log.pimpleFoam"
        if not path.exists():
            rows.append({"case": name, "status": "pending", "percent": 0.0})
            continue
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        times = re.findall(r"^Time = ([0-9.]+)$", tail, flags=re.MULTILINE)
        start = 82.0 if name.startswith("train_signed_pulse_") else 80.0
        percent = min(100.0, max(0.0, (float(times[-1]) - start) / 80.0 * 100)) if times else 0.0
        completed = bool(re.search(r"^End$", tail, flags=re.MULTILINE))
        rows.append({"case": name, "status": "complete" if completed else "running", "percent": 100.0 if completed else percent})
    return rows


def _curator_progress(root: Path) -> dict:
    path = root / "artifacts/tandem_cylinders/gate_b_aug_v3_curator.log"
    if not path.exists():
        return {"done": 0, "latest_frame": 0, "complete": False}
    log = path.read_text(encoding="utf-8", errors="replace")
    frames = {name: 0 for name in ("expanded_train_24", "expanded_train_25", "expanded_test_05")}
    for name, value in re.findall(r"^(expanded_(?:train_24|train_25|test_05)): sampled (\d+)/801 VTK frames", log, flags=re.MULTILINE):
        frames[name] = max(frames[name], int(value))
    pending = [value for value in frames.values() if 0 < value < 801]
    manifest = _read_json(root / "data/curated/tandem_cylinders_gate_b_aug_v3/manifest.json", {})
    split = _read_json(root / "artifacts/tandem_cylinders/gate_b_aug_v3_split_integrity.json", {})
    response = _read_json(root / "artifacts/tandem_cylinders/gate_b_aug_v3_control_response.json", {})
    complete = (manifest.get("trajectory_counts") == {"train": 26, "validation": 4, "test": 5}
                and split.get("status") == "SPLIT_INTEGRITY_OK"
                and response.get("status") == "CONTROLLED_CFD_RESPONSE_AUDIT_OK")
    return {"done": sum(value >= 801 for value in frames.values()), "latest_frame": min(pending) if pending else 0, "complete": complete}


def _v4_curator_progress(root: Path) -> dict:
    names = ("train_signed_pulse_p_v4_20261003", "train_signed_pulse_m_v4_20261003")
    data = root / "data/curated/tandem_cylinders_control_gap_v4"
    log_path = root / "artifacts/tandem_cylinders/control_gap_v4_build.log"
    frames = {name: 0 for name in names}
    if log_path.exists():
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 1048576))
            tail = stream.read().decode("utf-8", errors="replace")
        for name, count in re.findall(r"(train_signed_pulse_[pm]_v4_20261003): sampled (\d+)/801 VTK frames", tail):
            frames[name] = max(frames[name], int(count))
    finished = sum((data / "train" / f"{name}.h5").is_file() for name in names)
    manifest = _read_json(data / "manifest.json", {})
    split = _read_json(root / "artifacts/tandem_cylinders/control_gap_v4_split_integrity.json", {})
    return {"frames": sum(frames.values()), "new_hdf5": finished,
            "complete": manifest.get("trajectory_counts") == {"train": 28, "validation": 4, "test": 5}
                        and split.get("status") == "SPLIT_INTEGRITY_OK"}


class Sampler:
    def __init__(self, root: Path):
        self.path = root / SAMPLE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.samples = {"primary": deque(maxlen=360), "worker": deque(maxlen=360)}
        self.previous = {"primary": None, "worker": None}
        self.cuda_free_gib = None
        self.sample_count = 0
        self.lock = threading.Lock()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line in deque(stream, maxlen=720):
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    node = item.pop("node", None)
                    if node in self.samples:
                        self.samples[node].append(item)

    def run(self):
        while True:
            if self.sample_count % 6 == 0:
                self.cuda_free_gib = _primary_cuda_free_gib()
            for name in ("primary", "worker"):
                try:
                    sample, self.previous[name] = _parse_host(_host_output(name == "worker"), self.previous[name])
                except (OSError, subprocess.SubprocessError, ValueError, IndexError, KeyError) as exc:
                    sample = {"time": datetime.now(UTC).isoformat(timespec="seconds"), "error": str(exc)[:150]}
                if name == "primary":
                    sample["cuda_free_gib"] = self.cuda_free_gib
                with self.lock:
                    self.samples[name].append(sample)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"node": name, **sample}, ensure_ascii=False) + "\n")
            self.sample_count += 1
            time.sleep(10)


class Handler(BaseHTTPRequestHandler):
    root: Path
    sampler: Sampler

    def _send(self, body: bytes, mime: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if path == "/current-trial-field.png":
            registration = _read_json(self.root / "docs/LIVE_EXPERIMENT.json", {})
            if registration.get("progress_kind") != "exploratory_accelerated_long_h5_feedback":
                return self._send(b"not found", "text/plain", 404)
            try:
                versions = parse_qs(parsed.query, strict_parsing=True).get("v", [])
                if len(versions) != 1:
                    raise ValueError("one bound SHA is required")
                profile = _exploratory_mpc_profile(registration)
                document = _read_json(self.root / profile["base"] / "progress.json", {})
                rows = document.get("rows", [])
                payload = _current_trial_field_png(self.root, profile, rows, versions[0])
                if payload is None:
                    raise ValueError("current field evidence unavailable")
                return self._send(payload, "image/png")
            except (ValueError, KeyError, TypeError):
                return self._send(b"not found", "text/plain", 404)
        if path == "/final-ppo-field.png":
            try:
                versions = parse_qs(parsed.query, strict_parsing=True).get("v", [])
                if len(versions) != 1:
                    raise ValueError("one bound SHA is required")
                payload = _final_ppo_field_png(self.root, versions[0])
                if payload is None:
                    raise ValueError("final PPO field evidence unavailable")
                return self._send(payload, "image/png")
            except (ValueError, KeyError, TypeError):
                return self._send(b"not found", "text/plain", 404)
        if path == "/direct-cfd-pair.png":
            file = (
                self.root
                / "artifacts/direct_cfd/directppo2048_b00_eval80_v1/"
                "raw_pair_timeseries.png"
            )
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path.startswith("/figure/c-final/"):
            parts = path.removeprefix("/figure/c-final/").split("/")
            if (len(parts) != 2 or parts[0] not in ("zero", "minus", "plus")
                or parts[1] not in ("001.png", "010.png", "050.png", "100.png")
                or not _c_final_preview(self.root)["ready"]):
                return self._send(b"not found", "text/plain", 404)
            file = (self.root / C_FINAL_PREVIEW / f"figures/full40_dynamic_validation_b01_{parts[0]}"
                    / f"horizon_{parts[1].removesuffix('.png')}_start_0000.png")
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path.startswith("/figure/c-epoch1/"):
            name = path.removeprefix("/figure/c-epoch1/")
            if name not in ("001.png", "010.png", "050.png", "100.png") or not _c_epoch1_preview(self.root)["ready"]:
                return self._send(b"not found", "text/plain", 404)
            file = (self.root / C_EPOCH1_PREVIEW / "figures/full40_dynamic_validation_b01_plus"
                    / f"horizon_{name.removesuffix('.png')}_start_0000.png")
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path.startswith("/figure/paired/"):
            parts = path.removeprefix("/figure/paired/").split("/")
            if (len(parts) != 3 or parts[0] not in ("lambda0", "lambda10")
                or parts[1] not in ("zero", "minus", "plus")
                or parts[2] not in ("001.png", "010.png", "050.png", "100.png")):
                return self._send(b"not found", "text/plain", 404)
            file = (self.root / "artifacts" / f"paired_{parts[0]}_flow_visualization_20261004"
                    / "figures" / f"full40_dynamic_validation_b01_{parts[1]}"
                    / ("horizon_" + parts[2].removesuffix(".png") + "_start_0000.png"))
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path.startswith("/figure/h50-dynamic/"):
            parts = path.removeprefix("/figure/h50-dynamic/").split("/")
            if (
                len(parts) != 2
                or parts[0] not in ("zero", "minus", "plus")
                or parts[1] not in ("001.png", "010.png", "050.png", "100.png")
            ):
                return self._send(b"not found", "text/plain", 404)
            profile = f"full40_dynamic_validation_b01_{parts[0]}"
            filename = "horizon_" + parts[1].removesuffix(".png") + "_start_0000.png"
            file = self.root / H50_FIXED_FIGURES / profile / filename
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path.startswith("/figure/"):
            parts = path.removeprefix("/figure/").split("/")
            if len(parts) != 3 or parts[0] not in ("v3", "current", "previous") or parts[1] not in ("expanded_test_00", "expanded_test_01", "expanded_test_02", "expanded_test_04", "expanded_test_05") or parts[2] not in ("001.png", "010.png", "050.png", "100.png"):
                return self._send(b"not found", "text/plain", 404)
            base = (Path("artifacts/tandem_fno_gate_b_aug_v3_30epoch/heldout_figures") if parts[0] == "v3"
                    else RUN / "heldout_figures_dashboard" if parts[0] == "current"
                    else OLD / "heldout_figures")
            file = self.root / base / parts[1] / ("horizon_" + parts[2].removesuffix(".png") + "_start_0000.png")
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path == "/api/state":
            run = self.root / RUN
            audit = _read_json(run / "gate_b_audit.json", None)
            history = _read_json(run / "training_history.json", [])
            v3_training = _read_json(self.root / "artifacts/tandem_fno_gate_b_aug_v3_30epoch/training_history.json", [])
            heldout = _read_json(run / "heldout_evaluation.json", {})
            independent = _read_json(run / "heldout_evaluation_phase_independent.json", {})
            with self.sampler.lock:
                samples = {k: list(v) for k, v in self.sampler.samples.items()}
            figures = {}
            for case in ("expanded_test_00", "expanded_test_01", "expanded_test_02", "expanded_test_04", "expanded_test_05"):
                for horizon in ("001", "010", "050", "100"):
                    filename = f"{case}/horizon_{horizon}_start_0000.png"
                    v3 = self.root / "artifacts/tandem_fno_gate_b_aug_v3_30epoch/heldout_figures" / filename
                    current = run / "heldout_figures_dashboard" / filename
                    previous = self.root / OLD / "heldout_figures" / filename
                    if v3.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/v3/{case}/{horizon}.png", "version": int(v3.stat().st_mtime), "label": f"v3 单步 FNO · 历史 expanded_test 冻结测试 · H{int(horizon)} 步 · 真实 CFD / 预测 / 绝对误差；不是上方 v4 validation 评估"}
                    elif current.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/current/{case}/{horizon}.png", "version": int(current.stat().st_mtime), "label": f"Gate-B seed20261003 10轮多步 FNO · 历史 expanded_test 冻结测试 · H{int(horizon)} 步 · 真实 CFD / 预测 / 绝对误差；不是上方 v4 validation 评估"}
                    elif previous.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/previous/{case}/{horizon}.png", "version": int(previous.stat().st_mtime), "label": f"历史 30轮单步 FNO · expanded_test 冻结测试 · H{int(horizon)} 步 · 真实 CFD / 预测 / 绝对误差；不是上方 v4 validation 评估"}
            h50_figures = {}
            for profile in ("zero", "minus", "plus"):
                for horizon in ("001", "010", "050", "100"):
                    file = (
                        self.root
                        / H50_FIXED_FIGURES
                        / f"full40_dynamic_validation_b01_{profile}"
                        / f"horizon_{horizon}_start_0000.png"
                    )
                    if file.is_file():
                        h50_figures[f"{profile}/{horizon}"] = {
                            "path": f"/figure/h50-dynamic/{profile}/{horizon}.png",
                            "version": int(file.stat().st_mtime),
                        }
            data = {"server_time": datetime.now(UTC).isoformat(timespec="seconds"), "watchdog": _read_json(self.root / "artifacts/monitor/research_window_20261002/latest.json", None), "audit": audit, "v3_audit": _read_json(self.root / "artifacts/tandem_fno_gate_b_aug_v3_30epoch/gate_b_audit.json", None), "v3_worker_audit": _read_json(self.root / V3_WORKER_RUN / "gate_b_audit.json", None), "v3_primary_rollout_audit": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "gate_b_audit.json", None), "v3_rollout_audit": _read_json(self.root / V3_ROLLOUT_RUN / "gate_b_audit.json", None), "v3_h20_audit": _read_json(self.root / V3_H20_RUN / "gate_b_audit.json", None), "v3_observed": _read_json(self.root / "artifacts/tandem_fno_gate_b_aug_v3_30epoch/heldout_evaluation.json", None), "v3_worker_observed": _read_json(self.root / V3_WORKER_RUN / "heldout_evaluation.json", None), "v3_primary_rollout_observed": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "heldout_evaluation.json", None), "v3_rollout_observed": _read_json(self.root / V3_ROLLOUT_RUN / "heldout_evaluation.json", None), "v3_primary_rollout_validation": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "validation_long_horizon.json", None), "v3_rollout_validation": _read_json(self.root / V3_ROLLOUT_RUN / "validation_long_horizon.json", None), "v3_h20_observed": _read_json(self.root / V3_H20_RUN / "heldout_evaluation.json", None), "history": history, "v3_training": v3_training, "v3_primary_rollout_history": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "training_history.json", []), "second_seed": _read_json(self.root / SECOND_RUN / "heldout_evaluation.json", None), "evaluations": {"heldout": heldout.get("summary", {}), "independent": independent.get("summary", {})}, "resources": samples, "cfd": _cfd_progress(self.root), "curator": _curator_progress(self.root), "benchmark": _read_json(self.root / "artifacts/monitor/fno_inference_benchmark_seed20261003.json", None), "control_landscape_long": _read_json(self.root / "artifacts/tandem_cylinders/control_landscape_long_result_20261003.json", None), "periodic_benchmark": _read_json(self.root / "artifacts/tandem_cylinders/periodic_rotation_benchmark_result_20261003.json", None), "control_ranking": _read_json(self.root / "artifacts/tandem_cylinders/control_landscape_fno_ranking_h20_20261003.json", None), "figures": figures, "cem": (self.root / "artifacts/distributed_runs/gateb_multistep_20261002/formal/CEM_STAGE_C_COMPLETE").exists(), "ppo": False}
            data["v4_curator"] = _v4_curator_progress(self.root)
            data["matched_start_transfer"] = _matched_start_transfer_receipts(
                self.root
            )
            data["matched_start_pipeline"] = _matched_start_pipeline_status(
                self.root
            )
            data["matched_start_physics_summary"] = (
                _matched_start_physics_summary(self.root)
            )
            data["full40_extension"] = _full40_extension_status(self.root)
            data["full40_development_chain"] = _full40_development_chain(
                self.root, data["full40_extension"]
            )
            data["free_ar_ablation"] = _free_ar_ablation(self.root)
            data["research_overview"] = _research_overview(self.root)
            data["fc_p003c_epoch1_preview"] = _c_epoch1_preview(self.root)
            data["fc_p003c_final_preview"] = _c_final_preview(self.root)
            data["current_training_log"] = _current_training_log(self.root)
            data["fc_p003c_training_log"] = _current_training_log(self.root, "FC-P003C")
            data["training_evaluation_watchdog"] = _read_json(
                self.root / TRAINING_EVALUATION_WATCHDOG, None
            )
            train16_manifest = _read_json(
                self.root / DIRECTPPO_TRAIN16_MANIFEST, None
            )
            data["h50_dynamic_visualization"] = {
                "scientific_scope": (
                    "validation_only_roi_not_full_cfd_domain_"
                    "dynamic6_formal_fail_not_closed_loop_success"
                ),
                "figure_count": len(h50_figures),
                "figures": h50_figures,
                "train16_manifest": train16_manifest,
                "technical_probe_is_formal_candidate": False,
            }
            data["dynamic6_runtime"] = _dynamic6_runtime(self.root)
            data["dynamic6_physical_qc"] = _read_json(
                self.root
                / "artifacts/tandem_cylinders/full40_dynamic_validation_real_openfoam_qc_20261003.json",
                None,
            )
            data["full40_train20_physics"] = _read_json(
                self.root / FULL40_TRAIN20_PHYSICS, None
            )
            data["dual_node_watchdog"] = _dual_node_watchdog(self.root)
            data["gate_b_metric_integrity"] = _read_json(self.root / "artifacts/tandem_cylinders/gate_b_metric_integrity_v3_h20_rear_drag_test_v2_20261003.json", None)
            data["crossphase86_ranking"] = _read_json(self.root / "artifacts/tandem_cylinders/crossphase86_fno_cfd_ranking_20261003.json", None)
            data["control_ranking_weighted"] = _read_json(self.root / "artifacts/tandem_cylinders/control_landscape_fno_ranking_rear_weighted_h10_20261003.json", None)
            data["existing_open_loop"] = _read_json(self.root / "artifacts/tandem_cylinders/existing_open_loop_acceptance_audit_20261003.json", None)
            data["v3_rear_weighted_history"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "training_history.json", [])
            data["v3_rear_weighted_observed"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "heldout_evaluation.json", None)
            data["v3_rear_weighted_audit"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "gate_b_audit.json", None)
            data["v3_rear_weighted_validation_mid"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "validation_epoch5_long_horizon.json", None)
            data["v3_primary_seed_h20_observed"] = _read_json(self.root / V3_PRIMARY_SEED_H20_RUN / "heldout_evaluation.json", None)
            data["v3_primary_seed_h20_audit"] = _read_json(self.root / V3_PRIMARY_SEED_H20_RUN / "gate_b_audit.json", None)
            data["v3_h20_rear_drag_observed"] = _read_json(self.root / V3_H20_REAR_DRAG_RUN / "heldout_evaluation.json", None)
            data["v3_h20_rear_drag_audit"] = _read_json(self.root / V3_H20_REAR_DRAG_RUN / "gate_b_audit.json", None)
            data["v3_h20_rear_drag_validation_decision"] = _read_json(self.root / V3_H20_REAR_DRAG_RUN / "validation_decision.json", None)
            data["control_ranking_h20_rear_drag"] = _read_json(self.root / "artifacts/tandem_cylinders/control_landscape_fno_ranking_h20_rear_drag_20261003.json", None)
            data["v4_h20_history"] = _read_json(self.root / V4_H20_DEVELOPMENT_RUN / "training_history.json", [])
            data["phase_feedback_pilots"] = [_read_json(self.root / "artifacts/tandem_cylinders" / name / "result.json", None) for name in ("phase_feedback_pair_k075_20261003", "phase_feedback_pair_k020_20261003", "phase_feedback_pair_k050_l15_20261003")]
            data.update(_latest_evidence(self.root))
            data["fixed_feature_readout"] = _fixed_feature_readout(self.root)
            data["full_train_calibration"] = _full_train_calibration(self.root)
            data["free_ar_diagnostic"] = _free_ar_diagnostic(self.root)
            data["joint_readout_diagnostic"] = _joint_readout_diagnostic(self.root)
            data["decoder_scope_training"] = _fcp011_training(self.root)
            data["gradient_diagnostic"] = _fcp012_diagnostic(self.root)
            data["independent_force_training"] = _fcp013_training(self.root)
            data["window_accumulation_training"] = _fcp015_training(self.root)
            data["fixed_panel_probe"] = _fcp016_probe(self.root)
            data["reduced_rate_training"] = _fcp018_training(self.root)
            data["p018_formal_live"] = _fcp018_formal_live(self.root)
            data["p018_formal_result"] = _fcp018_formal_result(self.root)
            data["p020_live"] = _fcp020_live(self.root)
            data["p022_live"] = _fcp022_live(self.root)
            data["p023_live"] = _fcp023_live(self.root)
            data["p024_live"] = _fcp024_live(self.root)
            data["registered_experiment"] = _registered_experiment_live(self.root)
            data["exploratory_h5_ppo_training"] = _exploratory_h5_ppo_training(self.root)
            data["exploratory_diverse_h5_ppo_training"] = (
                _exploratory_diverse_h5_ppo_training(self.root))
            data["exploratory_diverse_h5_32768_ppo_training"] = (
                _exploratory_diverse_h5_32768_ppo_training(self.root))
            data["exploratory_diverse_ppo_real_cfd"] = (
                _exploratory_diverse_ppo_real_cfd(self.root))
            data["exploratory_diverse_32768_long_cfd"] = (
                _exploratory_diverse_32768_long_cfd(self.root))
            data["exploratory_final_ppo_real_cfd"] = _exploratory_final_ppo_real_cfd(self.root)
            data["p015_formal_result"] = _fcp015_formal_result(self.root)
            data["low_action_fno_h100"] = _low_action_fno_summary(self.root)
            return self._send(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8")
        return self._send(b"not found", "text/plain", 404)

    def log_message(self, fmt, *args):
        print("dashboard:", fmt % args, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    Handler.root = args.root.resolve()
    Handler.sampler = Sampler(Handler.root)
    threading.Thread(target=Handler.sampler.run, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
