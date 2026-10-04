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
from urllib.parse import urlparse

C_EPOCH1_PREVIEW = Path("artifacts/fcp003c_epoch1_flow_visualization_preview_20261005")
C_EPOCH1_SHA = "fac949916859211b24553c410005aaff8ace057ce2bac5e08ac4dec971ecefba"
C_FINAL_PREVIEW = Path("artifacts/fcp003c_final_flow_visualization_20261005")
C_FINAL_SHA = "f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4"

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
<section id="flow-current"><h2>流场预测 · 真实 CFD / FNO / 误差</h2>
<div class="card"><div class="row"><h3>当前 C 模型 · 第一轮训练中预览</h3><select id="c-preview-step"><option value="001">1 步 / 0.1 D/U</option><option value="010">10 步 / 1 D/U</option><option value="050">50 步 / 5 D/U</option><option value="100" selected>100 步 / 10 D/U</option></select></div><p id="c-preview-status">等待预测图及数据校验完成。</p><img id="c-preview-image" alt="第一轮模型：真实 CFD、连续预测及绝对误差" style="width:100%" hidden><p class="small">仅一条 b01 动态转速验证轨迹，从 tU/D=130 的真实流场出发，之后连续预测；不是完整验证集的精度，也不是最终模型或闭环控制结果。左列：真实 CFD；中列：模型预测；右列：绝对误差。真实值与预测值共用每幅图的 1–99% 色标，误差使用独立色标。</p></div>
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
 image.closest('.card').querySelector('h3').textContent=finalReady?'当前 C 候选 · 第二轮模型预测图':'当前 C 模型 · 第一轮训练中预览';
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
function renderActiveExperiment(d){
 const w=d.training_evaluation_watchdog||{}, units=w.active_units||[];
 const age=Date.now()-Date.parse(w.timestamp_utc||'');
 if(!Number.isFinite(age)||age<0||age>180000){$('lead-now').textContent='任务状态已过期或时间异常，不能确认当前训练或评估是否运行。';return;}
 const training=units.includes('fluid-control-fcp003c-true-state-step-20261005.service');
 const evaluating=units.includes('fluid-control-fcp003c-posteval-wait-fa08ce0-20261005.service');
 if(!training&&!evaluating)return;
 const p=d.fc_p003c_training_log||{};
 const progress=Number.isFinite(p.batch_percent)?`最近批次记录为该轮的 ${p.batch_percent.toFixed(2)}%，记录时间 ${p.logged_at_utc}；不是整个项目完成比例。`:'当前尚无批次进度记录，不估算百分比。';
 const title=training?'当前训练 · 动作响应监督（FC-P003C）':'当前任务 · FC-P003C 评估队列';
 const detail=training?`已记录 ${p.completed_epochs??'待核实'} / 2 个完整训练轮次。${progress} 保持现有模型和数据，调整动作引起的受力差训练项；训练后检查流场、阻力和升力预测。`:'训练服务当前不在运行；评估队列正在运行或等待模型完成记录，不能仅凭队列存活认定 GPU 正在评估。';
 $('lead-now').textContent=title+'。'+detail+' 尚未完成新模型的 PPO 与真实 CFD 闭环验收。';
 const card=document.createElement('div');card.className='card';
 const heading=document.createElement('h3');heading.textContent=title;card.appendChild(heading);
 const note=document.createElement('p');note.textContent=detail+' 下方为已完成候选的结果，不是本轮训练精度。';card.appendChild(note);
 $('lead-models').prepend(card);
 $('train16-formal-progress').textContent=title;
 $('train16-formal-progress').className='number';
 $('train16-formal-detail').textContent=detail+' 本轮目录：tandem_fno_true_state_paired_step_lambda10_20261005。';
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
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
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
