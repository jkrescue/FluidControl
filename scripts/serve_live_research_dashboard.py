#!/usr/bin/env python3
"""Read-only live dashboard for the tandem-cylinder research run.

Run on the primary Spark and access through an SSH local port forward. The
sampler records host CPU, GPU activity, and unified-memory headroom every 10 s.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


RUN = Path("artifacts/distributed_runs/gateb_multistep_20261002/formal/tandem_fno_total_drag_rollout_seed20261003")
SECOND_RUN = Path("artifacts/distributed_runs/gateb_multistep_seed20261004_20261002/formal/tandem_fno_total_drag_rollout_seed20261004")
V3_WORKER_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_seed20261005_30epoch")
V3_ROLLOUT_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_rollout_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_seed20261005_10epoch")
V3_PRIMARY_ROLLOUT_RUN = Path("artifacts/tandem_fno_gate_b_aug_v3_rollout_seed20261002_10epoch")
V3_REAR_WEIGHTED_RUN = Path("artifacts/tandem_fno_gate_b_aug_v3_rear_drag_seed20261007_10epoch")
V3_H20_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261005_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261005_10epoch")
V3_PRIMARY_SEED_H20_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_rollout_h20_seed20261002_dense_20261002/formal/tandem_fno_gate_b_aug_v3_rollout_h20_seed20261002_dense_10epoch")
V3_H20_REAR_DRAG_RUN = Path("artifacts/distributed_runs/gateb_aug_v3_h20_rear_drag_seed20261002_20261003/formal/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch")
OLD = Path("artifacts/tandem_fno_total_drag_spark_30epoch")
SAMPLE_FILE = Path("artifacts/monitor/live_resource_samples.jsonl")
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
    "jq -r 'length' /tmp/fluid_control_gateb_20261002/artifacts/tandem_fno_gate_b_aug_v3_h20_rear_drag_seed20261002_10epoch/training_history.json 2>/dev/null || true"
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
@media(max-width:750px){.grid,.resources,.summary{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><main>
<div class="top"><div><h1>串列双圆柱流动控制 · 实时进展</h1><div class="muted">目标：降低两圆柱总阻力，同时报告侧向载荷与动作代价</div></div><div class="stamp" id="clock">连接中…</div></div>
<div class="banner" id="decision">读取阶段判定…</div>
<div class="small" id="watchdog">读取科研监控…</div>
<div class="summary"><div class="card"><span class="label">OpenFOAM CFD</span><b id="cfd-progress">—</b><span class="small" id="cfd-sub">新增 2 条训练、1 条独立测试轨迹</span></div><div class="card"><span class="label">PhysicsNeMo FNO</span><b id="epoch">—</b><span class="small" id="second-seed">计算节点：读取中</span></div><div class="card"><span class="label">100 步总阻力预测误差</span><b id="heldout">—</b><span class="small" id="heldout-scope">真实 CFD 测试；目标 ≤10%</span></div><div class="card"><span class="label">HydroGym / PPO</span><b id="hydro-status">—</b><span class="small">等待可信代理和 CEM 控制筛选</span></div></div>
<h2>真实 CFD 收益与代理决策是否一致</h2><div class="grid"><div class="card"><h3>配对长窗口 · t=120–160</h3><div class="number" id="physical-drag">读取中…</div><div class="small" id="physical-tradeoff">—</div><div class="small" id="periodic-detail">—</div></div><div class="card"><h3>已完成的 20 步 FNO 与 CFD 动作排序 · 100 步</h3><div class="number" id="ranking-score">读取中…</div><div class="small" id="ranking-detail">—</div></div></div>
<h2>两台 DGX Spark · 当前任务与算力</h2><div class="grid">
<div class="card"><h3>主节点 · SPARK_HOST</h3><div class="task" id="primary-task">读取中…</div><div class="resources"><div><div class="label">GPU 计算利用率</div><div class="number" id="primary-gpu">—</div></div><div><div class="label">CPU 利用率</div><div class="number" id="primary-cpu">—</div></div><div><div class="label">可用统一内存</div><div class="number" id="primary-mem">—</div></div></div><svg id="primary-chart" role="img" aria-label="主节点 GPU 与 CPU 利用率历史"></svg><div class="small" id="primary-more"></div></div>
<div class="card"><h3>计算节点 · WORKER_HOST</h3><div class="task" id="worker-task">读取中…</div><div class="resources"><div><div class="label">GPU 计算利用率</div><div class="number" id="worker-gpu">—</div></div><div><div class="label">CPU 利用率</div><div class="number" id="worker-cpu">—</div></div><div><div class="label">可用统一内存</div><div class="number" id="worker-mem">—</div></div></div><svg id="worker-chart" role="img" aria-label="计算节点 GPU 与 CPU 利用率历史"></svg><div class="small" id="worker-more"></div></div>
</div><div class="legend"><span><i class="sw" style="background:#60c9fb"></i>GPU</span><span><i class="sw" style="background:#e9ae68"></i>CPU</span><span>GB10 采用统一内存；训练保护线：至少剩余 20 GiB。</span></div>
<h2>FNO 训练和推理结果</h2><div class="grid"><div class="card"><h3>旧数据多步 FNO：训练轮次 → 10 步预测误差</h3><svg class="tall" id="train-chart" role="img" aria-label="多步训练验证误差"></svg><div class="small">蓝：流场平均绝对误差；橙：四个受力系数平均绝对误差。数值来自验证数据。</div></div><div class="card"><h3 id="error-title">递推步数 → 总阻力预测误差</h3><svg class="tall" id="error-chart" role="img" aria-label="不同预测步长的总阻力误差"></svg><div class="small" id="error-legend">读取评估结果…</div></div></div>
<div class="small" id="v3-metrics" style="margin-top:8px">正在读取新数据 FNO 训练指标…</div><div class="small" id="infer-speed" style="margin-top:4px">正在读取 FNO 推理耗时…</div>
<h2>真实流场 / FNO 预测 / 误差</h2><div class="card"><div class="row"><div class="small" id="figure-label">读取图片…</div><div><select id="case"><option value="expanded_test_00">测试 00</option><option value="expanded_test_01">测试 01</option><option value="expanded_test_02">测试 02</option><option value="expanded_test_04">测试 04</option><option value="expanded_test_05">新测试 05</option></select> <select id="horizon"><option value="001">1 步</option><option value="010">10 步</option><option value="050">50 步</option><option value="100" selected>100 步</option></select></div></div><img id="flow" alt="真实 OpenFOAM 流场、FNO 预测、误差对照"></div>
<h2>HydroGym 闭环控制</h2><div class="card"><div id="cem">—</div><div class="small" id="ppo">—</div></div>
<div class="foot">图表读取原始训练与评估记录。真实 CFD 控制收益仍须通过相位匹配的 OpenFOAM 回放验证。</div>
</main><script>
const $=x=>document.getElementById(x);let latest=null;
function pct(x){return Number.isFinite(x)?(x*100).toFixed(2)+'%':'—'}
function num(x,d=1){return Number.isFinite(x)?x.toFixed(d):'—'}
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
function figure(){if(!latest)return;let key=$('case').value+'/'+$('horizon').value;let found=latest.figures[key];if(found){$('flow').src=found.path+'?v='+found.version;$('flow').hidden=false;$('figure-label').textContent=found.label}else{$('flow').hidden=true;$('figure-label').textContent='该工况暂无导出的对照图'}}
function render(d){latest=d;$('clock').textContent='服务器 '+d.server_time+' · 页面每 5 秒更新';let candidates=[{label:'20 步＋后柱阻力加权训练',audit:d.v3_h20_rear_drag_audit,observed:d.v3_h20_rear_drag_observed},{label:'主节点种子 20 步训练',audit:d.v3_primary_seed_h20_audit,observed:d.v3_primary_seed_h20_observed},{label:'主节点后圆柱加权训练',audit:d.v3_rear_weighted_audit,observed:d.v3_rear_weighted_observed},{label:'计算节点种子 20 步训练',audit:d.v3_h20_audit,observed:d.v3_h20_observed},{label:'主节点 10 步训练',audit:d.v3_primary_rollout_audit,observed:d.v3_primary_rollout_observed},{label:'计算节点 10 步训练',audit:d.v3_rollout_audit,observed:d.v3_rollout_observed},{label:'主节点单步训练',audit:d.v3_audit,observed:d.v3_observed},{label:'计算节点单步训练',audit:d.v3_worker_audit,observed:d.v3_worker_observed}];let picked=candidates.find(x=>x.audit?.status==='GATE_B_PASS')||candidates.find(x=>x.observed)||candidates.find(x=>x.audit);let audited=picked?.audit,a=audited||d.audit,c=a?.checks||{};
 let long=d.control_landscape_long?.cases||[],zero=long.find(x=>x.omega_final===0),plus=long.find(x=>x.omega_final===1),minus=long.find(x=>x.omega_final===-1);
 $('physical-drag').textContent=plus&&minus?`+1: ${num(plus.relative_to_zero.total_cd_change_percent,2)}% · −1: ${num(minus.relative_to_zero.total_cd_change_percent,2)}%`:'等待真实 CFD 审计';
 $('physical-tradeoff').textContent=plus&&minus?`恒转两案的平均后柱升力均超出既定上限；总升力 RMS 分别为零控制的 ${num(plus.relative_to_zero.rear_cl_total_rms_ratio,2)}、${num(minus.relative_to_zero.rear_cl_total_rms_ratio,2)} 倍。`:'统计窗口 t=120–160；同时检查阻力与侧向载荷。';
 let periodic=d.periodic_benchmark?.decisions||[];$('periodic-detail').textContent=periodic.length===2?`零均值周期 P10/P20：总阻力变化 ${num(-100*periodic[0].total_cd_reduction,2)}% / ${num(-100*periodic[1].total_cd_reduction,2)}%；后柱升力脉动比 ${num(periodic[0].rear_cl_fluctuation_rms_ratio,2)} / ${num(periodic[1].rear_cl_fluctuation_rms_ratio,2)}；均未通过联合门槛。仅单相位开环 CFD。`:'零均值周期转速基线：等待真实 CFD 审计。';
 let rank=d.control_ranking;$('ranking-score').textContent=rank?`${Math.round(rank.pairwise_ranking_accuracy*6)}/6 对动作排序正确`:'等待动作排序审计';
 $('ranking-detail').textContent=rank?`FNO 选 +1，真实 CFD 在启动窗口选 −1；选错的阻力代价 ${num(rank.cfd_regret_of_fno_selection,4)}。当前模型不得直接宣称可控。`:'用相同初始流场检验真实 CFD 与 FNO 决策是否一致。';
 let observed=picked?.observed,preliminary=observed?.summary?.['100']?.total_drag_nrmse;
 let b=audited?c.heldout_full_period_total_drag_nrmse?.total_drag_nrmse:(Number.isFinite(preliminary)?preliminary:c.heldout_full_period_total_drag_nrmse?.total_drag_nrmse),i=c.independent_phase_full_period_total_drag_nrmse?.total_drag_nrmse;
 $('heldout').textContent=pct(b);$('heldout').className=audited&&Number.isFinite(b)&&b<=.1?'good':Number.isFinite(b)&&b>.1?'bad':'';
 $('epoch').textContent=`主节点：10 步 ${d.v3_primary_rollout_history.length}/10，加权 ${d.v3_rear_weighted_history.length}/10 轮`;$('hydro-status').textContent=d.cem?'CEM 已完成':'当前目标待 Gate B';
 let watch=d.watchdog||{}, hours=Number.isFinite(watch.seconds_remaining)?(watch.seconds_remaining/3600).toFixed(1):'—';$('watchdog').textContent=`持续科研监控剩余 ${hours} 小时 · 监控采样 ${watch.timestamp_utc||'待启动'} · 告警 ${watch.alerts?.length?watch.alerts.join('、'):'无'}`;$('watchdog').className='small '+(watch.alerts?.length?'bad':'good');
 let finished=d.cfd.filter(x=>x.status==='complete').length,average=d.cfd.reduce((s,x)=>s+x.percent,0)/Math.max(1,d.cfd.length);
 $('cfd-progress').textContent=`${finished}/${d.cfd.length} 配对 CFD 完成`;
 $('cfd-sub').textContent='同初始场物理对照 7 条；另有训练相位 t=82 的正负镜像脉冲 2 条';
 let worker=d.resources.worker.at(-1)||{};$('second-seed').textContent=`计算节点：20 步＋后柱阻力加权 ${worker.v3_h20_rear_drag_epoch||0}/10 轮；主种子 20 步 ${worker.v3_primary_seed_h20_epoch||0}/10 轮已验收`;
 $('heldout-scope').textContent=audited?`${picked.label}；v3 五条 CFD 测试，完整审计` :Number.isFinite(preliminary)?`${picked.label}；v3 五条 CFD 测试初评，完整审计中`:'旧数据四条 CFD 测试；目标 ≤10%';
 $('decision').textContent=audited&&a.status==='GATE_B_PASS'?`${picked.label}已通过冻结的 100 步总阻力门槛（${pct(b)}）；下一步做 CEM 控制筛选，再用真实 CFD 验证。`:audited?`${picked.label}五工况 100 步总阻力误差 ${pct(b)}，未达到 10%；其他候选仍在训练或审计。CEM 与 PPO 暂不启动。`:Number.isFinite(preliminary)?`${picked.label}的 100 步五工况初评为 ${pct(b)}；动作扰动与独立相位审计未完成。CEM/PPO 暂不启动。`:`旧数据模型 100 步误差 ${pct(b)}，未达到 10%；v3 FNO 正在完成独立测试。CEM 与 PPO 暂不启动。`;
 resources('primary',d.resources.primary);resources('worker',d.resources.worker);
 plot('train-chart',[{values:d.history.map(x=>x.terminal_state_mae),color:'#60c9fb'},{values:d.history.map(x=>x.terminal_force_mae),color:'#e9ae68'}],.05);
 let steps=['1','10','50','100'],v3series=[{result:d.v3_observed,color:'#60c9fb',label:'主节点单步'},{result:d.v3_worker_observed,color:'#79d5a3',label:'计算节点单步'},{result:d.v3_primary_rollout_observed,color:'#dc95e4',label:'主节点 10 步训练'},{result:d.v3_rollout_observed,color:'#e9ae68',label:'计算节点 10 步训练'},{result:d.v3_h20_observed,color:'#f49ab8',label:'计算节点种子 20 步训练'},{result:d.v3_rear_weighted_observed,color:'#97e1e4',label:'主节点后圆柱加权训练'},{result:d.v3_primary_seed_h20_observed,color:'#dce779',label:'主节点种子 20 步训练'}].filter(x=>x.result?.summary);
 if(v3series.length){let series=v3series.map(x=>({values:steps.map(h=>x.result.summary[h]?.total_drag_nrmse),color:x.color}));let top=Math.max(.2,...series.flatMap(x=>x.values.filter(Number.isFinite)));series.push({values:steps.map(()=>.1),color:'#d77979'});plot('error-chart',series,Math.ceil(top*10)/10);$('error-title').textContent='v3 真实 CFD：递推步数 → 总阻力误差';$('error-legend').textContent=v3series.map(x=>x.label).join('、')+'；红线：10% 门槛。模型结果需以完整审计为准。'}
 else{plot('error-chart',[{values:steps.map(x=>d.evaluations.heldout?.[x]?.total_drag_nrmse),color:'#60c9fb'},{values:steps.map(x=>d.evaluations.independent?.[x]?.total_drag_nrmse),color:'#79d5a3'},{values:steps.map(()=>.1),color:'#d77979'}],.2);$('error-title').textContent='旧数据多步 FNO：递推步数 → 总阻力误差';$('error-legend').textContent='蓝：旧数据 4 条 CFD 测试；绿：独立初始相位；红：10% 门槛。'}
 let current=d.v3_rear_weighted_history.at(-1)||d.v3_primary_rollout_history.at(-1),primaryValidation100=d.v3_primary_rollout_validation?.summary?.['100']?.total_drag_nrmse,workerValidation100=d.v3_rollout_validation?.summary?.['100']?.total_drag_nrmse,weightedValidation100=d.v3_rear_weighted_validation_mid?.summary?.['100']?.total_drag_nrmse;$('v3-metrics').textContent=current?`主节点当前 FNO 验证滚动流场误差 ${num(current.rollout_state_mae,4)}；验证滚动受力误差 ${num(current.rollout_force_mae,4)}。${Number.isFinite(primaryValidation100)?`原 10 步模型验证集 100 步总阻力误差 ${pct(primaryValidation100)}。`:''}${Number.isFinite(weightedValidation100)?`后圆柱加权第 5 轮对应误差 ${pct(weightedValidation100)}，仍需完成 10 轮。`:Number.isFinite(workerValidation100)?`计算节点 10 步模型对应误差 ${pct(workerValidation100)}。`:''}`:'新数据 FNO：多步训练首轮指标尚未产生。';
 $('infer-speed').textContent=d.benchmark?.status==='FNO_REAL_CFD_INFERENCE_BENCHMARK_OK'?`旧数据多步 FNO、真实 CFD 输入：单步中位 ${num(d.benchmark.step_median_ms,2)} ms；连续 100 步 ${num(d.benchmark.rollout_100_step_seconds,2)} s。仅模型前向，不含 CFD 或控制通信。`:'FNO 推理耗时尚未测量。';
 $('cem').textContent=d.cem?'CEM 控制筛选已完成，结果待审计。':`CEM：等待 FNO 的 100 步总阻力误差降至 10% 以下。新增 CFD 平均求解进度 ${num(average,0)}%。`;
 $('ppo').textContent=d.ppo?'HydroGym PPO 有当前目标的新记录。':'当前总阻力目标的 HydroGym PPO 尚未启动。历史末柱目标的 PPO 曾完成 32 步真实 CFD 闭环，但目标差 +0.003855（更差），不能视为当前控制收益。';figure()}
async function refresh(){try{let r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json())}catch(e){$('clock').textContent='连接失败：'+e.message}}
$('case').onchange=figure;$('horizon').onchange=figure;window.onresize=()=>{if(latest)render(latest)};refresh();setInterval(refresh,5000);
</script></body></html>'''


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


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
    active = []
    for line in lines[task_start + 1 : epoch_start]:
        fields = line.split(None, 1)
        if len(fields) != 2:
            continue
        command, args = fields
        if command == "pimpleFoam":
            active.append("OpenFOAM CFD")
        elif command == "bash" and "finalize_tandem_multistep_worker.sh" in args:
            active.append("模型训练完成后自动回传与验收")
        elif command in ("python", "python3") and "spark_gpu_guard.py" not in args:
            if "train_tandem_fno_rollout.py" in args:
                active.append("PhysicsNeMo FNO 训练")
            elif "train_tandem_fno.py" in args:
                active.append("PhysicsNeMo FNO 新数据训练")
            elif "evaluate_tandem_fno.py" in args:
                active.append("PhysicsNeMo FNO 推理评估")
            elif "screen_tandem_cem_mpc.py" in args:
                active.append("CEM 控制筛选")
            elif "curate_tandem_cfd.py" in args:
                active.append("PhysicsNeMo Curator 数据整理")
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
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cpu": usage, "gpu": gpu[0], "temp_c": gpu[1], "power_w": gpu[2], "mem_available_gib": memory["MemAvailable"] / 1024**2, "mem_total_gib": memory["MemTotal"] / 1024**2, "tasks": sorted(set(active)), "task_count": len(active), "second_seed_epoch": second_seed.get("epoch", 0), "second_seed_force_mae": second_seed.get("rollout_force_mae"), "v3_worker_epoch": v3_epoch, "v3_rollout_epoch": rollout_epoch, "v3_h20_epoch": h20_epoch, "v3_primary_seed_h20_epoch": primary_h20_epoch, "v3_h20_rear_drag_epoch": h20_rear_drag_epoch}, (total, idle)


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
                    sample = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": str(exc)[:150]}
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

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
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
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/v3/{case}/{horizon}.png", "version": int(v3.stat().st_mtime), "label": "v3 单步 FNO · 真实 OpenFOAM CFD / 预测 / 绝对误差（初评）"}
                    elif current.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/current/{case}/{horizon}.png", "version": int(current.stat().st_mtime), "label": "当前 10 轮多步 FNO · 真实 CFD / 预测 / 绝对误差"}
                    elif previous.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/previous/{case}/{horizon}.png", "version": int(previous.stat().st_mtime), "label": "上一版单步 FNO · 真实 CFD / 预测 / 绝对误差（当前模型图待生成）"}
            data = {"server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "watchdog": _read_json(self.root / "artifacts/monitor/research_window_20261002/latest.json", None), "audit": audit, "v3_audit": _read_json(self.root / "artifacts/tandem_fno_gate_b_aug_v3_30epoch/gate_b_audit.json", None), "v3_worker_audit": _read_json(self.root / V3_WORKER_RUN / "gate_b_audit.json", None), "v3_primary_rollout_audit": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "gate_b_audit.json", None), "v3_rollout_audit": _read_json(self.root / V3_ROLLOUT_RUN / "gate_b_audit.json", None), "v3_h20_audit": _read_json(self.root / V3_H20_RUN / "gate_b_audit.json", None), "v3_observed": _read_json(self.root / "artifacts/tandem_fno_gate_b_aug_v3_30epoch/heldout_evaluation.json", None), "v3_worker_observed": _read_json(self.root / V3_WORKER_RUN / "heldout_evaluation.json", None), "v3_primary_rollout_observed": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "heldout_evaluation.json", None), "v3_rollout_observed": _read_json(self.root / V3_ROLLOUT_RUN / "heldout_evaluation.json", None), "v3_primary_rollout_validation": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "validation_long_horizon.json", None), "v3_rollout_validation": _read_json(self.root / V3_ROLLOUT_RUN / "validation_long_horizon.json", None), "v3_h20_observed": _read_json(self.root / V3_H20_RUN / "heldout_evaluation.json", None), "history": history, "v3_training": v3_training, "v3_primary_rollout_history": _read_json(self.root / V3_PRIMARY_ROLLOUT_RUN / "training_history.json", []), "second_seed": _read_json(self.root / SECOND_RUN / "heldout_evaluation.json", None), "evaluations": {"heldout": heldout.get("summary", {}), "independent": independent.get("summary", {})}, "resources": samples, "cfd": _cfd_progress(self.root), "curator": _curator_progress(self.root), "benchmark": _read_json(self.root / "artifacts/monitor/fno_inference_benchmark_seed20261003.json", None), "control_landscape_long": _read_json(self.root / "artifacts/tandem_cylinders/control_landscape_long_result_20261003.json", None), "periodic_benchmark": _read_json(self.root / "artifacts/tandem_cylinders/periodic_rotation_benchmark_result_20261003.json", None), "control_ranking": _read_json(self.root / "artifacts/tandem_cylinders/control_landscape_fno_ranking_h20_20261003.json", None), "figures": figures, "cem": (self.root / "artifacts/distributed_runs/gateb_multistep_20261002/formal/CEM_STAGE_C_COMPLETE").exists(), "ppo": False}
            data["v3_rear_weighted_history"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "training_history.json", [])
            data["v3_rear_weighted_observed"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "heldout_evaluation.json", None)
            data["v3_rear_weighted_audit"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "gate_b_audit.json", None)
            data["v3_rear_weighted_validation_mid"] = _read_json(self.root / V3_REAR_WEIGHTED_RUN / "validation_epoch5_long_horizon.json", None)
            data["v3_primary_seed_h20_observed"] = _read_json(self.root / V3_PRIMARY_SEED_H20_RUN / "heldout_evaluation.json", None)
            data["v3_primary_seed_h20_audit"] = _read_json(self.root / V3_PRIMARY_SEED_H20_RUN / "gate_b_audit.json", None)
            data["v3_h20_rear_drag_observed"] = _read_json(self.root / V3_H20_REAR_DRAG_RUN / "heldout_evaluation.json", None)
            data["v3_h20_rear_drag_audit"] = _read_json(self.root / V3_H20_REAR_DRAG_RUN / "gate_b_audit.json", None)
            data["v3_h20_rear_drag_validation_decision"] = _read_json(self.root / V3_H20_REAR_DRAG_RUN / "validation_decision.json", None)
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
