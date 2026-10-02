#!/usr/bin/env python3
"""Read-only live dashboard for the tandem-cylinder research run.

Run on the primary Spark and access through an SSH local port forward. The
sampler records host CPU, GPU activity, and unified-memory headroom every 10 s.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


RUN = Path("artifacts/distributed_runs/gateb_multistep_20261002/formal/tandem_fno_total_drag_rollout_seed20261003")
OLD = Path("artifacts/tandem_fno_total_drag_spark_30epoch")
SAMPLE_FILE = Path("artifacts/monitor/live_resource_samples.jsonl")
HOST_COMMAND = (
    "awk '/^cpu /{print}' /proc/stat; "
    "awk '/^MemTotal:|^MemAvailable:/{print}' /proc/meminfo; "
    "nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw "
    "--format=csv,noheader,nounits"
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
.label{font-size:12px;color:#9aafc4}.small{font-size:12px;color:#a9bdd0}.good{color:#79d5a3}.bad{color:#f69d97}
svg{width:100%;height:180px;background:#101b2b;border:1px solid #25374b;border-radius:5px}.tall{height:235px}
.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:#bed0df;margin:6px 0}.sw{display:inline-block;width:11px;height:3px;vertical-align:middle;margin-right:5px}
select{background:#152237;color:#e5eff9;border:1px solid #42617f;padding:7px;border-radius:5px}
img{width:100%;height:auto;background:white;border-radius:4px}.row{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:13px}.summary .card{padding:12px}
.summary b{font-size:19px;display:block;margin:3px 0}.foot{margin-top:35px;border-top:1px solid #2a3d53;padding-top:13px;font-size:12px;color:#9aafc4}
@media(max-width:750px){.grid,.resources,.summary{grid-template-columns:1fr}main{padding:16px}}
</style></head><body><main>
<div class="top"><div><h1>串列双圆柱 · 实时科研监控</h1><div class="muted">OpenFOAM 数据 → PhysicsNeMo 代理 → HydroGym 控制</div></div><div class="stamp" id="clock">连接中…</div></div>
<div class="banner" id="decision">读取阶段判定…</div>
<div class="summary"><div class="card"><span class="label">当前阶段</span><b id="stage">—</b><span class="small">按预设门槛推进</span></div><div class="card"><span class="label">训练</span><b id="epoch">—</b><span class="small">正式多步 FNO</span></div><div class="card"><span class="label">测试总阻力 NRMSE</span><b id="heldout">—</b><span class="small">100 步，门槛 ≤10%</span></div><div class="card"><span class="label">独立相位 NRMSE</span><b id="independent">—</b><span class="small">100 步，门槛 ≤10%</span></div></div>
<h2>主从节点资源 · 每 10 秒采样</h2><div class="grid">
<div class="card"><h3>主节点 · USER · SPARK_HOST</h3><div class="resources"><div><div class="label">GPU 利用率</div><div class="number" id="primary-gpu">—</div></div><div><div class="label">CPU 利用率</div><div class="number" id="primary-cpu">—</div></div><div><div class="label">统一内存可用</div><div class="number" id="primary-mem">—</div></div></div><div class="small" id="primary-more"></div><svg id="primary-chart" role="img" aria-label="主节点 GPU 与 CPU 利用率历史"></svg></div>
<div class="card"><h3>计算节点 · nvidia · WORKER_HOST</h3><div class="resources"><div><div class="label">GPU 利用率</div><div class="number" id="worker-gpu">—</div></div><div><div class="label">CPU 利用率</div><div class="number" id="worker-cpu">—</div></div><div><div class="label">统一内存可用</div><div class="number" id="worker-mem">—</div></div></div><div class="small" id="worker-more"></div><svg id="worker-chart" role="img" aria-label="计算节点 GPU 与 CPU 利用率历史"></svg></div>
</div><div class="legend"><span><i class="sw" style="background:#60c9fb"></i>GPU</span><span><i class="sw" style="background:#e9ae68"></i>CPU</span><span>GB10 使用统一内存；nvidia-smi 不报告独立显存占用，安全约束看可用内存 ≥20 GiB。</span></div>
<h2>训练与评估</h2><div class="grid"><div class="card"><h3>10 轮多步训练：验证自由 rollout 误差</h3><svg class="tall" id="train-chart" role="img" aria-label="多步训练验证误差"></svg><div class="small">蓝：终点流场 MAE；橙：终点受力 MAE。真实训练历史，无插值。</div></div><div class="card"><h3>真实 CFD 保留工况：总阻力预测误差</h3><svg class="tall" id="error-chart" role="img" aria-label="不同预测步长的总阻力误差"></svg><div class="small">蓝：标准测试；绿：独立相位。纵轴 NRMSE；虚线为 10% 门槛。</div></div></div>
<h2>真实流场与预测</h2><div class="card"><div class="row"><div class="small" id="figure-label">读取图片…</div><div><select id="case"><option value="expanded_test_00">测试 00</option><option value="expanded_test_01">测试 01</option><option value="expanded_test_02">测试 02</option><option value="expanded_test_04">测试 04</option></select> <select id="horizon"><option value="001">1 步</option><option value="010">10 步</option><option value="050">50 步</option><option value="100" selected>100 步</option></select></div></div><img id="flow" alt="真实 OpenFOAM 流场、FNO 预测、误差对照"></div>
<h2>控制研究阶段</h2><div class="grid"><div class="card"><h3>CEM-MPC</h3><div id="cem">—</div></div><div class="card"><h3>HydroGym PPO</h3><div id="ppo">—</div></div></div>
<div class="foot">页面实时读取主 Spark 的原始实验文件；图像由模型评估脚本产生。资源样本保存在主节点 artifacts/monitor。图中预测准确度不代表已获得 CFD 闭环减阻。</div>
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
 $(name+'-more').textContent=`${last.time} · GPU ${num(last.temp_c,0)}°C / ${num(last.power_w,1)} W · 保留 ${num(last.mem_available_gib,1)} GiB`;
 plot(name+'-chart',[{values:items.map(x=>x.gpu),color:'#60c9fb'},{values:items.map(x=>x.cpu),color:'#e9ae68'}])}
function figure(){if(!latest)return;let key=$('case').value+'/'+$('horizon').value;let found=latest.figures[key];if(found){$('flow').src=found.path+'?v='+found.version;$('flow').hidden=false;$('figure-label').textContent=found.label}else{$('flow').hidden=true;$('figure-label').textContent='该工况暂无导出的对照图'}}
function render(d){latest=d;$('clock').textContent='服务器 '+d.server_time+' · 页面每 5 秒更新';let a=d.audit,c=a?.checks||{};
 let b=c.heldout_full_period_total_drag_nrmse?.total_drag_nrmse,i=c.independent_phase_full_period_total_drag_nrmse?.total_drag_nrmse;
 $('heldout').textContent=pct(b);$('independent').textContent=pct(i);$('heldout').className=Number.isFinite(b)&&b<=.1?'good':'bad';
 $('stage').textContent=a?.status==='GATE_B_PASS'?'阶段 C':'阶段 B';$('epoch').textContent=d.history.length+'/10 epochs';
 $('decision').textContent=a?.status==='GATE_B_PASS'?'阶段 B 已通过。正在核对阶段 C 的控制实验。':`阶段 B 未通过：标准测试 100 步总阻力 NRMSE ${pct(b)}，高于预设 10% 门槛。多步训练和独立评估已完成；需要改进代理模型后再开展 CEM 与当前目标 PPO。`;
 resources('primary',d.resources.primary);resources('worker',d.resources.worker);
 plot('train-chart',[{values:d.history.map(x=>x.terminal_state_mae),color:'#60c9fb'},{values:d.history.map(x=>x.terminal_force_mae),color:'#e9ae68'}],.05);
 let steps=['1','10','50','100'];plot('error-chart',[{values:steps.map(x=>d.evaluations.heldout?.[x]?.total_drag_nrmse),color:'#60c9fb'},{values:steps.map(x=>d.evaluations.independent?.[x]?.total_drag_nrmse),color:'#79d5a3'},{values:steps.map(()=>.1),color:'#d77979'}],.2);
 $('cem').textContent=d.cem?'CEM 结果已生成；需查看完整审计。':'本次 CEM 已被阶段 B 门槛阻止，未启动。';
 $('ppo').textContent=d.ppo?'当前目标 PPO 有新记录；需查看完整审计。':'当前总阻力目标的 PPO 尚未启动。旧目标的探索性 PPO 记录不计入当前结果。';figure()}
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


def _parse_host(output: str, previous: tuple[int, int] | None):
    lines = output.strip().splitlines()
    cpu = [int(x) for x in lines[0].split()[1:]]
    total, idle = sum(cpu), cpu[3] + cpu[4]
    usage = None if previous is None or total == previous[0] else 100 * (1 - (idle - previous[1]) / (total - previous[0]))
    memory = {parts[0].rstrip(":"): int(parts[1]) for line in lines[1:3] if (parts := line.split())}
    gpu = [float(x.strip()) for x in lines[3].split(",")]
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cpu": usage, "gpu": gpu[0], "temp_c": gpu[1], "power_w": gpu[2], "mem_available_gib": memory["MemAvailable"] / 1024**2, "mem_total_gib": memory["MemTotal"] / 1024**2}, (total, idle)


class Sampler:
    def __init__(self, root: Path):
        self.path = root / SAMPLE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.samples = {"primary": deque(maxlen=360), "worker": deque(maxlen=360)}
        self.previous = {"primary": None, "worker": None}
        self.lock = threading.Lock()

    def run(self):
        while True:
            for name in ("primary", "worker"):
                try:
                    sample, self.previous[name] = _parse_host(_host_output(name == "worker"), self.previous[name])
                except (OSError, subprocess.SubprocessError, ValueError, IndexError, KeyError) as exc:
                    sample = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": str(exc)[:150]}
                with self.lock:
                    self.samples[name].append(sample)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"node": name, **sample}, ensure_ascii=False) + "\n")
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
            if len(parts) != 3 or parts[0] not in ("current", "previous") or parts[1] not in ("expanded_test_00", "expanded_test_01", "expanded_test_02", "expanded_test_04") or parts[2] not in ("001.png", "010.png", "050.png", "100.png"):
                return self._send(b"not found", "text/plain", 404)
            base = RUN / "heldout_figures_dashboard" if parts[0] == "current" else OLD / "heldout_figures"
            file = self.root / base / parts[1] / ("horizon_" + parts[2].removesuffix(".png") + "_start_0000.png")
            try:
                return self._send(file.read_bytes(), "image/png")
            except OSError:
                return self._send(b"not found", "text/plain", 404)
        if path == "/api/state":
            run = self.root / RUN
            audit = _read_json(run / "gate_b_audit.json", None)
            history = _read_json(run / "training_history.json", [])
            heldout = _read_json(run / "heldout_evaluation.json", {})
            independent = _read_json(run / "heldout_evaluation_phase_independent.json", {})
            with self.sampler.lock:
                samples = {k: list(v) for k, v in self.sampler.samples.items()}
            figures = {}
            for case in ("expanded_test_00", "expanded_test_01", "expanded_test_02", "expanded_test_04"):
                for horizon in ("001", "010", "050", "100"):
                    filename = f"{case}/horizon_{horizon}_start_0000.png"
                    current = run / "heldout_figures_dashboard" / filename
                    previous = self.root / OLD / "heldout_figures" / filename
                    if current.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/current/{case}/{horizon}.png", "version": int(current.stat().st_mtime), "label": "当前 10 轮多步 FNO · 真实 CFD / 预测 / 绝对误差"}
                    elif previous.exists():
                        figures[f"{case}/{horizon}"] = {"path": f"/figure/previous/{case}/{horizon}.png", "version": int(previous.stat().st_mtime), "label": "上一版单步 FNO · 真实 CFD / 预测 / 绝对误差（当前模型图待生成）"}
            data = {"server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "audit": audit, "history": history, "evaluations": {"heldout": heldout.get("summary", {}), "independent": independent.get("summary", {})}, "resources": samples, "figures": figures, "cem": (self.root / "artifacts/distributed_runs/gateb_multistep_20261002/formal/CEM_STAGE_C_COMPLETE").exists(), "ppo": False}
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
