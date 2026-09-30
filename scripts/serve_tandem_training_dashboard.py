#!/usr/bin/env python3
"""Serve a live browser dashboard for tandem-cylinder training histories."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PhysicsNeMo 训练监控</title>
<style>
:root { color-scheme: light dark; --bg:#0b1020; --panel:#121a2e; --fg:#e8edf7;
  --muted:#9aa8bf; --grid:#2d3952; --line:#5ec8ff; --best:#65d995; --warn:#ffbe55; }
@media (prefers-color-scheme:light) { :root { --bg:#f4f7fb; --panel:#fff; --fg:#172033;
  --muted:#62708a; --grid:#d9e0eb; --line:#087bb7; --best:#168653; --warn:#a76100; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1280px; margin:auto; }
header { display:flex; gap:18px; align-items:end; justify-content:space-between; flex-wrap:wrap; }
h1 { margin:0; font-size:24px; font-weight:600; }
.sub { color:var(--muted); margin-top:4px; }
select { padding:7px 10px; border:1px solid var(--grid); border-radius:6px;
  background:var(--panel); color:var(--fg); }
.stats { display:grid; grid-template-columns:repeat(4,minmax(140px,1fr)); gap:12px; margin:18px 0; }
.stat { background:var(--panel); padding:12px 14px; border-radius:8px; }
.stat span { display:block; color:var(--muted); font-size:12px; }
.stat strong { display:block; margin-top:3px; font-size:21px; font-weight:600; font-variant-numeric:tabular-nums; }
.plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.plot { background:var(--panel); padding:12px; border-radius:8px; min-width:0; }
.plot h2 { margin:0 0 7px; font-size:14px; font-weight:600; }
.note { margin:14px 0; padding:11px 14px; border-left:4px solid var(--warn);
  border-radius:6px; background:var(--panel); color:var(--muted); }
canvas { display:block; width:100%; height:280px; }
#status { margin-top:14px; color:var(--muted); font-variant-numeric:tabular-nums; }
@media(max-width:760px) { body{padding:14px}.stats{grid-template-columns:repeat(2,1fr)}.plots{grid-template-columns:1fr} }
</style>
</head>
<body><main>
<header><div><h1>PhysicsNeMo 训练监控</h1><div class="sub">每 3 秒自动更新</div></div>
<label>运行阶段 <select id="run"><option value="one_step">单步 FNO</option><option value="rollout">Rollout 微调</option><option value="no_tf">无 Teacher Forcing 对照</option></select></label></header>
<section class="stats" aria-live="polite">
 <div class="stat"><span>当前 Epoch</span><strong id="epoch">—</strong></div>
 <div class="stat"><span>训练 Loss</span><strong id="loss">—</strong></div>
 <div class="stat"><span id="best-label">最佳验证 MAE</span><strong id="best">—</strong></div>
 <div class="stat"><span>记录更新时间</span><strong id="updated" style="font-size:15px">—</strong></div>
</section>
<div class="note" id="explanation"></div>
<section class="plots">
 <div class="plot"><h2 id="title-0"></h2><canvas id="chart-0" role="img"></canvas></div>
 <div class="plot"><h2 id="title-1"></h2><canvas id="chart-1" role="img"></canvas></div>
 <div class="plot"><h2 id="title-2"></h2><canvas id="chart-2" role="img"></canvas></div>
 <div class="plot"><h2 id="title-3"></h2><canvas id="chart-3" role="img"></canvas></div>
 <div class="plot"><h2 id="title-4"></h2><canvas id="chart-4" role="img"></canvas></div>
 <div class="plot"><h2 id="title-5"></h2><canvas id="chart-5" role="img"></canvas></div>
</section>
<div id="status" aria-live="polite">等待训练记录……</div>
</main>
<script>
const runSelect=document.getElementById('run');
const specs={
 one_step:[['train_loss','训练 Loss',true],['state_mae_physical_units','验证场 MAE',false],['state_rmse_physical_units','验证场 RMSE',false],['force_mae_normalized','验证力系数 MAE（标准化）',false]],
 rollout:[['train_loss','课程训练 Loss（Epoch 1–15 难度递增）',true],['teacher_forcing_ratio','Teacher Forcing 比例',false],['selection_score','固定自由 Rollout 验证指标',false],['rollout_state_mae','Rollout 场 MAE',false],['terminal_state_mae','末步场 MAE',false],['terminal_force_mae','末步力系数 MAE',false]],
 no_tf:[['train_loss','固定纯 Rollout 训练 Loss',true],['teacher_forcing_ratio','Teacher Forcing 比例',false],['selection_score','固定自由 Rollout 验证指标',false],['rollout_state_mae','Rollout 场 MAE',false],['terminal_state_mae','末步场 MAE',false],['terminal_force_mae','末步力系数 MAE',false]]
};
const requestedRun=new URLSearchParams(window.location.search).get('run');
if(Object.prototype.hasOwnProperty.call(specs,requestedRun))runSelect.value=requestedRun;
function css(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim()}
function fmt(v){if(v===undefined||v===null||!Number.isFinite(+v))return '—';v=+v;return v===0?'0':(Math.abs(v)<.001||Math.abs(v)>=1000?v.toExponential(3):v.toFixed(6))}
function draw(canvas, rows, key, label, logScale){
 const box=canvas.getBoundingClientRect(), dpr=devicePixelRatio||1, w=Math.max(320,box.width), h=280;
 canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,w,h);
 const data=rows.map(r=>({x:+r.epoch,y:+r[key]})).filter(d=>Number.isFinite(d.x)&&Number.isFinite(d.y)&&(!logScale||d.y>0));
 canvas.setAttribute('aria-label',label+'，共 '+data.length+' 个 epoch');
 if(!data.length){c.fillStyle=css('--muted');c.fillText('尚无数据',20,35);return}
 const L=68,R=18,T=14,B=42, xs=data.map(d=>d.x), raw=data.map(d=>logScale?Math.log10(d.y):d.y);
 let xmin=Math.min(...xs),xmax=Math.max(...xs);if(xmax===xmin)xmax=xmin+1;
 let ymin=Math.min(...raw),ymax=Math.max(...raw);let pad=Math.max((ymax-ymin)*.12,Math.abs(ymax)*.03,1e-12);ymin-=pad;ymax+=pad;
 const X=x=>L+(x-xmin)/(xmax-xmin)*(w-L-R),Y=y=>T+(ymax-y)/(ymax-ymin)*(h-T-B);
 c.strokeStyle=css('--grid');c.lineWidth=1;c.fillStyle=css('--muted');c.font='12px system-ui';
 for(let i=0;i<=4;i++){const yy=T+(h-T-B)*i/4;c.beginPath();c.moveTo(L,yy);c.lineTo(w-R,yy);c.stroke();const val=ymax-(ymax-ymin)*i/4;c.textAlign='right';c.fillText(logScale?'10^'+val.toFixed(1):fmt(val),L-8,yy+4)}
 for(let i=0;i<=4;i++){const val=xmin+(xmax-xmin)*i/4,xx=X(val);c.textAlign='center';c.fillText(Math.round(val),xx,h-17)}
 c.fillStyle=css('--muted');c.textAlign='center';c.fillText('Epoch',L+(w-L-R)/2,h-2);
 c.strokeStyle=css('--line');c.lineWidth=2;c.beginPath();data.forEach((d,i)=>{const y=logScale?Math.log10(d.y):d.y;i?c.lineTo(X(d.x),Y(y)):c.moveTo(X(d.x),Y(y))});c.stroke();
 const last=data[data.length-1],ly=logScale?Math.log10(last.y):last.y;c.fillStyle=css('--line');c.beginPath();c.arc(X(last.x),Y(ly),4,0,Math.PI*2);c.fill();
 c.fillStyle=css('--fg');c.textAlign='right';c.fillText(fmt(last.y),w-R,T+12);
}
async function refresh(){
 const run=runSelect.value;
 try{
  const res=await fetch('/api/history?run='+encodeURIComponent(run),{cache:'no-store'});if(!res.ok)throw new Error('HTTP '+res.status);
  const payload=await res.json(),rows=payload.history||[],cfg=specs[run];
  document.querySelectorAll('.plot').forEach((plot,index)=>plot.style.display=index<cfg.length?'block':'none');
  document.getElementById('explanation').textContent=run==='rollout'
   ? 'Epoch 1–15 的 Teacher Forcing 从 0.5 降至 0，训练输入逐步减少真实状态提示，因此训练 Loss 的难度在变化。Epoch 16–30 为固定的纯自由 Rollout，可直接比较；选择指标与验证 MAE 始终使用自由 Rollout。'
   : run==='no_tf'
   ? 'Teacher Forcing 全程为 0，所有 Epoch 的训练 Loss 对应相同的纯自由 Rollout 目标，可以直接横向比较。'
   : '单步训练的目标保持固定，可直接比较各 Epoch 的训练 Loss 与验证误差。';
  cfg.forEach((s,i)=>{document.getElementById('title-'+i).textContent=s[1];draw(document.getElementById('chart-'+i),rows,...s)});
  const last=rows.at(-1);document.getElementById('epoch').textContent=last?.epoch??'—';document.getElementById('loss').textContent=fmt(last?.train_loss);
  const bestKey=run==='one_step'?'state_mae_physical_units':'selection_score';const values=rows.map(r=>+r[bestKey]).filter(Number.isFinite);
  document.getElementById('best-label').textContent=run==='one_step'?'最佳验证 MAE':'最佳选择指标';document.getElementById('best').textContent=values.length?fmt(Math.min(...values)):'—';
  document.getElementById('updated').textContent=payload.updated||'—';document.getElementById('status').textContent=payload.path+' · '+rows.length+' 条记录';
 }catch(e){document.getElementById('status').textContent='读取失败：'+e.message}
}
runSelect.addEventListener('change',refresh);window.addEventListener('resize',refresh);refresh();setInterval(refresh,3000);
</script></body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    root: Path

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/history":
            run = parse_qs(parsed.query).get("run", ["one_step"])[0]
            relative = {
                "one_step": "artifacts/tandem_fno_expanded_v1/training_history.json",
                "rollout": "artifacts/tandem_fno_rollout_expanded_v1/training_history.json",
                "no_tf": "artifacts/tandem_fno_rollout_no_tf_v1/training_history.json",
            }.get(run)
            if relative is None:
                self.send_bytes(b'{"error":"unknown run"}', "application/json", 400)
                return
            path = self.root / relative
            try:
                history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
                updated = (
                    __import__("datetime").datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
                    if path.exists() else None
                )
                body = json.dumps({"path": relative, "updated": updated, "history": history}).encode()
                self.send_bytes(body, "application/json; charset=utf-8")
            except (OSError, json.JSONDecodeError) as error:
                self.send_bytes(json.dumps({"error": str(error)}).encode(), "application/json", 503)
            return
        if parsed.path in ("/", "/index.html"):
            self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            return
        self.send_bytes(b"not found", "text/plain", 404)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DashboardHandler.root = args.root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
