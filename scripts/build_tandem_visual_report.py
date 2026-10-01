#!/usr/bin/env python3
"""Build an offline, evidence-linked dashboard from completed tandem-CFD artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(project / "artifacts" / "visualization"):
        parser.error("output must be under the project artifacts/visualization directory")
    model = project / "artifacts/tandem_fno_expanded_spark_20epoch"
    observed = read_json(model / "heldout_evaluation.json")
    zero = read_json(model / "heldout_evaluation_zero.json")
    flip = read_json(model / "heldout_evaluation_sign_flip.json")
    history = read_json(model / "training_history.json")
    gate = read_json(model / "control_readiness.json")
    if gate["status"] != "CANDIDATE_SURROGATE_SCREEN_PASS":
        raise ValueError("the expected held-out surrogate gate is not present")
    if len(history) != 20 or len(observed["cases"]) != 4:
        raise ValueError("expected 20 training epochs and four independent held-out cases")
    audit_candidates = (
        project / "artifacts/hydrogym/tandem_ppo_multistart_8192_spark/audit_independent.json",
        project / "artifacts/hydrogym/tandem_ppo_pilot_20epoch_spark_guarded/audit.json",
        project / "artifacts/hydrogym/tandem_ppo_pilot_20epoch_spark/audit.json",
    )
    audit_path = next((path for path in audit_candidates if path.exists()), audit_candidates[-1])
    audit = read_json(audit_path) if audit_path.exists() else None
    checkpoint_path = audit_path.parent / "checkpoint_evaluations.json"
    checkpoints = read_json(checkpoint_path) if checkpoint_path.exists() else None
    real_candidates = (
        project / "artifacts/hydrogym/tandem_ppo_cfd_feedback_multistart_8192_32step_01/result.json",
        project / "artifacts/hydrogym/tandem_ppo_cfd_feedback_multistart_8192_01/result.json",
    )
    real_path = next((path for path in real_candidates if path.exists()), real_candidates[-1])
    real_cfd = read_json(real_path) if real_path.exists() else None
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    figure_index = []
    for case in observed["cases"]:
        for figure in case["visualizations"]:
            source = model / "heldout_figures" / case["case"] / Path(figure["path"]).name
            if not source.is_file():
                raise FileNotFoundError(source)
            name = f'{case["case"]}_{source.name}'
            shutil.copy2(source, assets / name)
            figure_index.append({"case": case["case"], "horizon": figure["horizon"], "file": f"assets/{name}"})
    data = {
        "history": history,
        "summary": observed["summary"],
        "zero_summary": zero["summary"],
        "flip_summary": flip["summary"],
        "figures": figure_index,
        "gate": gate,
        "audit": audit,
        "checkpoints": checkpoints,
        "real_cfd": real_cfd,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    if audit:
        evaluated = sum(row["status"] == "evaluated" for row in audit["evaluations"])
        test_mean = audit.get("summary", {}).get("test", {}).get("reward_change_mean")
        benefit = "尚无平均奖励改善" if test_mean is not None and test_mean <= 0 else "出现候选平均奖励改善"
        ppo_status = f"PPO 代理环境评估 {evaluated}/24 组完成；{benefit}，仍不能宣称真实 CFD 减阻。"
    else:
        ppo_status = "PPO 已训练并保存策略；24 组代理环境评估仍在运行。"
    page = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>串列双圆柱 · CFD / PhysicsNeMo / HydroGym 进展</title>
<style>
:root{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;color:#dce8f5;background:#0b1320;font-size:15px}
*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:28px 24px 70px}
h1{font-size:2rem;margin:0 0 8px;font-weight:600;letter-spacing:-.03em}h2{font-size:1.4rem;margin:0 0 14px;font-weight:600}h3{font-size:1.08rem;margin:0 0 12px}
p{line-height:1.6;margin:8px 0 12px}.sub{color:#9eb0c5}.eyebrow{color:#69d6bf;font-size:.79rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700}
nav{display:flex;gap:16px;flex-wrap:wrap;padding:18px 0 25px}nav a{color:#a9d8ff;text-decoration:none}nav a:hover{text-decoration:underline}
.status{border-left:3px solid #eab765;padding:11px 15px;background:#182538;margin:18px 0 26px;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px}.metric{padding:16px;background:#162337;border:1px solid #2a3b51;border-radius:8px}.metric strong{display:block;font-size:1.55rem;margin:6px 0;color:#f2f7fc}.metric small{color:#9eb0c5}
section{margin:36px 0 44px}.headrow{display:flex;justify-content:space-between;align-items:end;gap:16px;flex-wrap:wrap}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
label{color:#b9c8da}select{background:#17263b;border:1px solid #42617f;color:#f3f8fe;border-radius:6px;padding:8px 12px;font:inherit}
.figure{width:100%;height:auto;display:block;background:#fff;border-radius:4px;margin-top:17px}.caption{font-size:.88rem;color:#a9b8ca}
.chart{width:100%;height:auto;background:#111d2d;border-radius:6px}.chart text{font-family:inherit;fill:#b6c7da;font-size:13px}.chart .gridline{stroke:#344458;stroke-width:1}.chart .axis{stroke:#8196aa;stroke-width:1}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #304157;padding:9px 10px;text-align:left;white-space:nowrap}th{color:#9fc1df;font-weight:500}td.num{text-align:right;font-variant-numeric:tabular-nums}
.note{padding:13px 16px;background:#162337;border-radius:6px;color:#afc0d2}.good{color:#73e5b7}.warn{color:#ffd182}.bad{color:#f19a9a}
footer{border-top:1px solid #304157;margin-top:40px;padding-top:15px;color:#90a4bb;font-size:.85rem}code{overflow-wrap:anywhere}
@media(max-width:760px){main{padding:20px 14px}.grid,.pair{grid-template-columns:1fr}h1{font-size:1.5rem}.figure{min-width:680px}.figurewrap{overflow-x:auto}}
</style></head><body><main>
<div class="eyebrow">真实 OpenFOAM 数据 · PhysicsNeMo 预测 · HydroGym 代理控制</div>
<h1>串列双圆柱流场控制：当前可视化</h1>
<p class="sub">独立测试集上的真实 CFD 与 FNO 预测，附带训练过程、误差随预测步长变化，以及真实记录的强化学习状态。生成于 <span id="time"></span>。</p>
<nav><a href="#flow">流场对照</a><a href="#model">模型训练与误差</a><a href="#rl">强化学习</a><a href="#provenance">数据来源和限制</a></nav>
<div class="status"><strong>当前结论：</strong>20 轮 FNO 通过同几何、独立 CFD 轨迹的代理模型筛选门槛；这<strong>不是</strong>闭环减阻的证明。<span id="ppoStatus"></span></div>
<div class="grid"><div class="metric"><small>训练轮数</small><strong>20 epochs</strong><small>PhysicsNeMo 2.2.2 官方 FNO</small></div><div class="metric"><small>独立测试轨迹</small><strong>4 cases</strong><small>每个预测步长 64 段</small></div><div class="metric"><small>10 步流场 MAE</small><strong id="mae10"></strong><small>真实 CFD 对照；物理量单位</small></div></div>
<section id="flow"><div class="headrow"><div><h2>真实流场 · 预测 · 绝对误差</h2><p class="sub">每张图的三列分别来自真实 OpenFOAM 场、20 轮 FNO 预测、逐点绝对误差；三行依次为 u、v、p。</p></div><div class="controls"><label>测试轨迹 <select id="case"></select></label><label>预测步长 <select id="horizon"><option value="1">1 步</option><option value="10" selected>10 步</option><option value="50">50 步</option></select></label></div></div><div class="figurewrap"><img id="flowImage" class="figure" alt="真实 CFD、FNO 预测和绝对误差的九宫格流场图"></div><p id="figureCaption" class="caption"></p></section>
<section id="model"><h2>模型训练与独立测试</h2><div class="pair"><div><h3>20 轮验证集流场 MAE</h3><svg id="trainChart" class="chart" viewBox="0 0 580 290" role="img" aria-label="20 轮验证集流场平均绝对误差曲线"></svg><p class="caption">来自逐轮 <code>training_history.json</code>，不是人为平滑曲线。</p></div><div><h3>多步预测误差</h3><svg id="errorChart" class="chart" viewBox="0 0 580 290" role="img" aria-label="1、10、50 步预测的平均绝对误差"></svg><p class="caption">4 条独立测试轨迹，每个步长 64 段。图中包含真实动作、错误的零动作输入和状态保持基线。</p></div></div><div class="tablewrap"><table><thead><tr><th>预测步长</th><th>真实动作输入 MAE</th><th>错误零动作输入 MAE</th><th>状态保持基线 MAE</th><th>后圆柱力 MAE</th></tr></thead><tbody id="metricRows"></tbody></table></div><p class="note">50 步后圆柱力 MAE 已达 0.5921；虽然模型优于此处的基线，长期滚动误差仍不可忽略，不能直接用作闭环控制收益证据。</p></section>
<section id="rl"><h2>HydroGym 强化学习：实际记录</h2><p id="rlLead" class="sub"></p><div id="checkpointPanel" hidden><h3>独立面板奖励随 PPO 迭代变化</h3><svg id="ppoChart" class="chart" viewBox="0 0 1160 300" role="img" aria-label="PPO 环境步数与独立面板平均奖励曲线"></svg><p id="checkpointCaption" class="caption"></p></div><div id="realCfd"></div><div id="rlContent"></div><p class="note">训练和评估环境是冻结 FNO 驱动的 HydroGym 代理环境。最终表格比较的是“请求转速回到 0”的限速基线与训练后策略，并非未训练随机策略。真实 OpenFOAM 回放仍需更长时间、网格敏感性和重复工况验证。</p></section>
<section id="provenance"><h2>数据来源与科学边界</h2><p>流场图片源自本项目 OpenFOAM 串列双圆柱仿真数据集 <code>data/curated/tandem_cylinders_expanded_independent_v2</code> 的独立测试轨迹，由 <code>heldout_evaluation.json</code> 对应的评估脚本导出。不是港理工大学实验数据，也不是合成或手绘场图。</p><p>FNO 权重：<code>artifacts/tandem_fno_expanded_spark_20epoch/best</code>。表格、曲线和控制门槛分别来自训练历史、三种 held-out 评估 JSON 和 <code>control_readiness.json</code>。PPO 数据仅在 <code>audit.json</code> 形成后加入。</p><p class="sub">同几何离线泛化、代理模型控制与真实 CFD 闭环是三种不同的证据层级。这里仍处于前两层；中等网格收敛性验证也正在计算中。</p></section>
<footer>页面与 PNG 均可离线查看；原始仿真和训练仍在 DGX Spark 的 <code>~/workspace/fluid_control/physicsnemo_control</code>。页面生成脚本：<code>scripts/build_tandem_visual_report.py</code>。</footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>
const d=JSON.parse(document.getElementById('report-data').textContent);
document.getElementById('time').textContent=d.generated_utc;
document.getElementById('mae10').textContent=d.summary['10'].state_mae_physical_units.toFixed(4);
document.getElementById('ppoStatus').textContent=' '+__PPO_STATUS__;
const cs=document.getElementById('case'),hs=document.getElementById('horizon');
[...new Set(d.figures.map(x=>x.case))].forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c;cs.append(o)});
function flow(){const f=d.figures.find(x=>x.case===cs.value&&String(x.horizon)===hs.value);if(!f)return;const im=document.getElementById('flowImage');im.src=f.file;im.alt=`${f.case}，${f.horizon} 步：真实 OpenFOAM 流场、FNO 预测和绝对误差`;document.getElementById('figureCaption').textContent=`${f.case} / 预测 ${f.horizon} 步。图题中的 omega 是评估目标帧的动作；Cd/Cl 为后圆柱力系数。`}
cs.addEventListener('change',flow);hs.addEventListener('change',flow);flow();
const NS='http://www.w3.org/2000/svg';function el(tag,attrs,parent){const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));parent.appendChild(n);return n}
function txt(svg,x,y,value,anchor='start'){const t=el('text',{x,y,'text-anchor':anchor},svg);t.textContent=value;return t}
function base(svg,maxY,xlabels){const L=67,R=18,T=22,B=49,W=580,H=290,plotW=W-L-R,plotH=H-T-B;for(let i=0;i<=4;i++){let y=T+plotH*i/4;el('line',{x1:L,y1:y,x2:W-R,y2:y,class:'gridline'},svg);txt(svg,L-8,y+5,(maxY*(4-i)/4).toFixed(maxY<.02?3:2),'end')}el('line',{x1:L,y1:T,x2:L,y2:H-B,class:'axis'},svg);xlabels.forEach(([x,label])=>txt(svg,L+x*plotW,H-16,label,'middle'));return {L,T,plotW,plotH}}
{
 const svg=document.getElementById('trainChart'),vals=d.history.map(x=>x.state_mae_physical_units),max=Math.max(...vals)*1.15,p=base(svg,max,[[0,'1'],[4/19,'5'],[9/19,'10'],[14/19,'15'],[1,'20']]);
 const points=vals.map((v,i)=>`${p.L+i/19*p.plotW},${p.T+(1-v/max)*p.plotH}`).join(' ');
 el('polyline',{points,fill:'none',stroke:'#64d9bd','stroke-width':3,'stroke-linejoin':'round'},svg);vals.forEach((v,i)=>el('circle',{cx:p.L+i/19*p.plotW,cy:p.T+(1-v/max)*p.plotH,r:3,fill:'#64d9bd'},svg));txt(svg,580-25,36,`epoch 20: ${vals.at(-1).toFixed(4)}`,'end');
}
{
 const svg=document.getElementById('errorChart'),keys=['1','10','50'],series=[['真实动作','#64d9bd',d.summary],['错误零动作','#f5be72',d.zero_summary],['状态保持','#8ba8f6',d.summary]],max=.19,p=base(svg,max,[[.13,'1 步'],[.5,'10 步'],[.87,'50 步']]);
 series.forEach(([name,color,src],si)=>{keys.forEach((k,i)=>{const v=si===2?src[k].persistence_state_mae_physical_units:src[k].state_mae_physical_units;const x=p.L+([.13,.5,.87][i]*p.plotW)+(si-1)*15,y=p.T+(1-v/max)*p.plotH;el('rect',{x:x-5,y,width:10,height:p.T+p.plotH-y,fill:color},svg)});el('rect',{x:90+si*160,y:12,width:11,height:11,fill:color},svg);txt(svg,107+si*160,22,name)})
}
for(const k of ['1','10','50']){const a=d.summary[k],z=d.zero_summary[k],tr=document.createElement('tr');[k,a.state_mae_physical_units.toFixed(4),z.state_mae_physical_units.toFixed(4),a.persistence_state_mae_physical_units.toFixed(4),a.rear_force_mae.toFixed(4)].forEach((v,i)=>{const c=document.createElement('td');c.textContent=v;if(i)c.className='num';tr.append(c)});document.getElementById('metricRows').append(tr)}
if(d.checkpoints){document.getElementById('checkpointPanel').hidden=false;const svg=document.getElementById('ppoChart'),vals=d.checkpoints.map(x=>x.reward_sum_mean),steps=d.checkpoints.map(x=>x.timesteps),L=78,R=28,T=25,B=50,W=1160,H=300,pw=W-L-R,ph=H-T-B,lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.25,.0001),y0=lo-pad,y1=hi+pad;for(let i=0;i<=4;i++){const y=T+ph*i/4,v=y1-(y1-y0)*i/4;el('line',{x1:L,y1:y,x2:W-R,y2:y,class:'gridline'},svg);txt(svg,L-9,y+5,v.toFixed(4),'end')}const pts=vals.map((v,i)=>`${L+i/(vals.length-1)*pw},${T+(y1-v)/(y1-y0)*ph}`).join(' ');el('polyline',{points:pts,fill:'none',stroke:'#64d9bd','stroke-width':3,'stroke-linejoin':'round'},svg);vals.forEach((v,i)=>{const x=L+i/(vals.length-1)*pw,y=T+(y1-v)/(y1-y0)*ph;el('circle',{cx:x,cy:y,r:5,fill:'#64d9bd'},svg);txt(svg,x,H-17,String(steps[i]),'middle');txt(svg,x,y-10,v.toFixed(4),'middle')});document.getElementById('checkpointCaption').textContent=`${steps.join('、')} 步均为实际保存的策略快照；固定面板包含 4 条验证和 4 条测试轨迹的 frame 100，每组滚动 ${d.checkpoints[0].horizon} 步。`}
if(d.real_cfd){const r=d.real_cfd,delta=r.mean_objective_difference_from_zero;document.getElementById('realCfd').innerHTML=`<div class="status"><strong class="${delta<0?'good':'warn'}">真实 OpenFOAM 闭环 ${r.steps} 步：</strong>平均瞬时目标相对零转速基线差为 ${delta.toFixed(6)}（负值更好）。这是短时诊断，不是长期减阻结论。</div>`}
const lead=document.getElementById('rlLead'),content=document.getElementById('rlContent');
if(!d.audit){lead.textContent='512 个 PPO 环境步已完成、策略已保存；跨验证集和测试集的 24 组对照仍在运行。';content.innerHTML='<p class="status">目前只有训练完成和策略落盘的证据，尚无完成的跨案例奖励对照。页面将在评估 JSON 产生后重新生成。</p>'}
else{const a=d.audit,valid=a.evaluations.filter(r=>r.status==='evaluated'),stable=valid.length===24,testMean=a.summary?.test?.reward_change_mean??0,benefit=testMean>0;lead.textContent=`PPO ${a.ppo_timesteps} 环境步、${a.episode_steps} 步/回合。${valid.length}/24 组代理环境评估完成；去重后的测试起点平均奖励差 ${testMean.toFixed(4)}，${benefit?'出现候选改善':'没有整体改善证据'}。`;let h=`<div class="status"><strong class="${stable&&benefit?'good':stable?'warn':'bad'}">${stable?'稳定性审计通过：':'稳定性审计未通过：'}</strong>${stable&&benefit?'共享 t=80 起点只计一次后，验证和测试平均奖励均改善；仍需真实 CFD 验证。':stable?'24 组均完整滚动，但平均奖励没有改善。':'存在提前终止或失败组。'}下表首先保留全部原始评估行。</div><div class="tablewrap"><table><thead><tr><th>分组</th><th>案例 / 起点</th><th>零请求奖励</th><th>${a.ppo_timesteps} 步 PPO 奖励</th><th>奖励差</th><th>平均 Cd 差</th><th>状态</th></tr></thead><tbody>`;for(const row of a.evaluations){let good=row.status==='evaluated';h+=`<tr><td>${row.split}</td><td>${row.case} / ${row.initial_frame}</td><td class="num">${good?row.zero.reward_sum.toFixed(3):'—'}</td><td class="num">${good?row.policy.reward_sum.toFixed(3):'—'}</td><td class="num">${good?row.reward_change.toFixed(3):'—'}</td><td class="num">${good?row.cd_mean_change.toFixed(3):'—'}</td><td class="${good?(row.reward_change>0?'good':'warn'):'bad'}">${good?(row.reward_change>0?'奖励改善':'未改善'):'评估失败'}</td></tr>`}h+='</tbody></table></div>';content.innerHTML=h}
</script></body></html>'''
    page = page.replace("__DATA__", payload).replace("__PPO_STATUS__", json.dumps(ppo_status, ensure_ascii=False))
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"generated_utc": data["generated_utc"], "figure_count": len(figure_index), "ppo_audit_included": audit is not None}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": len(figure_index), "ppo_audit_included": audit is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
