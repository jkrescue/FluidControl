#!/usr/bin/env python3
"""Build the offline technical blueprint and integrated implementation appendices.

Documentation only: no scientific imports, subprocess, model, or CFD execution.
"""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import markdown
from markdown.extensions.toc import slugify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    docs = repo / "docs"
    output = docs / "report_20261007"
    sections = [
        ("blueprint", "总体蓝图与实施框架", "TECHNICAL_BLUEPRINT_20261007.md"),
        ("data-stack", "分册一：数据与技术栈", "TECHNICAL_DESIGN_DATA_STACK_20261007.md"),
        ("surrogate", "分册二：代理模型训练", "TECHNICAL_DESIGN_SURROGATE_TRAINING_20261007.md"),
        ("control", "分册三：MPC、PPO与闭环", "TECHNICAL_DESIGN_CONTROL_ROADMAP_20261007.md"),
    ]
    anchors = {name: "#" + key for key, _, name in sections}
    chunks = []
    sources = []
    for key, title, name in sections:
        source = docs / name
        raw = source.read_bytes()
        sources.append({"path": "docs/" + name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        body = markdown.markdown(
            raw.decode("utf-8"),
            extensions=["tables", "fenced_code", "sane_lists", "toc"],
            extension_configs={"toc": {"slugify": lambda value, sep, prefix=key: prefix + "-" + slugify(value, sep)}},
        )

        def relocate(match):
            attr, target = match.groups()
            if urlsplit(target).scheme or target.startswith(("#", "//")):
                return match.group(0)
            if target in anchors:
                target = anchors[target]
            elif target.startswith("report_20261007/"):
                target = target[len("report_20261007/"):]
            else:
                target = "../" + target
            return f'{attr}="{target}"'

        body = re.sub(r'(href|src)="([^"]+)"', relocate, body)
        chunks.append(f'<section id="{key}" class="design-section"><div class="part-label">{html.escape(title)}</div>{body}</section>')

    # Reuse the already-reviewed report stylesheet, not any remote stylesheet/CDN.
    report_html = (output / "index.html").read_text(encoding="utf-8")
    style_match = re.search(r"<style>(.*?)</style>", report_html, re.S)
    if not style_match:
        raise ValueError("report stylesheet missing; build the main report first")
    css = style_match.group(1) + "\nhtml{scroll-behavior:auto}.part-label{margin-top:36px;padding:14px;background:#e8f2fb;font-weight:700}.design-section{scroll-margin-top:85px}h2,h3,a[id]{scroll-margin-top:85px}a[id]{display:block}.shell{max-width:1280px}.reading-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:16px 0 32px}.reading-card{display:block;padding:18px;border:1px solid #c4d7e8;border-radius:10px;background:#f3f8fc;text-decoration:none}.reading-card strong{display:block;font-size:19px;margin-bottom:8px}.reading-card span{color:#40546a;font-size:15px}@media(max-width:720px){.reading-grid{grid-template-columns:1fr}}"
    nav_items = [("#blueprint", "总体蓝图"), ("#model-network", "网络结构"), ("#model-training", "FNO训练配置"), ("#hydrogym-explained", "HydroGym做什么"), ("#ppo-training", "PPO网络与步数"), ("#mpc-explained", "MPC算法"), ("#mpc-example", "MPC真实例子"), ("#implementation", "实施框架"), ("index.html", "实测结果")]
    nav = " ".join(f'<a href="{url}">{html.escape(label)}</a>' for url, label in nav_items)
    guide = '''<div class="reading-grid" aria-label="从实际问题阅读">
<a class="reading-card" href="#model-network"><strong>网络到底是什么？</strong><span>两个官方二维FNO：逐层结构、输入输出、模态数和冻结范围；有结构图。</span></a>
<a class="reading-card" href="#model-training"><strong>怎样训练，训练了多少？</strong><span>B微调的256个窗口、32次AdamW更新；配置作用和实际计算过程逐项解释。</span></a>
<a class="reading-card" href="#hydrogym-explained"><strong>HydroGym具体做了什么？</strong><span>69个数各代表什么；从重置、提出动作、预测下一步到奖励；另列PPO网络与训练计数。</span></a>
<a class="reading-card" href="#mpc-example"><strong>MPC为什么选这个转速？</strong><span>用一次真实记录比较5个候选、预测得分和CFD结果；完整算法紧邻例子。</span></a>
</div>'''
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>串列双圆柱：总体技术方案、实施框架与技术栈</title><style>{css}</style></head>
<body><div class="shell"><header><h1>总体技术方案与实施框架</h1>
<p>总体方案 + 实际实现详解：网络、训练参数和步数、HydroGym交互、MPC真实决策</p>
<div class="status"><span class="pill">保留既有B/PPO成果</span><span class="pill">Re=100 / L/D=5</span><span class="pill">已实现与规划分开</span><span class="pill">离线完整页面</span></div></header>
<nav>{nav}</nav><main>{guide}{''.join(chunks)}</main>
<footer>设计说明，不是新实验结果或自动执行指令。实际物理结果、失败与来源见实测图文报告；图中虚线为待验证路线。所有内容本地可阅读，无遥测、JavaScript或CDN。</footer></div></body></html>'''
    ids = re.findall(r'\bid="([^"]+)"', page)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate HTML anchor")
    missing = []
    for target in re.findall(r'(?:href|src)="([^"]+)"', page):
        parts = urlsplit(target)
        if parts.scheme or target.startswith("//"):
            continue
        if not parts.path:
            if parts.fragment and parts.fragment not in ids:
                missing.append(target)
        elif not (output / parts.path).resolve().exists():
            missing.append(target)
    if missing:
        raise ValueError("broken local links: " + repr(missing))
    page_path = output / "technical_design.html"
    page_path.write_text(page, encoding="utf-8")
    for name in ("technical_blueprint.svg", "fno_network_structure.svg", "training_vs_deployment.svg"):
        diagram = output / "assets" / name
        diagram_bytes = diagram.read_bytes()
        sources.append({"path": str(diagram.relative_to(repo)), "sha256": hashlib.sha256(diagram_bytes).hexdigest(), "bytes": len(diagram_bytes), "kind": "implementation schematic, not a measured result"})
    example = output / "evidence/b_h5_mpc_result.json"
    example_bytes = example.read_bytes()
    example_sha = hashlib.sha256(example_bytes).hexdigest()
    if example_sha != "f189508e962e17c5e98fa8fa6381c18664a4af9a58bd23da5da6977a50324939":
        raise ValueError("saved MPC worked-example evidence differs from reviewed result")
    sources.append({"path": str(example.relative_to(repo)), "sha256": example_sha, "bytes": len(example_bytes), "kind": "exact copy of existing ten-cycle result; worked example uses rows[0]"})
    script = Path(__file__).resolve()
    source_bytes = script.read_bytes()
    record = {"purpose": "documentation build only; no scientific experiment", "inputs": sources, "stylesheet_source": "docs/report_20261007/index.html", "stylesheet_sha256": hashlib.sha256(style_match.group(1).encode("utf-8")).hexdigest(), "renderer_sha256": hashlib.sha256(source_bytes).hexdigest(), "html_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(), "local_links_missing": missing, "html_id_count": len(ids)}
    (output / "technical_design_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(page_path), "source_documents": len(sections), "broken_local_links": len(missing), "html_id_count": len(ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
