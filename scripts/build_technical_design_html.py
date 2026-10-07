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
    css = style_match.group(1) + "\nhtml{scroll-behavior:auto}.part-label{margin-top:36px;padding:14px;background:#e8f2fb;font-weight:700}.design-section{scroll-margin-top:85px}h2,h3,a[id]{scroll-margin-top:85px}a[id]{display:block}.shell{max-width:1280px}"
    nav_items = [("#blueprint", "总体蓝图"), ("#implementation", "实施框架"), ("#stages", "阶段与验收"), ("#data-stack", "数据与工具栈"), ("#surrogate", "模型训练"), ("#control", "MPC/PPO闭环"), ("index.html", "实测图文报告")]
    nav = " ".join(f'<a href="{url}">{html.escape(label)}</a>' for url, label in nav_items)
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>串列双圆柱：总体技术方案、实施框架与技术栈</title><style>{css}</style></head>
<body><div class="shell"><header><h1>总体技术方案与实施框架</h1>
<p>先看蓝图 → 再看如何开展 → 最后查数据、训练与控制的详细实现</p>
<div class="status"><span class="pill">保留既有B/PPO成果</span><span class="pill">Re=100 / L/D=5</span><span class="pill">已实现与规划分开</span><span class="pill">离线完整页面</span></div></header>
<nav>{nav}</nav><main>{''.join(chunks)}</main>
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
    diagram = output / "assets/technical_blueprint.svg"
    diagram_bytes = diagram.read_bytes()
    sources.append({"path": str(diagram.relative_to(repo)), "sha256": hashlib.sha256(diagram_bytes).hexdigest(), "bytes": len(diagram_bytes), "kind": "architecture schematic, not a measured result"})
    script = Path(__file__).resolve()
    source_bytes = script.read_bytes()
    record = {"purpose": "documentation build only; no scientific experiment", "inputs": sources, "stylesheet_source": "docs/report_20261007/index.html", "stylesheet_sha256": hashlib.sha256(style_match.group(1).encode("utf-8")).hexdigest(), "renderer_sha256": hashlib.sha256(source_bytes).hexdigest(), "html_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(), "local_links_missing": missing, "html_id_count": len(ids)}
    (output / "technical_design_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(page_path), "source_documents": len(sections), "broken_local_links": len(missing), "html_id_count": len(ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
