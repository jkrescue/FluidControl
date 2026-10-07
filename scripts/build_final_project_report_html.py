#!/usr/bin/env python3
"""Render the reviewed Markdown report to a dependency-free offline HTML page."""
from pathlib import Path
import argparse
import re
import html
import markdown

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path.cwd())
args = parser.parse_args()
repo = args.repo.resolve()
md_path = repo / "docs/PROJECT_FINAL_REPORT_20261007.md"
out_path = repo / "docs/report_20261007/index.html"

body = markdown.markdown(
    md_path.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code", "sane_lists", "toc"],
    extension_configs={"toc": {"permalink": False}},
)
# The Markdown lives one directory above the offline HTML. Keep its links valid
# in Git while making image links local to report_20261007/index.html.
body = body.replace('src="report_20261007/assets/', 'src="assets/')
body = body.replace('href="report_20261007/assets/', 'href="assets/')
body = body.replace('href="report_20261007/index.html"', 'href="index.html"')
body = body.replace('href="report_20261007/technical_design.html"', 'href="technical_design.html"')
body = re.sub(
    r'href="((?!https?://|#|assets/|report_20261007/)[^"]+\.(?:md|json))"',
    r'href="../\1"',
    body,
)

css = r"""
:root { color-scheme: light; --ink:#172033; --muted:#5d6677; --line:#dce2ea;
  --blue:#185fa8; --soft:#f4f7fb; --good:#0f766e; --warn:#a16207; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
body { margin:0; color:var(--ink); background:#eef2f7;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  line-height:1.72; }
.shell { max-width:1180px; margin:0 auto; background:white; min-height:100vh;
  box-shadow:0 0 42px rgba(27,39,60,.10); }
header { padding:42px 56px 30px; color:white;
  background:linear-gradient(120deg,#102b4e,#185fa8 62%,#19857b); }
header h1 { margin:0 0 10px; font-size:34px; line-height:1.25; }
header p { margin:0; opacity:.9; }
.status { display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
.pill { border:1px solid rgba(255,255,255,.45); border-radius:999px; padding:5px 11px; font-size:13px; }
nav { position:sticky; top:0; z-index:5; padding:10px 22px; background:rgba(255,255,255,.96);
  border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
nav a { display:inline-block; margin:3px 12px 3px 0; color:var(--blue); text-decoration:none; font-size:13px; }
main { padding:34px 56px 70px; }
h1 { font-size:32px; } h2 { margin-top:52px; padding-top:10px; border-top:2px solid #b9c9dc; font-size:25px; }
h3 { margin-top:32px; font-size:20px; } h4 { font-size:17px; }
p, li { max-width:92ch; }
a { color:var(--blue); text-underline-offset:3px; }
blockquote { margin:20px 0; padding:12px 18px; background:#edf5ff; border-left:4px solid var(--blue); color:#30445f; }
table { width:100%; border-collapse:collapse; display:block; overflow:auto; margin:18px 0 28px; font-size:14px; }
th,td { border:1px solid var(--line); padding:9px 11px; vertical-align:top; min-width:110px; }
th { background:#edf3f9; text-align:left; }
tr:nth-child(even) td { background:#fafbfd; }
code { font-family:"SFMono-Regular",Consolas,monospace; background:#f0f3f7; border-radius:4px; padding:.12em .35em; }
pre { overflow:auto; padding:16px; background:#172033; color:#edf5ff; border-radius:8px; }
pre code { background:transparent; padding:0; }
img { max-width:100%; height:auto; display:block; margin:18px auto 8px; border:1px solid #cfd8e5; border-radius:8px; background:white; }
a > img { cursor:zoom-in; transition:box-shadow .15s ease; }
a > img:hover { box-shadow:0 8px 28px rgba(24,95,168,.20); }
strong { color:#102b4e; }
hr { border:0; border-top:1px solid var(--line); }
.note { margin:18px 0; padding:14px 18px; border-radius:8px; background:#fff8e8; border:1px solid #ead39a; }
footer { padding:26px 56px; background:#172033; color:#cbd5e1; font-size:13px; }
@media (max-width:720px) { header,main,footer { padding-left:20px; padding-right:20px; } header h1{font-size:27px;} nav{position:static;} }
@media print { body{background:white;} .shell{box-shadow:none;} nav{display:none;} main{padding:20px;} a{color:inherit;} }
"""

toc = "".join(
    f'<a href="#{anchor}">{html.escape(label)}</a>'
    for anchor, label in [
        ("1", "结论"), ("2", "目标与判据"),
        ("3-openfoamcurator", "数据链"), ("4", "模型"),
        ("5", "策略谱系"), ("6", "真实闭环"),
        ("7", "预测实验"), ("8", "证据边界"),
        ("9", "复现"), ("14", "证据索引")]
)

doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>串列双圆柱主动流动控制项目最终图文报告</title><style>{css}</style></head>
<body><div class="shell"><header><h1>串列双圆柱主动流动控制项目最终图文报告</h1>
<p>限定 Re=100、L/D=5 案例｜真实 OpenFOAM 反馈闭环已交付｜完整 FNO 精度与 MPC 目标仍未完成</p>
<div class="status"><span class="pill">默认策略：B</span><span class="pill">E114 新增 80 D/U</span>
<span class="pill">减阻 4.1326%</span><span class="pill">离线、无 CDN</span></div></header>
<nav><a href="technical_design.html">总体技术方案与实施框架</a>{toc}</nav><main>{body}</main>
<footer>生成自 docs/PROJECT_FINAL_REPORT_20261007.md。图片为已保存实验资产；点击图片可查看同目录原分辨率文件。此页面不包含在线脚本、遥测或 CDN。</footer>
</div></body></html>"""
out_path.write_text(doc, encoding="utf-8")
print(out_path)
