"""Render the existing project brief as a restrained, printable offline page."""
from pathlib import Path
import markdown

root = Path(__file__).resolve().parents[1]
source = root / 'docs/PROJECT_BRIEF_20261007.md'
body = markdown.markdown(source.read_text(), extensions=['tables'])
css = '''
@page {size:A4;margin:17mm 18mm;}
*{box-sizing:border-box}
body{margin:0;background:#eee;color:#222;font-family:"PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.8}
main{max-width:820px;margin:28px auto;padding:38px 48px;background:white}
h1{font-size:23px;text-align:center;margin:0 0 12px;font-weight:600}
h1+p{text-align:center;color:#555;font-size:13px;margin-bottom:24px}
h2{font-size:17px;margin:19px 0 7px;break-after:avoid}
p{margin:8px 0;text-align:justify;orphans:3;widows:3}
strong{font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0;break-inside:avoid}
th,td{text-align:left;border:1px solid #bbb;padding:5px 9px}
th{background:#f5f5f5;font-weight:600}
main>p:last-child{font-size:11px;color:#666;margin-top:18px}
@media print{body{background:white;font-size:10pt;line-height:1.65}main{max-width:none;margin:0;padding:0}h1{font-size:17pt}h2{font-size:12pt;margin:12px 0 5px}table{font-size:9pt}h1+p{margin-bottom:14px}}
@media(max-width:600px){main{margin:0;padding:24px}h1{font-size:21px}}
'''
page = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>串联双圆柱主动流动控制项目工作简报</title><style>{css}</style></head><body><main>{body}</main></body></html>\n'
target = root / 'docs/report_20261007/project_brief.html'
target.write_text(page, encoding='utf-8')
print(target)
