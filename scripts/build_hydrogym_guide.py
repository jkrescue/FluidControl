"""Render a three-section, print-paginated HydroGym project explanation."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit
import markdown

root = Path(__file__).resolve().parents[1]
source = root / 'docs/HYDROGYM_PROJECT_GUIDE_20261007.md'
out = root / 'docs/report_20261007'
sections = source.read_text().split('<!-- pagebreak -->')
assert len(sections) == 3
pages = []
for i, raw in enumerate(sections, 1):
    body = markdown.markdown(raw, extensions=['tables'])
    def relocate(match):
        attr, value = match.groups()
        if not urlsplit(value).scheme:
            value = value.removeprefix('report_20261007/') if value.startswith('report_20261007/') else '../' + value
        return f'{attr}="{value}"'
    body = re.sub(r'(href|src)="([^"]+)"', relocate, body)
    pages.append(f'<section class="page" id="page{i}">{body}<footer>{i} / 3</footer></section>')
css = '''
@page{size:A4;margin:15mm 17mm}
*{box-sizing:border-box}body{margin:0;background:#eee;color:#222;font-family:"PingFang SC","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.75}
.page{max-width:820px;margin:24px auto;padding:32px 44px;background:white}h1{font-size:24px;margin:0 0 10px;text-align:center}h1+p{text-align:center;color:#666;font-size:12px}
h2{font-size:19px;margin:16px 0 12px}h3{font-size:15px;margin:14px 0 7px}p{margin:9px 0;text-align:justify}strong{font-weight:600}
img{width:100%;height:auto;display:block;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:13px}th,td{border:1px solid #bbb;padding:5px 8px;text-align:left}th{background:#f5f5f5}
a{color:#245879}code{font-size:.92em}footer{text-align:right;color:#777;font-size:11px;margin-top:18px}h2,h3{break-after:avoid}p,table{orphans:3;widows:3}table,img{break-inside:avoid}
@media print{body{background:white;font-size:10pt;line-height:1.6}.page{max-width:none;margin:0;padding:0;break-after:page}.page:last-child{break-after:auto}h1{font-size:17pt}h2{font-size:13pt}h3{font-size:11pt}table{font-size:9pt}img{margin:8px 0}a{color:inherit;text-decoration:none}footer{margin-top:12px}}
@media(max-width:600px){.page{padding:20px;margin:12px 0}h1{font-size:21px}}
'''
page = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HydroGym在双圆柱流动控制项目中的应用</title><style>' + css + '</style></head><body>' + ''.join(pages) + '</body></html>\n'
local = [out / v for v in re.findall(r'(?:src|href)="([^"]+)"', page) if not urlsplit(v).scheme]
assert all(p.exists() for p in local), [str(p) for p in local if not p.exists()]
(out / 'hydrogym_guide.html').write_text(page)
bound = [source, Path(__file__).resolve()] + [p for p in local if p.suffix == '.svg']
manifest = {'kind':'documentation only; diagrams are schematics, not CFD results', 'sections':3, 'inputs':[{'path':str(p.relative_to(root)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in bound], 'html_sha256':hashlib.sha256(page.encode()).hexdigest()}
(out / 'hydrogym_guide_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'sections':3,'local_links_checked':len(local),'output':str(out/'hydrogym_guide.html')}))
