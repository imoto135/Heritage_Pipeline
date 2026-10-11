from pathlib import Path
import json, re, hashlib, zipfile, subprocess, shutil
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
report=OUT/'進捗報告.md'
missing=[]
for ref in re.findall(r'\]\(([^)]+)\)',report.read_text()):
    if not ref.startswith(('http:', 'https:')) and not (OUT/ref).exists(): missing.append(ref)
assert not missing,missing
sources=json.loads((OUT/'image_sources.json').read_text())
figures={p.name[:2]:p for p in (OUT/'figures').glob('*.png')}
assert len(figures)==10
results=[]
for s in sources:
    src=ROOT/s['source']; crop=Image.open(OUT/s['native']).convert('RGB')
    assert hashlib.sha256(src.read_bytes()).hexdigest()==s['sha256']
    x,y,w,h=s['crop_xywh']
    expected=Image.open(src).convert('RGB').crop((x,y,x+w,y+h))
    assert np.array_equal(np.asarray(expected),np.asarray(crop))
    k=s['scale']; expanded=np.asarray(crop.resize((w*k,h*k),Image.Resampling.NEAREST))
    px,py=s['paste_xy']
    rendered=np.asarray(Image.open(figures[s['panel'][:2]]).crop((px,py,px+w*k,py+h*k)))
    allowed=np.zeros((h*k,w*k),bool)
    if s['panel'].startswith('04_'):
        # Explicit locator outline only; interior still compared to native pixels.
        xx=(650-615)*5; yy=(446-411)*5
        for width in range(3):
            allowed[yy-5+width,xx-5:xx+56]=True
            allowed[yy+50-width,xx-5:xx+56]=True
            allowed[yy-5:yy+51,xx-5+width]=True
            allowed[yy-5:yy+51,xx+55-width]=True
    assert not np.any(np.any(rendered!=expanded,axis=2)&~allowed),s['panel']
    results.append(dict(panel=s['panel'],source_hash_matches=True,native_exact=True,final_panel_exact_except_declared_locator=True))
pdf=OUT/'NPJ進捗報告_20261007.pdf'
assert pdf.read_bytes().startswith(b'%PDF-') and pdf.stat().st_size>100000
pdf_pages=len(re.findall(rb'/Type\s*/Page\b',pdf.read_bytes()))
assert pdf_pages==10,pdf_pages
verification=json.loads((OUT/'verification.json').read_text())
verification.update(final_file_checks=results,report_links_all_resolve=True,pdf_pages=pdf_pages,visual_review='All 10 PNG pages inspected for labels, crops, and text layout.')
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
readme='''NPJからの進捗報告（2026-10-07）

まず「NPJ進捗報告_20261007.pdf」（10ページ）を開いてください。
詳しい経緯、成功・失敗の一覧、出典、2分説明原稿は「進捗報告.md」です。
figures/ のPNGが比較画像の一次成果物です。native_crops/ は無注釈の原寸画像です。
このZIPには報告資料と作図用スクリプト、検証記録を含みます。
進捗報告.mdから ../ で参照する元実験記録は元のプロジェクト内にあります。
作図スクリプトの再実行には元の実験出力が必要です。

新しい学習・モデル推論は行っていません。実験記録の対象は2026-10-04までです。
孤立斑点v3の画素数を保存画像から訂正：指定7,259、変更7,227、指定外変更0。
'''
(OUT/'はじめに.txt').write_text(readme)
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts)
hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='artifact_hashes.json'}
(OUT/'artifact_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts)
zp=OUT/'NPJ進捗報告_画像資料一式.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.relative_to(OUT))
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None
    for p in files:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
print(json.dumps(dict(figures=len(figures),panels=len(results),pdf_pages=pdf_pages,links='all resolved',zip_files=len(files),pdf_mb=round(pdf.stat().st_size/1e6,2),zip_mb=round(zp.stat().st_size/1e6,2)),ensure_ascii=False))
