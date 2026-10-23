"""Exploratory closing after per-region oracle binarization; not a proposal."""
from pathlib import Path
import importlib.util
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

P=Path(__file__).resolve().parent/'evaluate_classical_partial_labels.py'
spec=importlib.util.spec_from_file_location('partial_eval',P)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
OUT=a.OUT/'closing_diagnostic';OUT.mkdir(exist_ok=True)
KS=[1,3,5,7,9,13,17]


def main():
    regions=json.loads((a.OUT/'source_regions.json').read_text())
    ann={r['id']:r for r in json.loads((a.OUT/'annotations_source_only.json').read_text())['regions']}
    frontiers=json.loads((a.OUT/'frontiers.json').read_text())
    cases={c['id']:c for c in json.loads((a.BASE/'protocol.json').read_text())['cases']}
    a.dump(OUT/'protocol.json',dict(kernels=KS,shape='OpenCV ellipse',role='Post-hoc diagnostic after parameter search; not held-out evaluation',
        annotation_sha256=a.sha(a.OUT/'annotations_source_only.json'),input='Existing full-page binarization cropped with guard band; annotation ROI is > 2 kernel radii inside guard band',
        caveat='Closing can join wrong strokes. Selected nonink patches do not establish shape fidelity or dense precision.'))
    rows=[];figures=[]
    for region in regions:
        rid=region['id'];labels,paths=a.raster(region,ann[rid]);x,y,w,h=region['roi'];base=cases[region['case']];bx,by,bw,bh=base['roi']
        assert min(x-bx,y-by,bx+bw-x-w,by+bh-y-h)>2*(max(KS)//2)
        raw=a.read(a.OUT/(rid+'_original.png'))
        for family in ['howe','sauvola']:
            key=next(f['oracle']['key'] for f in frontiers if f['region']==rid and f['mode']=='high' and f['family']==family)
            pred=np.asarray(Image.open(a.BASE/'crops'/region['case']/(key+'.png')))==0
            images={}
            for k in KS:
                kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
                full=cv2.morphologyEx(pred.astype(np.uint8),cv2.MORPH_CLOSE,kernel)>0
                cut=a.crop(full,[x-bx,y-by,w,h]);images[k]=np.repeat(np.where(cut,0,255).astype(np.uint8)[...,None],3,2)
                rows.append(dict(region=rid,family=family,key=key,closing_kernel=k,**a.measure(cut,labels,paths)))
            if rid not in ['thin','water']:continue
            scale=3;pw,ph=w*scale,h*scale;gap=20;head=100;lh=42
            canvas=Image.new('RGB',(3*pw+4*gap,head+2*(ph+lh+gap)+85),'#faf9f5');d=ImageDraw.Draw(canvas)
            font=ImageFont.truetype(a.FONT,17)
            d.text((15,12),rid+' / '+family+' 領域別最良 + closing',font=ImageFont.truetype(a.FONT,25),fill='#203442')
            d.text((15,51),'元設定：'+key+' / 同じ範囲・整数3倍',font=font,fill='#203442')
            panels=[]
            for i,(title,tile) in enumerate([('原画像',raw),('後処理なし',images[1]),('closing 3',images[3]),('closing 5',images[5]),('closing 9',images[9]),('closing 17',images[17])]):
                tile=np.repeat(np.repeat(tile,scale,0),scale,1);px=gap+(i%3)*(pw+gap);py=head+(i//3)*(ph+lh+gap)
                d.text((px,py),title,font=font,fill='#203442');canvas.paste(Image.fromarray(tile),(px,py+lh));panels.append(dict(title=title,xywh=[px,py+lh,pw,ph],expected=tile))
            d.text((15,canvas.height-65),'大きいカーネルは本来別の線を結ぶ可能性もある。字形の正解は未確認。',font=font,fill='#203442')
            d.text((15,canvas.height-38),'既知の開発用部分注釈を使う追加診断。全ページの改善として採用しない。',font=font,fill='#203442')
            path=OUT/(rid+'_'+family+'_closing比較.png');canvas.save(path);saved=a.read(path)
            for p in panels:assert np.array_equal(a.crop(saved,p['xywh']),p.pop('expected'))
            figures.append(dict(path=str(path.relative_to(a.ROOT)),sha256=a.sha(path),panels=panels))
    a.dump(OUT/'scores.json',rows);a.dump(OUT/'figure_verification.json',figures)
    print('Closing checks:',len(rows),'conditions; figures',len(figures))
    for r in rows:
        if r['region']=='thin':print(r['family'],r['closing_kernel'],r['ink_r2'],r['ink_n'],r['nonink_false'],[(p['id'],p['connected']) for p in r['paths']])


if __name__=='__main__':main()
