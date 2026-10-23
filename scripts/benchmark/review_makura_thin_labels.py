"""Review source-traced ink centers, then diagnose masks without dense-GT claims."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi

from evaluate_classical_partial_labels import ordered_line
from probe_makura_damage_limits import ROOT,OUT,rgb,gray,dump,sha,rel

REGIONS={'200017197_00103_1_p00':(95,365,155,420),
 '200017197_00214_1_p00':(125,30,160,90),
 '200017197_00172_0_p00':(240,125,390,250)}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def sources():return {p['patch_id']:p for p in json.loads((OUT/'protocol.json').read_text())['selections']}


def render():
    source_map=sources();ann=json.loads((OUT/'thin_annotations_source_only.json').read_text())
    figures=[]
    for patch,box in REGIONS.items():
        source=source_map[patch];assert sha(ROOT/source['original'])==source['original_sha256']
        raw=rgb(ROOT/source['original']);overlay=raw.copy();labels=np.zeros((512,512),np.uint8)
        for path in [p for p in ann['paths'] if p['patch_id']==patch]:
            points=ordered_line(path['xy']);x,y=points.T
            overlay[y,x]=[0,100,255] if path['confidence']=='high' else [255,135,0]
            if path['confidence']=='high':labels[y,x]=1
        out=OUT/'source_thin_review';Image.fromarray(labels).save(out/(patch+'_high_centers.png'))
        x0,y0,x1,y1=box;scale=4;w=(x1-x0)*scale;h=(y1-y0)*scale
        canvas=Image.new('RGB',(max(2*w+60,900),h+140),'#faf9f5');draw=ImageDraw.Draw(canvas)
        def text(pos,s,size=16):draw.text(pos,s,font=ImageFont.truetype(FONT,size),fill='#203442')
        text((15,10),patch+' / 原画像による細線の位置確認',20)
        for i,array in enumerate((raw,overlay)):
            tile=np.repeat(np.repeat(array[y0:y1,x0:x1],scale,axis=0),scale,axis=1)
            canvas.paste(Image.fromarray(tile),(20+i*(w+20),55))
        text((15,canvas.height-65),'青=筆画中心、橙=低確信・主診断から除外。4倍最近傍表示。未着色は不明。')
        text((15,canvas.height-37),'Codexの暫定部分注釈。密な正解・専門家確認済み教師ではない。')
        dest=out/(patch+'_細線注釈確認.png');canvas.save(dest)
        figures.append(dict(patch_id=patch,source_sha256=source['original_sha256'],
            annotation_sha256=sha(OUT/'thin_annotations_source_only.json'),centers=int(labels.sum()),
            path=rel(dest),sha256=sha(dest),labels_sha256=sha(out/(patch+'_high_centers.png'))))
    dump(OUT/'thin_annotation_figure_verification.json',figures)
    print('Rendered source-only traces',sum(r['centers'] for r in figures),'high-confidence center pixels')


def score():
    source_map=sources();ann=json.loads((OUT/'thin_annotations_source_only.json').read_text())
    figures=json.loads((OUT/'thin_annotation_figure_verification.json').read_text())
    assert all(f['annotation_sha256']==sha(OUT/'thin_annotations_source_only.json') for f in figures)
    audits=[];result=[]
    all_records=json.loads((OUT/'all_results.json').read_text())
    for patch in REGIONS:
        source=source_map[patch];valid=gray(ROOT/source['valid'])>0;target=gray(ROOT/source['target'])==0
        points=np.unique(np.concatenate([ordered_line(p['xy']) for p in ann['paths'] if p['patch_id']==patch and p['confidence']=='high']),axis=0)
        x,y=points.T
        audits.append(dict(patch_id=patch,ink_center_pixels=len(points),
            old_teacher_valid_ink=int((valid&target)[y,x].sum()),
            old_teacher_valid_paper=int((valid&~target)[y,x].sum()),
            old_teacher_unknown=int((~valid)[y,x].sum()),
            formal_gt=False))
        for record in [r for r in all_records if r['patch_id']==patch]:
            folder=ROOT/record['folder']
            for method in record['scores']:
                pred=gray(folder/(method['method']+'.png'))==0
                distance=ndi.distance_transform_edt(~pred)
                result.append(dict(patch_id=patch,family=record['family'],strength=record['strength'],
                    method=method['method'],n=len(points),
                    hit_r0=int(pred[y,x].sum()),hit_r1=int((distance[y,x]<=1).sum()),
                    hit_r2=int((distance[y,x]<=2).sum()),formal_accuracy=False))
    dump(OUT/'source_thin_probe_diagnostics.json',dict(
        status='Codex source-only provisional ink centerlines, not expert/dense/blinded GT',
        annotation_sha256=sha(OUT/'thin_annotations_source_only.json'),
        primary_confidence='high only',tolerance='Euclidean distance, 0/1/2 native pixels',
        teacher_audit=audits,rows=result))
    print(json.dumps(audits,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['render','score']);args=parser.parse_args()
    if args.action=='render':render()
    else:score()
