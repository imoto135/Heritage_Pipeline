"""Screen original thumbnails; review saved corpus npj v1 before frozen v6.

Screening scores are candidate finders, never failure labels or ink GT.
All paths and artifacts remain within this project.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/npj_failure_collection_20261001'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import sha, dump, overlay, figure, write, FONT_PATH
cv2.setNumThreads(1)


def inventory():
    return list(csv.DictReader((OUT/'page_inventory.csv').open()))


def screen_one(row):
    im = cv2.imread(str(ROOT/row['original']), cv2.IMREAD_REDUCED_COLOR_8)
    if im is None:
        raise OSError(row['original'])
    h,w=im.shape[:2]
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg=cv2.GaussianBlur(gray,(0,0),8)
    b,g,r=cv2.split(im.astype(np.float32))
    interior=np.zeros((h,w),bool); interior[round(h*.08):round(h*.92),round(w*.08):round(w*.92)]=True
    brown=(r-b>18)&(r-g>4)&(bg-gray>15)&interior
    count,labels,stats,_=cv2.connectedComponentsWithStats(brown.astype(np.uint8))
    blobs=[]
    for i in range(1,count):
        x,y,bw,bh,a=map(int,stats[i])
        if a>=25 and max(bw/bh,bh/bw)<=3 and a/(bw*bh)>.35:
            blobs.append((a,x,y,bw,bh))
    blob=max(blobs,default=(0,0,0,0,0))
    bright=(gray-bg>18)&interior
    bright_score=int(bright.sum())
    return dict(page=row['page'],book=row['book'],split=row['split'],
                brown_blob_score=blob[0],brown_blob_thumb_xywh=list(blob[1:]),
                bright_damage_score=bright_score,brown_pixels=int(brown.sum()),
                thumbnail_width=w,thumbnail_height=h)


def screen():
    rows=inventory()
    scores=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,value in enumerate(pool.map(screen_one,rows),1):
            scores.append(value)
            if i%500==0: print('Original thumbnails screened:',i,'/',len(rows),flush=True)
    with (OUT/'screening_scores.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scores[0])); writer.writeheader();writer.writerows(scores)
    selected=[]
    # Two separate rankings, at most one page per book in each ranking.
    for metric in ['brown_blob_score','bright_damage_score']:
        books=set()
        for row in sorted(scores,key=lambda s:(-s[metric],s['page'])):
            if row['book'] in books or row[metric]==0: continue
            books.add(row['book'])
            if row['page'] not in [r['page'] for r in selected]:
                selected.append(dict(**row,screening_reason=metric))
            if len(books)==20: break
    known=json.loads((ROOT/'tmp_work/warm_ink_hold_v6_20260930/manifest.json').read_text())['cases']
    ids={r['page'] for r in selected}
    for case in known:
        if case['id'] not in ids and any(r['page']==case['id'] for r in rows):
            selected.append(dict(page=case['id'],book=case['book'],screening_reason='existing_diagnostic'))
    dump(OUT/'screening.json',dict(pages_screened=len(rows),selection=selected,
        purpose='Original-only candidate screening, not confirmed failures; v6 not consulted',
        source_sha256=sha(Path(__file__)),illustration_filter='visual page review required',
        rankings=['brown_blob_score','bright_damage_score'],max_books_per_ranking=20))
    sheets(selected)


def saved_masks(row):
    result={}
    for name,key,inverse in [('black','npj_black',True),('red','npj_red',True),('edit','npj_degradation',False)]:
        a=cv2.imread(str(ROOT/row[key]),0)
        if a is None: raise OSError(row[key])
        assert set(np.unique(a)).issubset({0,255}),row[key]
        result[name]=(a==0) if inverse else (a>0)
    return result


def sheets(selected):
    rows={r['page']:r for r in inventory()}
    font=ImageFont.truetype(FONT_PATH,19)
    for offset in range(0,len(selected),6):
        canvas=Image.new('RGB',(1500,1710),(250,248,243));d=ImageDraw.Draw(canvas)
        for i,choice in enumerate(selected[offset:offset+6]):
            row=rows[choice['page']]
            im=cv2.imread(str(ROOT/row['original']),cv2.IMREAD_REDUCED_COLOR_4)
            masks=saved_masks(row)
            masks={k:cv2.resize(v.astype(np.uint8),im.shape[1::-1],interpolation=cv2.INTER_NEAREST)>0 for k,v in masks.items()}
            painted=overlay(im,[('edit',masks['edit']),('blue',masks['black']),('red',masks['red'])])
            x=(i%2)*750;y=(i//2)*570
            d.text((x+12,y+8),f'{offset+i+1:02d} {row["page"]}',font=font,fill='black')
            d.text((x+12,y+34),choice['screening_reason'],font=font,fill='black')
            for j,img in enumerate([im,painted]):
                pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));pil.thumbnail((355,492))
                canvas.paste(pil,(x+12+j*370,y+66))
        path=OUT/'screening_sheets'/f'sheet_{offset//6+1:02d}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
        canvas.save(path,quality=92)
        print('Contact sheet:',path.relative_to(ROOT),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['screen','sheets'])
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='screen':screen()
    else:sheets(json.loads((OUT/'screening.json').read_text())['selection'])


if __name__=='__main__':main()
