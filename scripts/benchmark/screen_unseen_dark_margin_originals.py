"""Original-only broader dark-support screening after a no-op new-book check."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, sha, FONT_PATH

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/unseen_dark_margin_screening_20261001'
PRIOR=ROOT/'tmp_work/unseen_thin_margin_validation_20261001'
cv2.setNumThreads(1)


def score(row):
    im=cv2.imread(str(ROOT/row['original']),cv2.IMREAD_REDUCED_COLOR_4)
    assert im is not None
    H,W=im.shape[:2]
    lab=cv2.cvtColor(im.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    maximum=cv2.dilate(lab[:,:,0],np.ones((33,33),np.uint8))
    bright=(lab[:,:,0]>=maximum-8).astype(np.float32)
    mass=cv2.GaussianBlur(bright,(0,0),16)
    paper=cv2.GaussianBlur(lab*bright[:,:,None],(0,0),16)/np.maximum(mass[:,:,None],1e-6)
    fallback=cv2.GaussianBlur(lab,(0,0),16)
    paper=np.where((mass>1e-5)[:,:,None],paper,fallback)
    drop=paper[:,:,0]-lab[:,:,0]; da=lab[:,:,1]-paper[:,:,1]
    bw=((drop>=8)&(lab[:,:,1]<=6)).astype(np.uint8)
    opened=cv2.morphologyEx(bw,cv2.MORPH_OPEN,np.ones((1,max(3,round(.28*W))),np.uint8))
    n,_,stats,_=cv2.connectedComponentsWithStats(opened,connectivity=8)
    rules=[int(stats[i,1]) for i in range(1,n) if stats[i,2]>=.55*W and stats[i,3]<=.03*H and .075*H<=stats[i,1]<=.55*H]
    rule=min(rules) if rules else None
    margin=np.zeros((H,W),bool)
    if rule is not None:margin[round(.04*H):max(round(.04*H),rule-8),round(.06*W):round(.94*W)]=True
    dark=(drop>=8)&(lab[:,:,0]<=85)&margin
    # Screening only: quarter-resolution samples can break narrow lines; no closure/growth is shown or used as a method result.
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),connectivity=8)
    seed=margin&(da>=3)&(drop>=6)&(lab[:,:,1]>=1)&(lab[:,:,2]>=8)
    seed_counts=np.bincount(labels.ravel(),weights=seed.ravel(),minlength=n)
    distance=cv2.distanceTransform(dark.astype(np.uint8),cv2.DIST_L2,5)
    peak=np.zeros(n,np.float32);np.maximum.at(peak,labels.ravel(),distance.ravel())
    inner=cv2.erode(margin.astype(np.uint8),np.ones((3,3),np.uint8))>0
    boundary=np.bincount(labels.ravel(),weights=(dark&~inner).ravel(),minlength=n)
    candidates=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if 8<=area<=.01*H*W and 3<=max(w,h)<=.2*min(H,W) and boundary[i]==0:
            candidates.append(dict(xywh=[x,y,w,h],area=area,peak=float(peak[i]),warm_seeds=int(seed_counts[i]),
                                   kind='warm_supported' if seed_counts[i]>=2 else 'neutral_dark',
                                   score=float(area*(1 if seed_counts[i]>=2 else .35))))
    candidates.sort(key=lambda r:-r['score'])
    return dict(page=row['page'],book=row['book'],split=row['split'],original=row['original'],
                score=candidates[0]['score'] if candidates else 0,candidates=candidates[:10],
                size_reduced=[W,H],rule_y_reduced=rule,original_only=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((PRIOR/'baseline_review.json').read_text())
    excluded=set(previous['excluded_books'])|{r['book'] for r in previous['cases']};assert len(excluded)==33
    reject={p for p,s in previous['illustration_and_decorative_filter'].items() if s.startswith('excluded')}
    inventory_path=ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv'
    rows=[r for r in csv.DictReader(inventory_path.open()) if r['book'] not in excluded and r['page'] not in reject]
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(score,rows),1):
            results.append(r)
            if i%200==0:print('Original dark-margin screen:',i,'/',len(rows),flush=True)
    results.sort(key=lambda r:(-r['score'],r['page']))
    selected=[];count={}
    for r in results:
        if r['score'] and count.get(r['book'],0)<2:
            selected.append(r);count[r['book']]=count.get(r['book'],0)+1
    dependencies={str(p.relative_to(ROOT)):sha(p) for p in [inventory_path,PRIOR/'baseline_review.json']}
    dump(OUT/'screening.json',dict(source_sha256=sha(Path(__file__)),original_only=True,excluded_books=sorted(excluded),
        rejected_pages=sorted(reject),screened_pages=len(rows),candidates=selected,dependencies=dependencies,
        screening_parameters_only=True,no_v6_or_candidate_output_seen=True,scores_not_GT=True,visual_filter_required=True))
    dump(OUT/'scores.json',results);(OUT/'source_snapshot.py').write_text(Path(__file__).read_text())
    font=ImageFont.truetype(FONT_PATH,20)
    for start in range(0,len(selected),6):
        sheet=Image.new('RGB',(1500,1480),(250,248,243));draw=ImageDraw.Draw(sheet)
        top=Image.new('RGB',(1500,1040),(250,248,243));td=ImageDraw.Draw(top)
        for d in [draw,td]:d.text((14,8),'原画像だけの暗色候補探索。損傷・筆記の正解ではない。イラストは目視除外。',font=font,fill='black')
        for i,r in enumerate(selected[start:start+6]):
            bgr=cv2.imread(str(ROOT/r['original']),cv2.IMREAD_REDUCED_COLOR_4)
            im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));small=im.copy();small.thumbnail((476,625))
            x,y=(i%3)*500,50+(i//3)*700
            draw.text((x+12,y),r['page']+' · '+str(round(r['score'],1)),font=font,fill='black');sheet.paste(small,(x+12,y+35))
            tt=im.crop((0,0,im.width,round(im.height*.38)));tt.thumbnail((476,430))
            tx,ty=(i%3)*500,50+(i//3)*490
            td.text((tx+12,ty),r['page'],font=font,fill='black');top.paste(tt,(tx+12,ty+35))
        sheet.save(OUT/f'original_sheet_{start//6+1:02d}.png');top.save(OUT/f'original_top_{start//6+1:02d}.png')
    print('Original-only dark-support candidates:',len(selected),'pages',len(count),'books',flush=True)


if __name__=='__main__':main()
