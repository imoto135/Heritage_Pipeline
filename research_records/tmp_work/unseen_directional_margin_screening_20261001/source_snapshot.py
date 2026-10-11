"""Original-only screen of all remaining book groups, no mask/output access."""
import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from build_npj_v6_comparison import dump,sha,FONT_PATH
import screen_unseen_dark_margin_originals as previous

ROOT=previous.ROOT
OUT=ROOT/'tmp_work/unseen_directional_margin_screening_20261001'
cv2.setNumThreads(1)


def score(row):
    result=previous.score(row)
    image=cv2.imread(str(ROOT/row['original']),cv2.IMREAD_REDUCED_COLOR_4)
    H,W=image.shape[:2];L=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
    # This ranking is not the frozen method. Full-width upper original crops also
    # expose missing rules, writing, illustrations, and new unanchored patterns.
    y0,y1=4,max(5,round(.22*H));x0,x1=round(.06*W),round(.94*W)
    smooth=cv2.GaussianBlur(L,(7,7),1)
    gx=np.clip(cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8),-5,5)[y0:y1,x0:x1]
    gy=np.clip(cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8),-5,5)[y0:y1,x0:x1]
    tensor=np.array([[np.mean(gx*gx),np.mean(gx*gy)],[np.mean(gx*gy),np.mean(gy*gy)]])
    vals=np.linalg.eigvalsh(tensor);energy=float(vals.sum());anisotropy=float((vals[1]-vals[0])/max(energy,1e-6))
    result.update(warm_score=max([c['area'] for c in result['candidates'] if c['warm_seeds']>=2]+[0]),
                  neutral_score=max([c['area'] for c in result['candidates'] if c['warm_seeds']<2]+[0]),
                  upper_direction_score=energy*anisotropy,upper_direction_energy=energy,upper_direction_anisotropy=anisotropy)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selection=ROOT/'tmp_work/unseen_dark_margin_validation_20261001/baseline_review.json'
    old=json.loads(selection.read_text());excluded=set(old['excluded_books'])|{c['book'] for c in old['cases']}
    assert len(excluded)==36
    inventory=ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv'
    rows=[r for r in csv.DictReader(inventory.open()) if r['book'] not in excluded]
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,r in enumerate(pool.map(score,rows),1):
            results.append(r)
            if i%100==0:print('Original directional screen:',i,'/',len(rows),flush=True)
    books=sorted({r['book'] for r in rows});assert len(books)==8 and len(rows)==1041
    selected=[]
    for book in books:
        chosen=set();group=[r for r in results if r['book']==book]
        for metric in ['warm_score','neutral_score','upper_direction_score']:
            ranked=sorted(group,key=lambda r:(-r[metric],r['page']))
            picked=0
            for r in ranked:
                if r['page'] in chosen:continue
                selected.append({**r,'ranking_metric':metric});chosen.add(r['page']);picked+=1
                if picked==2:break
    deps={str(p.relative_to(ROOT)):sha(p) for p in [selection,inventory,Path(previous.__file__)]}
    dump(OUT/'screening.json',dict(source_sha256=sha(Path(__file__)),dependencies=deps,excluded_books=sorted(excluded),
        screened_pages=1041,remaining_book_groups=books,candidates=selected,original_only=True,
        masks_or_restoration_outputs_read=False,visual_illustration_filter_required=True,ranking_not_GT=True,
        original_sha256={r['original']:sha(ROOT/r['original']) for r in selected}))
    dump(OUT/'scores.json',results);(OUT/'source_snapshot.py').write_text(Path(__file__).read_text())
    font=ImageFont.truetype(FONT_PATH,18)
    for j,book in enumerate(books,1):
        subset=[r for r in selected if r['book']==book]
        full=Image.new('RGB',(1500,1490),(250,248,243));top=Image.new('RGB',(1500,1090),(250,248,243))
        for sheet,is_top in [(full,False),(top,True)]:
            draw=ImageDraw.Draw(sheet);draw.text((12,8),book+' · 未使用冊の原画像のみ。イラストを目視除外。順位は損傷の正解ではない。',font=font,fill='black')
            for i,r in enumerate(subset):
                image=cv2.imread(str(ROOT/r['original']),cv2.IMREAD_REDUCED_COLOR_4)
                pil=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                if is_top:pil=pil.crop((0,0,pil.width,round(.32*pil.height)))
                pil.thumbnail((475,445 if is_top else 635))
                x,y=(i%3)*500,50+(i//3)*(515 if is_top else 710)
                draw.text((x+10,y),r['page']+' / '+r['ranking_metric'],font=font,fill='black');sheet.paste(pil,(x+10,y+30))
        full.save(OUT/f'original_sheet_{j:02d}.png');top.save(OUT/f'original_top_{j:02d}.png')
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='complete.json'}))
    print('Complete:48 original candidates /8 unused book groups',flush=True)


if __name__=='__main__':main()
