"""Input-only relative-red ranking; a biased search aid, never a classifier GT."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json,os,time
from datetime import datetime,timezone
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'relative_red_screen'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def score(path):
    cv2.setNumThreads(1)
    p=Path(path)
    bgr=cv2.imread(str(p),cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None:return dict(original=str(p.relative_to(ROOT)),status='decode_failed')
    h,w=bgr.shape[:2]
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    B,G,R=np.moveaxis(bgr.astype('float32')+1,2,0)
    c=np.log(R/G)
    local=cv2.GaussianBlur(c,(0,0),8,borderType=cv2.BORDER_REFLECT_101)
    band=c-cv2.GaussianBlur(c,(0,0),1.4,borderType=cv2.BORDER_REFLECT_101)
    mask=(c-local>.012)&(band>.004)&(R>B)&((hsv[:,:,0]<32)|(hsv[:,:,0]>164))&(hsv[:,:,2]>30)
    sl=np.s_[round(h*.07):round(h*.93),round(w*.07):round(w*.93)]
    inner=mask[sl]
    sat=(hsv[:,:,1]>100)&(hsv[:,:,0]<30)&(R>G*1.18)
    coverage=float(sat[sl].mean())
    n,labels,stats,_=cv2.connectedComponentsWithStats(inner.astype('uint8'),8)
    areas=stats[1:,cv2.CC_STAT_AREA]
    sizes=np.maximum(stats[1:,cv2.CC_STAT_WIDTH],stats[1:,cv2.CC_STAT_HEIGHT])
    components=(areas>=3)&(sizes>=5)
    evidence=float(np.maximum(c-local,0)[sl][inner].sum())
    value=float(areas[components].sum())+20*evidence
    if coverage>.18:value*=.02
    return dict(original=str(p.relative_to(ROOT)),id=p.stem,book=p.parent.name,split=p.parent.parent.name,
        status='scored',score=value,relative_red_pixels=int(inner.sum()),line_component_pixels=int(areas[components].sum()),
        saturated_fraction=coverage,reduced_size=[w,h])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    paths=sorted(p for split in ['train','val','test'] for p in (ROOT/'data'/split).glob('*/*.jpg'))
    manifest=dict(purpose='Input-only discovery of faint relative-red writing; not restoration selection or semantic annotations',
        reduced_decode_factor=4,local_sigma=8,band_sigma=1.4,relative_log_R_G=.012,band_log_R_G=.004,
        cover_penalty_saturated_fraction=.18,inner_margin=.07,selection='top 3 per book; up to 120 overall; keep pages apart',
        positives_biased=True,output_driven_selection=False,scan_originals=len(paths),
        created=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,row in enumerate(pool.map(score,map(str,paths),chunksize=12),1):
            rows.append(row)
            if i%500==0:print('Input scan:',i,'/',len(paths),'seconds',round(time.monotonic()-start,1),flush=True)
    (OUT/'scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    old=json.loads((ROOT/'tmp_work/red_positive_collection_20260930/selection.json').read_text())
    excluded={c['book'] for c in old['cases']}
    chosen=[]
    for book in sorted({r['book'] for r in rows if r['status']=='scored'}-excluded):
        candidates=sorted([r for r in rows if r.get('book')==book and r.get('status')=='scored'],key=lambda r:-r['score'])
        selected=[]
        pageorder={r['id']:i for i,r in enumerate(sorted(candidates,key=lambda r:r['id']))}
        for r in candidates:
            if r['score']<30:break
            if any(abs(pageorder[r['id']]-pageorder[a['id']])<3 for a in selected):continue
            selected.append(r)
            if len(selected)>=3:break
        chosen.extend(selected)
    chosen=sorted(chosen,key=lambda r:(-r['score'],r['id']))[:120]
    for i,r in enumerate(chosen):r['queue_index']=i
    (OUT/'queue.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
    for offset in range(0,len(chosen),16):
        subset=chosen[offset:offset+16]
        sheet=Image.new('RGB',(1840,2680),'#faf9f5');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(subset):
            a=cv2.imread(str(ROOT/r['original']),cv2.IMREAD_REDUCED_COLOR_2)
            im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((440,600),Image.Resampling.LANCZOS)
            x=(j%4)*460+10;y=(j//4)*670+10
            d.text((x,y),str(r['queue_index'])+' '+r['id'],font=ImageFont.truetype(FONT,16),fill='#20313b')
            d.text((x,y+24),'入力のみ score='+str(round(r['score'])),font=ImageFont.truetype(FONT,16),fill='#20313b')
            sheet.paste(im,(x,y+52))
        sheet.save(OUT/f'候補_{offset//16+1:02d}.png')
    summary=dict(pages=len(rows),candidate_pages=len(chosen),candidate_books=len({r['book'] for r in chosen}),
        excluded_development_books=sorted(excluded),seconds=time.monotonic()-start)
    (OUT/'complete.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(summary,flush=True)

if __name__=='__main__':main()
