"""Input-only search for small red marks near ordinary black text.

Not a red classifier benchmark. Discovery ranks cannot be restoration metrics.
"""
import json,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

def score(path):
    cv2.setNumThreads(1);p=Path(path);a=cv2.imread(str(p),cv2.IMREAD_REDUCED_COLOR_2)
    if a is None:return dict(original=str(p.relative_to(ROOT)),status='decode_failed')
    h,w=a.shape[:2];hsv=cv2.cvtColor(a,cv2.COLOR_BGR2HSV);B,G,R=np.moveaxis(a.astype(np.float32)+1,2,0)
    chroma=np.log(R/G);reference=cv2.GaussianBlur(chroma,(0,0),10);band=chroma-cv2.GaussianBlur(chroma,(0,0),1.1)
    red=(chroma-reference>.014)&(band>.004)&(R>B)&((hsv[:,:,0]<30)|(hsv[:,:,0]>165))&(hsv[:,:,2]>40)
    gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;mean=cv2.boxFilter(gray,-1,(41,41));sd=np.sqrt(np.maximum(0,cv2.boxFilter(gray*gray,-1,(41,41))-mean*mean))
    black=gray<mean*(1+.2*(sd/.5-1));black&=(np.abs(chroma-reference)<.06)
    nearby=cv2.dilate(black.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))>0
    density=cv2.boxFilter(black.astype(np.float32),-1,(71,71));red&=nearby&(density>.012)
    red[:round(h*.12)]=False;red[round(h*.94):]=False;red[:,:round(w*.07)]=False;red[:,round(w*.93):]=False
    n,labels,stats,centers=cv2.connectedComponentsWithStats(red.astype(np.uint8),8)
    candidates=[]
    for s,c in zip(stats[1:],centers[1:]):
        x,y,ww,hh,area=map(int,s);long=max(ww,hh);short=max(1,min(ww,hh))
        if not 3<=area<=400 or not 3<=long<=70:continue
        if area>100 and long/short<1.8:continue
        xx,yy=map(int,c);strength=float((chroma-reference)[y:y+hh,x:x+ww][red[y:y+hh,x:x+ww]].max())
        value=(min(long/short,10)+1)*np.sqrt(area)*min(strength,.15)
        candidates.append(dict(x=xx*2,y=yy*2,width=ww*2,height=hh*2,pixels_reduced=area,value=float(value)))
    candidates.sort(key=lambda r:-r['value']);sat=float(((hsv[:,:,1]>150)&(hsv[:,:,0]<30)).mean())
    value=sum(r['value'] for r in candidates[:60])
    if sat>.12:value*=.03
    return dict(original=str(p.relative_to(ROOT)),id=p.stem,book=p.parent.name,split=p.parent.parent.name,status='scored',
        score=float(value),small_components=len(candidates),candidate_locations=candidates[:12],saturated_fraction=sat)

def main():
    dest=OUT/'small_annotation_screen';dest.mkdir(exist_ok=True);start=time.monotonic()
    protocol=json.loads((OUT/'cases_frozen.json').read_text());excluded=set(protocol['development_books']+protocol['validation_books']+['200010454'])
    # Entire synthetic-training books excluded from prospective native examples.
    excluded.update(json.loads((OUT/'synthetic_audit/protocol.json').read_text())['splits']['train'])
    paths=sorted(p for split in ['train','val','test'] for p in (ROOT/'data'/split).glob('*/*.jpg') if p.parent.name not in excluded)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),scan_pages=len(paths),excluded_books=sorted(excluded),
        purpose='Original-only discovery of small red notes/punctuation near black text; not semantic labels or model-output selection',
        reduced_decode_factor=2,source_sha256=file_hash(__file__),parameters_fixed_before_scan=True)
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(score,map(str,paths),chunksize=8),1):
            rows.append(r)
            if i%250==0:print('Small-note scan',i,'/',len(paths),round(time.monotonic()-start,1),'sec',flush=True)
    (dest/'scores.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    chosen=[]
    for book in sorted({r.get('book') for r in rows if r['status']=='scored'}):
        pool=sorted([r for r in rows if r.get('book')==book],key=lambda r:-r['score'])
        chosen.extend([r for r in pool[:3] if r['score']>=2])
    chosen=sorted(chosen,key=lambda r:-r['score'])[:96]
    for i,r in enumerate(chosen):r['queue_index']=i
    (dest/'queue.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for off in range(0,len(chosen),16):
        sheet=Image.new('RGB',(4*420,4*600),'#faf9f5');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(chosen[off:off+16]):
            im=Image.open(ROOT/r['original']).convert('RGB');im.thumbnail((405,535));x=j%4*420+6;y=j//4*600+6
            d.text((x,y),str(r['queue_index'])+' '+r['id'],font=font,fill='#20313a');d.text((x,y+23),'原画像のみ・小朱候補探索',font=font,fill='#20313a');sheet.paste(im,(x,y+50))
        sheet.save(dest/f'原画像候補_{off//16+1:02d}.png')
    (dest/'complete.json').write_text(json.dumps(dict(pages=len(rows),candidate_pages=len(chosen),candidate_books=len({r['book'] for r in chosen}),seconds=time.monotonic()-start),indent=2));print('Small annotation scan complete',len(chosen),flush=True)

if __name__=='__main__':main()
