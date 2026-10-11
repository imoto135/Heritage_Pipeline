"""Secondary ORIGINAL-only discovery, including margins, after false-fringe review.

Quarter-size red core cues are a search heuristic, never a restoration classifier.
"""
import json,time
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from synthetic_layers import OUT,ROOT,file_hash

def score(path):
    cv2.setNumThreads(1);p=Path(path);a=cv2.imread(str(p),cv2.IMREAD_REDUCED_COLOR_4)
    if a is None:return dict(original=str(p.relative_to(ROOT)),status='decode_failed')
    h,w=a.shape[:2];f=a.astype(np.float32)+1;B,G,R=np.moveaxis(f,2,0)
    cg=np.log(R/G);cb=np.log(R/B);rg=cg-cv2.GaussianBlur(cg,(0,0),6);rb=cb-cv2.GaussianBlur(cb,(0,0),6)
    hsv=cv2.cvtColor(a,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    core=(rg>.045)&(rb>.02)&(R>125)&(gray>.28)&(R>B)&((hsv[:,:,0]<28)|(hsv[:,:,0]>166))
    # Keep genuine margins; eliminate only the outermost image frame.
    core[:2]=False;core[-2:]=False;core[:,:2]=False;core[:,-2:]=False
    mean=cv2.boxFilter(gray,-1,(31,31));sd=np.sqrt(np.maximum(0,cv2.boxFilter(gray*gray,-1,(31,31))-mean*mean));black=gray<mean*(1+.2*(sd/.5-1))
    nearby=cv2.dilate(black.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))>0
    n,labels,stats,centers=cv2.connectedComponentsWithStats(core.astype(np.uint8),8);locations=[]
    for i,(s,c) in enumerate(zip(stats[1:],centers[1:]),1):
        x,y,ww,hh,area=map(int,s)
        if not 1<=area<=1000 or max(ww,hh)>100:continue
        support=labels[y:y+hh,x:x+ww]==i;strength=float(np.maximum(rg[y:y+hh,x:x+ww],0)[support].mean());adj=float(nearby[y:y+hh,x:x+ww][support].mean())
        value=np.sqrt(area)*strength*(1+2*adj)
        locations.append(dict(x=int(c[0]*4),y=int(c[1]*4),width=ww*4,height=hh*4,pixels_reduced=area,value=float(value),black_nearby_fraction=adj))
    locations.sort(key=lambda r:-r['value'])
    return dict(original=str(p.relative_to(ROOT)),id=p.stem,book=p.parent.name,status='scored',score=sum(r['value'] for r in locations[:30]),candidate_locations=locations[:10])

def main():
    start=time.monotonic();dest=OUT/'bright_annotation_screen';dest.mkdir(exist_ok=True)
    frozen=json.loads((OUT/'cases_frozen.json').read_text());splits=json.loads((OUT/'synthetic_audit/protocol.json').read_text())['splits']
    excluded=set(frozen['development_books']+frozen['validation_books']+['200010454'])|{b for rows in splits.values() for b in rows}
    paths=sorted(p for split in ['train','val','test'] for p in (ROOT/'data'/split).glob('*/*.jpg') if p.parent.name not in excluded)
    protocol=dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=file_hash(__file__),pages=len(paths),excluded_books=sorted(excluded),
        purpose='Secondary original-only bright chromatic cores; fix false black-edge candidates, include marginal notes. No restoration outputs or labels determine scores',
        parameters_fixed_before_scan=True,decode_factor=4,relative_RG_core=.045,relative_RB_core=.02,minimum_R=125,minimum_gray=.28,
        original_first_search='small_annotation_screen; native previews mostly black fringes, paper grain, intentional brown/blue lines',
        limitations='Faint or very small marks may be missed; no candidates does not prove absence of annotations')
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));checkpoint=dest/'per_page_checkpoint.jsonl';cached={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            try:r=json.loads(line);cached[r['original']]=r
            except json.JSONDecodeError:pass
    pending=[p for p in paths if str(p.relative_to(ROOT)) not in cached];rows=list(cached.values())
    with ProcessPoolExecutor(max_workers=8) as pool:
        with checkpoint.open('a') as f:
            for i,r in enumerate(pool.map(score,map(str,pending),chunksize=4),1):
                rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
                if i%200==0:print('Bright annotation scan',len(rows),'/',len(paths),round(time.monotonic()-start,1),flush=True)
    chosen=[]
    for book in sorted({r['book'] for r in rows if r['status']=='scored'}):
        chosen+=sorted([r for r in rows if r.get('book')==book and r['candidate_locations']],key=lambda r:-r['score'])[:4]
    chosen=sorted(chosen,key=lambda r:-r['score']);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    records=[]
    for offset in range(0,len(chosen),12):
        sheet=Image.new('RGB',(4*360,3*620),'#faf9f5');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(chosen[offset:offset+12]):
            im=Image.open(ROOT/r['original']).convert('RGB');w,h=im.size;c=r['candidate_locations'][0];xx=max(0,min(w-330,c['x']-165));yy=max(0,min(h-430,c['y']-215));x=j%4*360+5;y=j//4*620+5
            d.text((x,y),str(offset+j)+' '+r['id'],font=font,fill='#20313a');d.text((x,y+22),'原画像／まだ朱とは判定していない',font=font,fill='#536470');thumb=im.copy();thumb.thumbnail((150,135));sheet.paste(thumb,(x,y+52));sheet.paste(im.crop((xx,yy,xx+330,yy+430)),(x,y+190))
            records.append(dict(**r,preview_index=offset+j,preview_ROI=[xx,yy,330,430],source_sha256=file_hash(ROOT/r['original'])))
        sheet.save(dest/f'原画像_明るい朱候補_{offset//12+1:02d}.png')
    (dest/'queue.json').write_text(json.dumps(dict(records=records,source_sha256=file_hash(__file__),input_only=True,semantic_GT=False),ensure_ascii=False,indent=2))
    (dest/'complete.json').write_text(json.dumps(dict(pages=len(rows),candidate_pages=len(records),candidate_books=len({r['book'] for r in records}),seconds=time.monotonic()-start),indent=2));print('Bright scan complete',len(records),flush=True)

if __name__=='__main__':main()
