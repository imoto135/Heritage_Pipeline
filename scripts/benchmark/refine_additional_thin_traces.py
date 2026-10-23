"""Snap hand-placed source traces onto nearby source-RGB darkness ridges.

This is an inspectable candidate; only the original RGB affects snapping.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from review_additional_thin_traces import points

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/makura_additional_thin_sources_20261021'
REJECT={'200017197_00344_0_p02':'Draft trace misses a very faint tail; source ridge not reliably located'}
OFFSETS=np.arange(-5,6)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snap(raw,vertices):
    rough=points(vertices);g=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
    paper=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    contrast=paper.astype(np.int16)-g.astype(np.int16)
    n=len(rough);candidate=np.zeros((n,len(OFFSETS),2),int);reward=np.zeros((n,len(OFFSETS)),float)
    for i,(x,y) in enumerate(rough):
        before=rough[max(0,i-2)];after=rough[min(n-1,i+2)]
        tangent=(after-before).astype(float);norm=np.linalg.norm(tangent)
        if norm<1:tangent=np.array([1.,0.]);norm=1.
        normal=np.array([-tangent[1],tangent[0]])/norm
        for j,o in enumerate(OFFSETS):
            q=np.rint([x,y]+o*normal).astype(int);qx,qy=map(int,q)
            qx=int(np.clip(qx,0,511));qy=int(np.clip(qy,0,511));candidate[i,j]=[qx,qy]
            reward[i,j]=.2*contrast[qy,qx]-.4*abs(o)
    score=reward[0].copy();back=[]
    for i in range(1,n):
        transition=score[:,None]-1.0*np.abs(OFFSETS[:,None]-OFFSETS[None,:])
        arg=np.argmax(transition,axis=0);score=reward[i]+transition[arg,np.arange(len(OFFSETS))]
        back.append(arg)
    route=np.empty(n,int);route[-1]=int(np.argmax(score))
    for i in range(n-2,-1,-1):route[i]=back[i][route[i+1]]
    chosen=candidate[np.arange(n),route]
    refined=points(chosen.tolist())
    x,y=refined.T
    return refined,dict(n_rough=n,n_refined=len(refined),mean_abs_normal_offset=float(np.mean(np.abs(OFFSETS[route]))),
        max_abs_normal_offset=int(np.max(np.abs(OFFSETS[route]))),
        source_contrast_min=int(contrast[y,x].min()),source_contrast_median=float(np.median(contrast[y,x])),
        source_contrast_ge12=int((contrast[y,x]>=12).sum()),source_contrast_ge20=int((contrast[y,x]>=20).sum()))

def main():
    draft=json.loads((OUT/'new_thin_paths_draft.json').read_text())['paths']
    candidates={r['patch']:r for r in json.loads((OUT/'candidates.json').read_text())['rows']}
    rows=[];audit=[];font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    for item in draft:
        patch=item['patch']
        if patch in REJECT:
            audit.append(dict(patch=patch,status='rejected',reason=REJECT[patch]));continue
        row=candidates[patch];source=ROOT/row['source'];assert sha(source)==row['source_sha256']
        raw=np.asarray(Image.open(source).convert('RGB'));refined,stats=snap(raw,item['xy'])
        entry=dict(patch=patch,id=item['id'],confidence='candidate_visible_source',xy=refined.tolist(),
            rough_vertices=item['xy'],source_sha256=sha(source))
        rows.append(entry);audit.append(dict(patch=patch,status='candidate',**stats))
        over=raw.copy();rough=points(item['xy']);over[rough[:,1],rough[:,0]]=[255,50,40]
        over[refined[:,1],refined[:,0]]=[0,100,255]
        x0,y0,w,h=row['roi'];ww=4*w;hh=4*h;gap=16;head=54
        canvas=Image.new('RGB',(2*(ww+gap)+gap,head+hh+gap),'#faf9f5');d=ImageDraw.Draw(canvas)
        d.text((gap,8),patch+' / 赤=仮置き 青=原本の濃淡で補正',font=font,fill='#203442')
        for i,a in enumerate((raw[y0:y0+h,x0:x0+w],over[y0:y0+h,x0:x0+w])):
            X=gap+i*(ww+gap);canvas.paste(Image.fromarray(a).resize((ww,hh),Image.Resampling.NEAREST),(X,head))
        p=OUT/'refined_trace_review'/(patch+'.png');p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
    path=OUT/'new_thin_paths_candidate.json';path.write_text(json.dumps(dict(
        status='Source-only review candidates; not expert GT or final teacher',
        rejected=REJECT,paths=rows),ensure_ascii=False,indent=2)+'\n')
    (OUT/'refined_trace_audit.json').write_text(json.dumps(dict(annotation_sha256=sha(path),rows=audit),ensure_ascii=False,indent=2)+'\n')
    for a in audit:print(a['patch'],a['status'],a.get('n_refined'),a.get('source_contrast_median'),a.get('source_contrast_ge12'),flush=True)

if __name__=='__main__':main()
