"""Source-only draft trace review; writes no training labels."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/makura_additional_thin_sources_20261021'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def points(vertices):
    out=[]
    for a,b in zip(vertices[:-1],vertices[1:]):
        a=np.asarray(a);b=np.asarray(b);n=int(np.abs(b-a).max())
        for p in np.rint(np.linspace(a,b,n+1)).astype(int):
            q=tuple(p)
            if not out or out[-1]!=q:out.append(q)
    return np.asarray(out)

def main():
    draft=OUT/'new_thin_paths_draft.json';records=json.loads(draft.read_text())['paths']
    candidates={r['patch']:r for r in json.loads((OUT/'candidates.json').read_text())['rows']}
    audits=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    for record in records:
        patch=record['patch'];candidate=candidates[patch];source=ROOT/candidate['source']
        assert sha(source)==candidate['source_sha256']
        raw=np.asarray(Image.open(source).convert('RGB'));x0,y0,w,h=candidate['roi']
        pts=points(record['xy']);x,y=pts.T
        assert np.all((x>=x0)&(x<x0+w)&(y>=y0)&(y<y0+h))
        g=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY);paper=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
        contrast=paper.astype(np.int16)-g.astype(np.int16)
        over=raw.copy();over[y,x]=[0,100,255]
        crop=raw[y0:y0+h,x0:x0+w];crop_over=over[y0:y0+h,x0:x0+w]
        scale=4;ww=w*scale;hh=h*scale;gap=16;head=54
        canvas=Image.new('RGB',(2*(ww+gap)+gap,head+hh+gap),'#faf9f5');d=ImageDraw.Draw(canvas)
        d.text((gap,8),patch+' / '+record['id']+' / 中心位置の仮置き',font=font,fill='#203442')
        for i,(a,name) in enumerate(((crop,'原本'),(crop_over,'青=仮中心'))):
            X=gap+i*(ww+gap);d.text((X,head-24),name,font=font,fill='#203442')
            im=Image.fromarray(a).resize((ww,hh),Image.Resampling.NEAREST);canvas.paste(im,(X,head))
        p=OUT/'draft_trace_review'/(patch+'.png');p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
        audits.append(dict(patch=patch,id=record['id'],source_sha256=sha(source),n=len(x),
            contrast_min=int(contrast[y,x].min()),contrast_median=float(np.median(contrast[y,x])),
            contrast_ge12=int((contrast[y,x]>=12).sum()),contrast_ge20=int((contrast[y,x]>=20).sum()),
            figure=str(p.relative_to(OUT)),figure_sha256=sha(p)))
    (OUT/'draft_trace_audit.json').write_text(json.dumps(dict(annotation_sha256=sha(draft),rows=audits),ensure_ascii=False,indent=2)+'\n')
    for row in audits:print(row['patch'],row['n'],row['contrast_min'],row['contrast_median'],row['contrast_ge12'],flush=True)

if __name__=='__main__':main()
