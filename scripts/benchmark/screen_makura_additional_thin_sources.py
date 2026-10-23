"""Source-only screen for additional thin-stroke annotation candidates.

Ranking proposes where to inspect; it does not create teacher labels.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt,label
from skimage.morphology import skeletonize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/makura_additional_thin_sources_20261021'
MANIFEST=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021/unet_v3/train_manifest.jsonl'
EXCLUDE={'200017197_00103','200017197_00214'}
ROI=128;STEP=32

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))

def proposals(raw):
    g=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
    paper=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    contrast=paper.astype(np.int16)-g.astype(np.int16)
    binary=contrast>=14
    skel=skeletonize(binary)
    radius=distance_transform_edt(binary)
    thin=skel&(radius<=2.5)&(contrast>=18)&(contrast<=105)
    components,n=label(thin,np.ones((3,3),bool))
    counts=np.bincount(components.ravel());long=np.where(counts>=9)[0];long=long[long>0]
    selected=np.isin(components,long)
    return selected,contrast

def main():
    assert not OUT.exists(),f'Preserve existing source screen: {OUT}'
    OUT.mkdir(parents=True)
    records=[json.loads(x) for x in MANIFEST.read_text().splitlines()]
    unique={r['patch_id']:r for r in records if r['damage_family']=='identity' and r['group'] not in EXCLUDE}
    groups={}
    for row in unique.values():
        raw=rgb(ROOT/row['clean']);mask,contrast=proposals(raw)
        for y in range(48,512-ROI-47,STEP):
            for x in range(48,512-ROI-47,STEP):
                q=mask[y:y+ROI,x:x+ROI]
                count=int(q.sum())
                if count<8:continue
                # Distinct long, moderately contrasted source structures; review is mandatory.
                score=count+int(((contrast[y:y+ROI,x:x+ROI]>=18)&(contrast[y:y+ROI,x:x+ROI]<=65)&q).sum())
                groups.setdefault(row['group'],[]).append(dict(group=row['group'],patch=row['patch_id'],
                    source=row['clean'],source_sha256=sha(ROOT/row['clean']),roi=[x,y,ROI,ROI],
                    thin_skeleton=count,screen_score=score))
    selected=[]
    for group,options in sorted(groups.items()):
        chosen=max(options,key=lambda q:(q['screen_score'],q['thin_skeleton'],q['patch']))
        selected.append(chosen)
    assert len(selected)==22,(len(selected),sorted(groups))
    selected.sort(key=lambda q:(-q['screen_score'],q['group']))
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    pages=[]
    for offset in range(0,len(selected),8):
        subset=selected[offset:offset+8];tile=256;labelh=54;gap=12;cols=4;rows=2
        canvas=Image.new('RGB',(cols*(tile+gap)+gap,rows*(tile+labelh+gap)+gap),'#faf9f5');d=ImageDraw.Draw(canvas)
        for i,row in enumerate(subset):
            x0,y0,w,h=row['roi'];raw=rgb(ROOT/row['source'])
            crop=raw[y0:y0+h,x0:x0+w]
            im=Image.fromarray(crop).resize((tile,tile),Image.Resampling.NEAREST)
            x=gap+(i%cols)*(tile+gap);y=gap+(i//cols)*(tile+labelh+gap)
            d.text((x,y),row['patch'],font=font,fill='#203442')
            d.text((x,y+21),f"score {row['screen_score']} / xy {x0},{y0}",font=font,fill='#203442')
            canvas.paste(im,(x,y+labelh))
        p=OUT/'contact_sheets'/(f'source_candidates_{offset//8+1:02d}.png');p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
        pages.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p),groups=[r['group'] for r in subset]))
    dump(OUT/'protocol.json',dict(purpose='Find source-only thin-stroke regions for manual review, not label generation',
        source='96 existing train clean 512 patches, excluding two already annotated source groups',
        heuristic='Morphological close41 background contrast; skeleton of contrast>=14; local radius<=2.5, contrast18..105; connected run>=9; top 128x128 window per source group',
        selection='One top-scoring 128 ROI per 22 groups; known group IDs not scored against method outputs',
        result='Candidate positions only; uninspected regions are not foreground labels',formal_gt=False))
    dump(OUT/'candidates.json',dict(rows=selected,contact_sheets=pages))
    print('Screened',len(unique),'patches; selected',len(selected),'distinct groups; contact sheets',len(pages),flush=True)
    for row in selected[:10]:print(row['patch'],row['roi'],row['screen_score'],flush=True)

if __name__=='__main__':main()
