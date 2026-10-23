"""Source-only conservative core proposals for five reviewed training traces."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
SCREEN=ROOT/'tmp_work/makura_additional_thin_sources_20261021'
OUT=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from propose_makura_thin_width import make

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))

def main():
    assert not OUT.exists(),f'Preserve existing candidates: {OUT}'
    OUT.mkdir(parents=True)
    paths=json.loads((SCREEN/'new_thin_paths_candidate.json').read_text())['paths']
    sources={r['patch']:r for r in json.loads((SCREEN/'candidates.json').read_text())['rows']}
    assert len(paths)==5 and len({p['patch'] for p in paths})==5
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    rows=[]
    for item in paths:
        patch=item['patch'];source=sources[patch];sp=ROOT/source['source'];assert sha(sp)==source['source_sha256']
        raw=rgb(sp);points=np.asarray(item['xy']);x,y=points.T
        trace=np.zeros((512,512),bool);trace[y,x]=True
        core,corridor,contrast,threshold=make(raw,trace)
        folder=OUT/'masks';folder.mkdir(parents=True,exist_ok=True)
        paths_saved={}
        for name,a in [('trace',trace),('core',core),('corridor',corridor)]:
            p=folder/(patch+'_'+name+'.png');Image.fromarray(a.astype(np.uint8)*255).save(p);paths_saved[name]=dict(path=str(p.relative_to(OUT)),sha256=sha(p))
        x0,y0,w,h=source['roi'];scale=4;ww=w*scale;hh=h*scale;gap=14;head=58
        overlay=raw.copy();overlay[core&~trace]=[0,170,255];overlay[trace]=[255,40,50]
        core_img=np.full_like(raw,255);core_img[core]=0;core_img[corridor&~core]=[225,70,155]
        tiles=[raw[y0:y0+h,x0:x0+w],overlay[y0:y0+h,x0:x0+w],core_img[y0:y0+h,x0:x0+w]]
        canvas=Image.new('RGB',(3*(ww+gap)+gap,head+hh+gap),'#faf9f5');d=ImageDraw.Draw(canvas)
        d.text((gap,8),patch+' / 原本で確認した細線の墨芯候補',font=font,fill='#203442')
        for i,(tile,title) in enumerate(zip(tiles,('原本','赤=中心 青=墨芯追加','墨芯=黒 未確定=桃'))):
            X=gap+i*(ww+gap);d.text((X,head-23),title,font=font,fill='#203442')
            canvas.paste(Image.fromarray(tile).resize((ww,hh),Image.Resampling.NEAREST),(X,head))
        f=OUT/'figures'/(patch+'_source_core.png');f.parent.mkdir(parents=True,exist_ok=True);canvas.save(f)
        rows.append(dict(patch=patch,group=source['group'],source=source['source'],source_sha256=sha(sp),
            trace_n=int(trace.sum()),core_n=int(core.sum()),new_width_n=int((core&~trace).sum()),
            corridor_n=int(corridor.sum()),uncertain_n=int((corridor&~core).sum()),
            trace_contrast_min=int(contrast[trace].min()),trace_contrast_median=float(np.median(contrast[trace])),
            core_contrast_min=int(contrast[core].min()),core_contrast_median=float(np.median(contrast[core])),
            masks=paths_saved,figure=str(f.relative_to(OUT)),figure_sha256=sha(f)))
    dump(OUT/'protocol.json',dict(source='Five source-only reviewed traces on distinct additional training groups',
        annotation_sha256=sha(SCREEN/'new_thin_paths_candidate.json'),
        core_rule='Same conservative source-only rule as makura_thin_width_candidate_20261021',
        core_rule_sha256=sha(ROOT/'scripts/benchmark/propose_makura_thin_width.py'),
        rejected_draft='00344 hairline draft was offset and excluded; no teacher label created',
        labels_not_yet_applied=True,formal_gt=False,expert_verified=False))
    dump(OUT/'audit.json',dict(rows=rows))
    for r in rows:print(r['patch'],'trace',r['trace_n'],'core',r['core_n'],'min contrast',r['trace_contrast_min'],'uncertain',r['uncertain_n'],flush=True)

if __name__=='__main__':main()
