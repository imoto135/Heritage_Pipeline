"""Propose conservative visible ink cores around source-traced thin lines.

Source RGB and source-only traces are the only evidence. This is a candidate
partial label; no method prediction or synthetic input influences placement.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_dilation,distance_transform_edt,label

ROOT=Path(__file__).resolve().parents[2]
STRESS=ROOT/'tmp_work/makura_damage_challenge_20261021'
OUT=ROOT/'tmp_work/makura_thin_width_candidate_20261021'
PATCHES=('200017197_00103_1_p00','200017197_00214_1_p00','200017197_00172_0_p00')
BOXES={'200017197_00103_1_p00':(95,365,155,420),
       '200017197_00214_1_p00':(125,30,160,90),
       '200017197_00172_0_p00':(240,125,390,250)}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def load_mask(p):return np.asarray(Image.open(p).convert('L'))>0

def make(raw,trace):
    g=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
    paper=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    contrast=paper.astype(np.int16)-g.astype(np.int16)
    yy,xx=np.ogrid[-2:3,-2:3];corridor=binary_dilation(trace,structure=xx*xx+yy*yy<=4)
    _,indices=distance_transform_edt(~trace,return_indices=True)
    near=contrast[tuple(indices)]
    threshold=np.maximum(12,np.ceil(.35*near)).astype(np.int16)
    proposal=corridor&(contrast>=threshold)
    proposal|=trace
    comps,n=label(proposal,np.ones((3,3),bool))
    seeded=np.unique(comps[trace]);seeded=seeded[seeded>0]
    core=np.isin(comps,seeded)&corridor
    assert np.all(core[trace])
    return core,corridor,contrast,threshold

def figure(path,patch,raw,trace,core,corridor):
    x0,y0,x1,y1=BOXES[patch];scale=5
    q=raw[y0:y1,x0:x1]
    over=raw.copy();over[core&~trace]=[0,170,255];over[trace]=[255,30,50]
    candidate=np.full_like(raw,255);candidate[core]=0
    unknown=np.full_like(raw,255);unknown[corridor&~core]=[225,70,155];unknown[core]=[0,0,0]
    tiles=[q,over[y0:y1,x0:x1],candidate[y0:y1,x0:x1],unknown[y0:y1,x0:x1]]
    w=(x1-x0)*scale;h=(y1-y0)*scale;gap=14;head=66;lab=26
    canvas=Image.new('RGB',(4*(w+gap)+gap,head+h+lab+gap),'#faf9f5')
    d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    d.text((gap,8),patch+' / 原本RGBだけから求めた見える墨芯の候補',font=font,fill='#203442')
    for i,(tile,name) in enumerate(zip(tiles,('原本','赤=中心 青=幅候補','墨芯候補','桃=未確定'))):
        x=gap+i*(w+gap);d.text((x,head-27),name,font=font,fill='#203442')
        im=Image.fromarray(tile).resize((w,h),Image.Resampling.NEAREST);canvas.paste(im,(x,head))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def main():
    assert not OUT.exists(),f'Candidate already exists: {OUT}'
    OUT.mkdir(parents=True)
    selection={r['patch_id']:r for r in json.loads((STRESS/'protocol.json').read_text())['selections']}
    rows=[]
    for patch in PATCHES:
        source=ROOT/selection[patch]['original'];assert sha(source)==selection[patch]['original_sha256']
        raw=rgb(source);trace_path=STRESS/'source_thin_review'/(patch+'_high_centers.png');trace=load_mask(trace_path)
        core,corridor,contrast,threshold=make(raw,trace)
        folder=OUT/'masks';folder.mkdir(exist_ok=True)
        core_path=folder/(patch+'_core.png');cor_path=folder/(patch+'_corridor.png')
        Image.fromarray(core.astype(np.uint8)*255).save(core_path)
        Image.fromarray(corridor.astype(np.uint8)*255).save(cor_path)
        path=OUT/'figures'/(patch+'_source_core.png');figure(path,patch,raw,trace,core,corridor)
        rows.append(dict(patch=patch,source=selection[patch]['original'],source_sha256=sha(source),
            trace_sha256=sha(trace_path),trace_n=int(trace.sum()),corridor_n=int(corridor.sum()),
            core_n=int(core.sum()),new_width_pixels=int((core&~trace).sum()),unknown_width_pixels=int((corridor&~core).sum()),
            contrast_min=int(contrast[core].min()),contrast_median=float(np.median(contrast[core])),
            threshold_min=int(threshold[core].min()),threshold_max=int(threshold[core].max()),
            core_sha256=sha(core_path),corridor_sha256=sha(cor_path),figure_sha256=sha(path)))
    dump(OUT/'protocol.json',dict(source='original RGB and previously source-traced centers only',
        candidate='within Euclidean radius2 of source trace; grayscale contrast to close41 paper estimate >= max(12, ceil(.35*nearest-center contrast)); retain only connected components touching source traces',
        negative_label='none; pixels outside proposed core remain unchanged or unknown',
        proposal_not_expert_verified=True,formal_gt=False,
        no_predictions_or_damaged_inputs_read=True))
    dump(OUT/'audit.json',dict(rows=rows))
    for r in rows:print(r['patch'],'core',r['core_n'],'trace',r['trace_n'],'new',r['new_width_pixels'],'unknown corridor',r['unknown_width_pixels'],flush=True)

if __name__=='__main__':main()
