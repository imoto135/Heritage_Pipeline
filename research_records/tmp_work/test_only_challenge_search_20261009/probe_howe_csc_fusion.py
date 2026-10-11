"""Development ablation: CSC-degradation veto vs conservative component removal.

Use one fixed rule for all six native pages. Removed components are candidates,
not annotated dirt. Preserve the original files and make reviewable PNGs.
"""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'howe_csc_fusion'
ROIS = {'01': (570,600,750,850), '02': (700,1200,850,850),
        '03': (750,700,850,850), '04': (550,160,800,820),
        '06': (80,95,620,530), '07': (800,750,850,900)}
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
RULE = dict(stroke_width='2 * median foreground distance transform',
            support_min_area_stroke_squared=16, support_min_csc_fraction=.1,
            degradation_max_area_stroke_squared=8, degradation_min_fraction=.6,
            degradation_max_csc_fraction=.02, degradation_min_distance_strokes=2,
            isolated_max_area_stroke_squared=16, isolated_min_fill=.4,
            isolated_aspect_range=[.5,2], isolated_min_distance_strokes=12)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def load(path,foreground_black):
    a=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    assert a is not None,path
    assert np.isin(a,[0,255]).all(),path
    return a<127 if foreground_black else a>127

def save(path,mask):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),np.where(mask,0,255).astype(np.uint8))
    assert np.array_equal(load(path,True),mask)

def components(howe,csc,deg,red):
    count,labels,stats,centers=cv2.connectedComponentsWithStats(howe.astype(np.uint8),8)
    area=stats[:,cv2.CC_STAT_AREA].astype(float)
    csc_n=np.bincount(labels[csc].ravel(),minlength=count)
    deg_n=np.bincount(labels[deg].ravel(),minlength=count)
    red_n=np.bincount(labels[red].ravel(),minlength=count)
    csc_frac=csc_n/np.maximum(area,1)
    deg_frac=deg_n/np.maximum(area,1)
    dist=cv2.distanceTransform(howe.astype(np.uint8),cv2.DIST_L2,5)
    sw=max(2.,float(2*np.median(dist[howe])))
    support=(area>=16*sw*sw)&(csc_frac>=.1)
    support[0]=False
    support_distance=cv2.distanceTransform((~support[labels]).astype(np.uint8),cv2.DIST_L2,5)
    nearest=np.full(count,np.inf,dtype=float)
    np.minimum.at(nearest,labels.ravel(),support_distance.ravel())
    width=stats[:,cv2.CC_STAT_WIDTH].astype(float)
    height=stats[:,cv2.CC_STAT_HEIGHT].astype(float)
    fill=area/np.maximum(width*height,1)
    aspect=width/np.maximum(height,1)
    degraded=(area<=8*sw*sw)&(deg_frac>=.6)&(csc_frac<=.02)&(nearest>=2*sw)
    isolated=(area<=16*sw*sw)&(fill>=.4)&(aspect>=.5)&(aspect<=2)&(nearest>=12*sw)
    remove=(degraded|isolated)&(red_n==0)
    remove[0]=False
    removed=remove[labels]
    rows=[]
    for i in np.flatnonzero(remove):
        rows.append(dict(component=int(i),xywh=stats[i,:4].tolist(),area=int(area[i]),
            csc_pixels=int(csc_n[i]),deg_pixels=int(deg_n[i]),csc_fraction=float(csc_frac[i]),
            deg_fraction=float(deg_frac[i]),nearest_support=float(nearest[i]),
            reason=('csc_degradation' if degraded[i] else 'isolated_round_mark')))
    return howe&~removed,removed,dict(stroke_width=sw,removed_components=rows)

def render(case,original,howe,veto,candidate,removed):
    x,y,w,h=ROIS[case['id']];box=(x,y,x+w,y+h)
    header=70;gap=16
    canvas=Image.new('RGB',(w*5+gap*6,h+header+gap),'#f4f2ec')
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,24)
    labels=['原画像','Howe 固定','Howe − CSC劣化','連結成分による試作','除去候補（赤）']
    overlay=original.copy();overlay[removed]=[255,45,45]
    images=[original,np.where(howe[:,:,None],0,255).repeat(3,axis=2).astype(np.uint8),
            np.where(veto[:,:,None],0,255).repeat(3,axis=2).astype(np.uint8),
            np.where(candidate[:,:,None],0,255).repeat(3,axis=2).astype(np.uint8),overlay]
    for i,(name,array) in enumerate(zip(labels,images)):
        px=gap+i*(w+gap)
        draw.text((px,10),f"{case['id']} {name}",font=font,fill='black')
        canvas.paste(Image.fromarray(array).crop(box),(px,header))
    path=OUT/'figures'/f"{case['id']}_マスク融合比較.png"
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)

def review_components(case,original,removed,rows):
    if not rows:return
    x,y,w,h=ROIS[case['id']]
    def overlaps_roi(c):
        xx,yy,ww,hh=c['xywh']
        return xx<x+w and yy<y+h and xx+ww>x and yy+hh>y
    selected=sorted(rows,key=lambda c:(not overlaps_roi(c),-c['csc_fraction'],-c['area']))[:20]
    size,gap,header=160,12,58
    panel_w=size*2+gap;panel_h=size+header
    cols=3;nr=(len(selected)+cols-1)//cols
    sheet=Image.new('RGB',(cols*(panel_w+gap)+gap,nr*(panel_h+gap)+gap),'#f4f2ec')
    draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,17)
    overlay=original.copy();overlay[removed]=[255,45,45]
    for i,c in enumerate(selected):
        xx,yy,ww,hh=c['xywh'];cx,cy=xx+ww//2,yy+hh//2
        px,py=gap+(i%cols)*(panel_w+gap),gap+(i//cols)*(panel_h+gap)
        draw.text((px,py),f"{case['id']} xy={xx},{yy} A={c['area']}",font=font,fill='black')
        draw.text((px,py+23),f"CSC率 {c['csc_fraction']:.2f} / {c['reason']}",font=font,fill='black')
        box=(cx-size//2,cy-size//2,cx+size//2,cy+size//2)
        for j,array in enumerate([original,overlay]):
            sheet.paste(Image.fromarray(array).crop(box),(px+j*(size+gap),py+header))
    sheet.save(OUT/'figures'/f"{case['id']}_除去候補の文脈確認.png")

def main():
    cv2.setNumThreads(2)
    cases=json.loads((BASE/'test_candidates.json').read_text())['cases']
    records=[];dependencies={}
    for case in cases:
        page=case['name'];source=ROOT/case['source']
        assert sha(source)==case['sha256']
        original=np.asarray(Image.open(source).convert('RGB'))
        d=ROOT/'data/split_dataset_csc_mask_v2/test'/page.split('_')[0]/page
        paths=dict(howe=BASE/'three_method_comparison/masks'/page/'howe.png',
                   csc=d/'02_text_black.png',red=d/'03_text_red.png',
                   degradation=d/'04_degradation.png',background=d/'05_background.png')
        masks={k:load(p,k in ['howe','csc','red']) for k,p in paths.items()}
        assert all(m.shape==original.shape[:2] for m in masks.values())
        h,b,r,deg,bg=(masks[k] for k in ['howe','csc','red','degradation','background'])
        veto=h&~deg
        candidate,removed,meta=components(h,b,deg,r)
        for method,mask in [('howe',h),('pixel_veto',veto),('component',candidate)]:
            save(OUT/'masks'/method/page/'02_text_black.png',mask)
            save(OUT/'masks'/method/page/'03_text_red.png',r)
        x,y,w,height=ROIS[case['id']];s=np.s_[y:y+height,x:x+w]
        union=b|r|deg|bg
        row=dict(id=case['id'],page=page,source=case['source'],roi_xywh=ROIS[case['id']],
                 howe_pixels_roi=int(h[s].sum()),howe_added_roi=int((h&~b)[s].sum()),
                 howe_added_unassigned_roi=int((h&~union)[s].sum()),
                 pixel_veto_removed_roi=int((h&deg)[s].sum()),
                 component_removed_roi=int(removed[s].sum()),component_removed_page=int(removed.sum()),
                 csc_pixels_removed_roi=int((removed&b)[s].sum()),**meta)
        records.append(row)
        for p in [source,*paths.values()]:dependencies[str(p.relative_to(ROOT))]=sha(p)
        render(case,original,h,veto,candidate,removed)
        review_components(case,original,removed,meta['removed_components'])
        print(case['id'],'sw',round(meta['stroke_width'],2),'veto ROI',row['pixel_veto_removed_roi'],
              'component ROI',row['component_removed_roi'],'components',len(meta['removed_components']),flush=True)
    dump(OUT/'protocol.json',dict(rule=RULE,rois_xywh=ROIS,dependencies=dependencies,
          scope='development diagnostic; no ground-truth labels',source_sha256=sha(Path(__file__))))
    dump(OUT/'diagnostics.json',records)

if __name__=='__main__':main()
