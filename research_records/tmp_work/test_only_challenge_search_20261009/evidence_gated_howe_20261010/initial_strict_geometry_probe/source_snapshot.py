"""Development prototype: preserve Howe ink unless independent evidence agrees.

Neighbor matching is optional. Neither page IDs nor reviewed coordinates enter
the selector. CSC-unassigned pixels remain explicitly unknown, never dirt labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/test_only_challenge_search_20261009'
OUT=BASE/'evidence_gated_howe_20261010'
RULE=dict(min_area=16,max_target_dark_fraction=.15,max_csc_black_fraction=.02,
          gray_dark=130,min_dark_support=.7,min_donor_contrast_gap=20,
          ncc_min=.55,split_ncc_min=.35,split_shift_tolerance=5,
          search_radius=65,neighbor_step_radius=2,context_min=160,
          context_margin_strokes=12,group_gap_strokes=8,max_context_side=512,
          support_radius_strokes=2,contrast_sigma=12,
          csc_unassigned_is_degradation=False,red_protected=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def savejson(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def load(path,black=True):
    a=np.asarray(Image.open(path).convert('L'))
    assert np.isin(a,[0,255]).all(),path
    return a<127 if black else a>127

def feature(gray):
    a=gray.astype(np.float32)
    return np.maximum(cv2.GaussianBlur(a,(0,0),RULE['contrast_sigma'])-a,0)

def neighbors(source):
    """Only adjacent pages present in the same declared data/test directory."""
    book,number,side=source.stem.split('_')
    ordinal=2*int(number)+int(side)-1
    selected=[]
    for p in sorted(source.parent.glob(book+'_*.jpg')):
        _,n,s=p.stem.split('_'); distance=abs(2*int(n)+int(s)-1-ordinal)
        if 0<distance<=RULE['neighbor_step_radius']:selected.append(p)
    return selected

def bestfit(target_feature,reference_feature,box):
    x,y,w,h=box;radius=RULE['search_radius'];H,W=target_feature.shape
    x0,y0=max(0,x-radius),max(0,y-radius)
    x1,y1=min(W,x+w+radius),min(H,y+h+radius)
    t=target_feature[y:y+h,x:x+w]
    r=reference_feature[y0:y1,x0:x1]
    if min(t.shape)<8 or t.std()<1 or r.shape[0]<h or r.shape[1]<w:return None
    scores=cv2.matchTemplate(r,t,cv2.TM_CCOEFF_NORMED)
    _,score,_,loc=cv2.minMaxLoc(scores)
    return dict(ncc=float(score),shift=[loc[0]+x0-x,loc[1]+y0-y])

def decide(gray,howe,csc,red,deg,bg,references):
    count,labels,stats,_=cv2.connectedComponentsWithStats(howe.astype(np.uint8),8)
    sw=max(2.,float(2*np.median(cv2.distanceTransform(howe.astype(np.uint8),cv2.DIST_L2,5)[howe])))
    area=stats[:,4]
    fractions={name:np.bincount(labels[mask].ravel(),minlength=count)/np.maximum(area,1)
               for name,mask in [('csc',csc),('red',red),('dark',gray<RULE['gray_dark']),
                                 ('degradation',deg),('background',bg),('unassigned',~(csc|red|deg|bg))]}
    eligible=(area>=RULE['min_area'])&(fractions['csc']<=RULE['max_csc_black_fraction'])&(
        fractions['dark']<=RULE['max_target_dark_fraction'])&(fractions['red']==0)
    eligible[0]=False
    gap=max(1,round(RULE['group_gap_strokes']*sw))
    _,groups=cv2.connectedComponents(cv2.dilate(eligible[labels].astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(gap*2+1,gap*2+1))),8)
    tf=feature(gray);refdata=[(p,a,feature(a)) for p,a in references]
    removed=np.zeros_like(howe);rows=[]
    group_ids=np.unique(groups[eligible[labels]])
    for gi in group_ids:
        ids=np.unique(labels[(groups==gi)&eligible[labels]]);ids=ids[ids!=0]
        sub=stats[ids];left,top=sub[:,:2].min(axis=0);right=(sub[:,0]+sub[:,2]).max();bottom=(sub[:,1]+sub[:,3]).max()
        margin=round(RULE['context_margin_strokes']*sw)
        w=max(RULE['context_min'],int(right-left+2*margin));h=max(RULE['context_min'],int(bottom-top+2*margin))
        x=max(0,min(gray.shape[1]-w,round((left+right-w)/2)));y=max(0,min(gray.shape[0]-h,round((top+bottom-h)/2)))
        box=[int(x),int(y),int(w),int(h)]
        row=dict(group=int(gi),components=ids.tolist(),context_xywh=box,matches=[],selected_components=[])
        if max(w,h)>RULE['max_context_side'] or w>gray.shape[1] or h>gray.shape[0]:
            row['decision']='hold_context_too_large';rows.append(row);continue
        for path,rg,rf in refdata:
            fit=bestfit(tf,rf,box)
            if fit is None:continue
            parts=[bestfit(tf,rf,[x,y,w,h//2]),bestfit(tf,rf,[x,y+h//2,w,h-h//2])]
            consistent=all(p is not None and p['ncc']>=RULE['split_ncc_min'] and
                np.linalg.norm(np.array(p['shift'])-fit['shift'])<=RULE['split_shift_tolerance'] for p in parts)
            row['matches'].append(dict(reference=str(path.relative_to(ROOT)),**fit,parts=parts,consistent=bool(consistent)))
        valid=[m for m in row['matches'] if m['consistent'] and m['ncc']>=RULE['ncc_min']]
        best=max(valid,key=lambda m:m['ncc']) if valid else None
        for i in ids:
            comp=labels==i;item=dict(component=int(i),xywh=stats[i,:4].tolist(),area=int(area[i]),
                csc_fractions={k:float(v[i]) for k,v in fractions.items()},decision='hold_no_verified_correspondence')
            if best:
                rg=next(a for p,a in references if str(p.relative_to(ROOT))==best['reference'])
                dx,dy=best['shift'];ys,xs=np.nonzero(comp);rx,ry=xs+dx,ys+dy
                inside=(rx>=0)&(rx<rg.shape[1])&(ry>=0)&(ry<rg.shape[0])
                rad=max(1,round(RULE['support_radius_strokes']*sw))
                support=cv2.dilate((rg<RULE['gray_dark']).astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1)))>0
                sf=float(support[ry[inside],rx[inside]].sum()/len(xs))
                donor_median=float(np.median(rg[ry[inside],rx[inside]])) if inside.any() else 255.
                target_median=float(np.median(gray[comp]))
                item.update(reference=best['reference'],ncc=best['ncc'],shift=best['shift'],
                    dark_support=sf,donor_gray_median=donor_median,target_gray_median=target_median)
                if sf>=RULE['min_dark_support'] and target_median-donor_median>=RULE['min_donor_contrast_gap']:
                    removed|=comp;item['decision']='exclude_external_shape_evidence';row['selected_components'].append(int(i))
                else:item['decision']='hold_insufficient_dark_support_or_contrast'
            row.setdefault('component_decisions',[]).append(item)
        row['decision']='local_exclusion' if row['selected_components'] else 'hold';rows.append(row)
    assert not (removed&red).any() and not (removed&csc).any()
    return howe&~removed,removed,dict(stroke_width=sw,eligible_components=int(eligible.sum()),groups=rows,
        removed_pixels=int(removed.sum()),removed_components=sum(len(r['selected_components']) for r in rows))

def render(case,source,howe,new,removed):
    x,y,w,h=case['roi_xywh'];pw=430;ph=round(h*pw/w);header=65
    sheet=Image.new('RGB',(4*pw,ph+header),'white');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',20)
    overlay=source.copy();overlay[removed]=[255,40,40]
    panels=[source,np.repeat(np.where(howe[:,:,None],0,255).astype(np.uint8),3,axis=2),
            np.repeat(np.where(new[:,:,None],0,255).astype(np.uint8),3,axis=2),overlay]
    for i,(title,arr) in enumerate(zip(['原画像','固定Howe','自動対応条件後','除去候補（赤）'],panels)):
        draw.text((i*pw+4,8),case['id']+' '+title,font=font,fill='black')
        sheet.paste(Image.fromarray(arr[y:y+h,x:x+w]).resize((pw,ph)),(i*pw,header))
    p=OUT/'figures'/f"{case['id']}_自動候補比較.png";p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',default='');args=parser.parse_args()
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    protocols=[('six_challenge_fixed_review',json.loads((BASE/'six_challenge_fixed_review/protocol.json').read_text())),
               ('fixed_holdout_validation',json.loads((BASE/'fixed_holdout_validation/protocol.json').read_text()))]
    cases=[]
    for origin,protocol in protocols:
        for c in protocol['cases']:
            c=dict(c);c['origin']=origin
            if 'roi_xywh' not in c:c['roi_xywh']=protocol.get('rois_xywh',{}).get(c['id'])
            if c['roi_xywh'] is None:
                import sys;sys.path.insert(0,str(BASE));import run_ddrm_pilot as driver
                c['roi_xywh']=driver.ROIS[c['id']]
            if not args.only or c['id'] in args.only.split(','):cases.append(c)
    savejson(OUT/'protocol.json',dict(rule=RULE,cases=cases,source_sha256=sha(Path(__file__)),
        status='conditions_frozen_before_run',development_only=True,final_test=False,
        uses_manual_roi_for_selection=False,uses_manual_component_ids=False,
        display_rois_only=True,full_ddrm_multiclass_integration=False))
    results=[]
    for c in cases:
        start=time.monotonic();source=ROOT/c['source'];raw=np.asarray(Image.open(source).convert('RGB'))
        H,W=raw.shape[:2];gray=cv2.cvtColor(raw,cv2.COLOR_RGB2GRAY)
        md=BASE/c['origin']/'masks/howe'/c['page']
        h=load(md/'02_text_black.png');r=load(md/'03_text_red.png')
        cd=ROOT/'data/split_dataset_csc_mask_v2/test'/source.parent.name/c['page']
        b=load(cd/'02_text_black.png');d=load(cd/'04_degradation.png',False);bg=load(cd/'05_background.png',False)
        refs=[(p,cv2.cvtColor(cv2.resize(cv2.imread(str(p))[:,::-1],(W,H)),cv2.COLOR_BGR2GRAY)) for p in neighbors(source)]
        new,removed,meta=decide(gray,h,b,r,d,bg,refs)
        dest=OUT/'masks/automatic'/c['page'];dest.mkdir(parents=True,exist_ok=True)
        for name,mask in [('02_text_black.png',new),('03_text_red.png',r)]:
            Image.fromarray(np.where(mask,0,255).astype(np.uint8)).save(dest/name)
        # Mutually exclusive routing classes. Unknown is retained, not repaired.
        route=np.full((H,W),4,np.uint8)
        route[bg&~d&~b&~r&~h]=2
        route[d&~bg&~b&~r&~h]=3
        route[new]=1;route[removed]=3;route[r]=0
        Image.fromarray(route).save(dest/'routing_labels.png')
        meta.update(id=c['id'],page=c['page'],origin=c['origin'],neighbors=[str(p.relative_to(ROOT)) for p,_ in refs],
          masks_unchanged=bool(np.array_equal(h,new)),route_counts={k:int((route==v).sum()) for k,v in
            [('red',0),('text',1),('background',2),('repair_candidate',3),('unknown',4)]},
          dependencies={str(p.relative_to(ROOT)):sha(p) for p in [source,md/'02_text_black.png',md/'03_text_red.png',
                    cd/'02_text_black.png',cd/'04_degradation.png',cd/'05_background.png']+[p for p,_ in refs]},
          seconds=round(time.monotonic()-start,2))
        assert sum(meta['route_counts'].values())==H*W
        savejson(OUT/'diagnostics'/f"{c['id']}.json",meta);render(c,raw,h,new,removed);results.append(meta)
        print(c['id'],'eligible',meta['eligible_components'],'removed',meta['removed_components'],meta['removed_pixels'],flush=True)
    savejson(OUT/'complete.json',dict(results=results,rule=RULE,code_sha256=sha(Path(__file__)),
        development_only=True,new_inferences=0,routing_labels_not_yet_ddrm_input=True))

if __name__=='__main__':main()
