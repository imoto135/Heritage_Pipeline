"""Development: full-page ruled-frame context for dirt-only saved-npj edits.

Pair top/bottom rules and vertical sides before declaring any frame exterior.
Keep saved restoration, mandatory colors and frame interior unchanged.
Thresholds are written before processing the same eight recorded frames.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_role_separated_review import figure
from probe_defect_context_preprocessing import paper_estimate
from probe_saved_npj_role_separated_postprocess import frame_guard
import probe_npj_dirt_clean_reference as renderer

OUT=ROOT/'tmp_work/npj_full_page_layout_20261004'
TRANSFER=ROOT/'tmp_work/npj_dirt_transfer_20261004'
UNSEEN=ROOT/'tmp_work/npj_dirt_unseen_books_20261004'
SETTINGS=dict(
    layout_scope='Full original page; no crop coordinates, ids, book names or semantic ROIs supplied to detect_frame',
    dark_input='Median3 original Lab L; 31px closing minus L >= 8',
    hough=dict(rho=1,theta_degrees=.5,threshold='max(80,.08*width)',
               horizontal_min_length='.45*width',vertical_min_length='.25*height',
               max_gap='.025*max(width,height)',slope_limit=.04),
    frame=dict(min_height_fraction=.25,min_width_fraction=.45,
               interior_side_border_fraction=.02,vertical_overlap_fraction=.60,
               min_internal_dark_fraction=.015,rule_halo_px=8,
               selection='Largest supported frame; conservative axis-aligned envelope of all four line endpoints'),
    paper_footprint='Largest external contour of original L>=35 after 65px closing, eroded 5px',
    broad_damage='Previous sigma8 L-drop: soft>=5, seed>=8, original b>=5; 128px area,32px seeds, peak radius>=4',
    saved_npj_residual='65px closing L-drop>=12;32px and >=3% residual within candidate',
    mandatory='Existing fixed-v6 red|extra and stored CSC red, native crop; no blanket removal of keep',
    mask='Only selected broad components outside full-page frame; frame border2px excluded',
    renderer='Unchanged Oct2 clean-reference brightness/closing thresholds, sigma2 and four-neighbour solve',
    no_stroke_generation=True, independent_test=False, adopted=False,
    limitations=['Frame exterior can contain real writing; color and shape are not material GT',
                 'Two monochrome books remain subject to the original warm-color requirement',
                 'Broad components are evaluated on available saved-restoration frames, not on a nonexistent whole-page restoration',
                 'Full-page envelope deliberately holds nearby slanted rules'])


def detect_frame(original):
    """No page identifier or crop metadata enters this function."""
    H,W=original.shape[:2]
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    L=lab[:,:,0]
    local=cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    ink=(local-L)>=8
    lines=[]
    for direction,min_length in [('h',round(.45*W)),('v',round(.25*H))]:
        detected=cv2.HoughLinesP(ink.astype(np.uint8)*255,1,np.pi/360,
            threshold=max(80,round(.08*W)),minLineLength=min_length,maxLineGap=max(16,round(.025*max(W,H))))
        if detected is None:continue
        for raw in detected[:,0]:
            x0,y0,x1,y1=map(int,raw);dx=x1-x0;dy=y1-y0
            if direction=='h' and abs(dx)>=.45*W and abs(dy)<=.04*abs(dx):
                if x1<x0:x0,y0,x1,y1=x1,y1,x0,y0
                lines.append(dict(kind='h',xyxy=[x0,y0,x1,y1],x0=x0,x1=x1,center=(y0+y1)/2))
            elif direction=='v' and abs(dy)>=.25*H and abs(dx)<=.04*abs(dy):
                if y1<y0:x0,y0,x1,y1=x1,y1,x0,y0
                lines.append(dict(kind='v',xyxy=[x0,y0,x1,y1],y0=y0,y1=y1,center=(x0+x1)/2))
    horizontal=[s for s in lines if s['kind']=='h'];vertical=[s for s in lines if s['kind']=='v']
    candidates=[]
    for top in horizontal:
        for bottom in horizontal:
            yt,yb=top['center'],bottom['center']
            if yb-yt<.25*H:continue
            xl=max(top['x0'],bottom['x0']);xr=min(top['x1'],bottom['x1'])
            if xr-xl<.45*W:continue
            left=[];right=[]
            for side in vertical:
                overlap=max(0,min(side['y1'],yb)-max(side['y0'],yt))/(yb-yt)
                if overlap<.60:continue
                if abs(side['center']-xl)<=.04*W:left.append(side)
                if abs(side['center']-xr)<=.04*W:right.append(side)
            if not left or not right:continue
            lv=min(left,key=lambda s:abs(s['center']-xl));rv=min(right,key=lambda s:abs(s['center']-xr))
            if rv['center']-lv['center']<.45*W:continue
            all_points=np.array([s['xyxy'] for s in [top,bottom,lv,rv]]).reshape(-1,2)
            # Use actual line endpoints, not an optimistic fitted inner rectangle.
            xmin,ymin=all_points.min(0);xmax,ymax=all_points.max(0)
            if xmin<.02*W or xmax>.98*W or ymin<.02*H or ymax>.98*H:continue
            inside=ink[int(yt)+8:int(yb)-8,round(lv['center'])+8:round(rv['center'])-8]
            fraction=float(inside.mean()) if inside.size else 0.
            if fraction<.015:continue
            candidates.append(dict(box=[int(xmin),int(ymin),int(xmax),int(ymax)],
                lines=[s['xyxy'] for s in [top,bottom,lv,rv]],ink_inside=fraction,
                score=float((xr-xl)*(yb-yt))))
    chosen=max(candidates,key=lambda c:c['score']) if candidates else None
    interior=np.ones((H,W),bool) if chosen is None else np.zeros((H,W),bool)
    exterior=np.zeros((H,W),bool)
    if chosen:
        x0,y0,x1,y1=chosen['box'];radius=8
        interior[max(0,y0-radius):min(H,y1+radius+1),max(0,x0-radius):min(W,x1+radius+1)]=True
        exterior=~interior
    bright=(L>=35).astype(np.uint8)
    footprint=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))
    contours,_=cv2.findContours(footprint,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    page=np.zeros((H,W),np.uint8)
    if contours:cv2.drawContours(page,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    page=cv2.erode(page,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
    exterior &= page
    return dict(interior=interior,exterior=exterior,footprint=page,ink=ink),dict(chosen=chosen,
        supported_frames=len(candidates),horizontal_lines=len(horizontal),vertical_lines=len(vertical))


def damage_features(original):
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    paper=paper_estimate(lab)
    drop=cv2.GaussianBlur(paper[:,:,0]-lab[:,:,0],(0,0),8,borderType=cv2.BORDER_REFLECT_101)
    return drop,lab[:,:,2]


def select_update(base,drop,b,exterior,mandatory):
    L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
    background=cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))
    residual=background-L>=12
    valid=exterior&frame_guard(base.shape[:2],2)&(b>=5)&~mandatory
    soft=(drop>=5)&valid;seed=(drop>=8)&soft
    n,labels,stats,_=cv2.connectedComponentsWithStats(soft.astype(np.uint8),connectivity=8)
    seeds=np.bincount(labels.ravel(),weights=seed.ravel(),minlength=n)
    residuals=np.bincount(labels.ravel(),weights=residual.ravel(),minlength=n)
    distance=cv2.distanceTransform(soft.astype(np.uint8),cv2.DIST_L2,5)
    peaks=np.zeros(n,np.float32);np.maximum.at(peaks,labels.ravel(),distance.ravel())
    area=stats[:,4]
    use=(area>=128)&(seeds>=32)&(peaks>=4)&(residuals>=32)&(residuals/np.maximum(area,1)>=.03)
    use[0]=False
    update=use[labels]&soft
    return update,dict(components=[dict(id=i,xywh=stats[i,:4].tolist(),area=int(area[i]),
        seeds=int(seeds[i]),saved_npj_dark_residual=int(residuals[i]),peak_radius=float(peaks[i]),accepted=bool(use[i])) for i in range(1,n)],
        candidate_pixels=int(update.sum()))


def sources():
    frames=[]
    for folder in [TRANSFER,UNSEEN]:
        selection=json.loads((folder/'selection.json').read_text())
        for c in selection['cases']:frames.append(dict(c,source_folder=str(folder.relative_to(ROOT)),
             saved_npj=str((folder/'native'/c['id']/'baseline.png').relative_to(ROOT)),
             legacy_output=str((folder/'native'/c['id']/'ruled_clean_reference.png').relative_to(ROOT)),
             original_crop=str((folder/'native'/c['id']/'original.png').relative_to(ROOT)),
             v6_roles=str((folder/'classification/v6_masks'/c['page']).relative_to(ROOT))))
    frames.append(dict(id='saved_hnsd',page='hnsd001_011',book='hnsd00000',
        original='data/split_dataset/train/hnsd00000/hnsd001_011.jpg',xywh=[80,2280,620,620],
        saved_npj='tmp_work/npj_dirt_clean_reference_20261002/native/current_control/baseline.png',
        legacy_output='tmp_work/npj_dirt_clean_reference_20261002/native/current_control/clean_reference.png',
        original_crop='tmp_work/npj_dirt_clean_reference_20261002/native/current_control/original.png',
        npj_red='data/split_dataset_csc_mask/train/hnsd00000/hnsd001_011/03_text_red.png',
        v6_roles='tmp_work/npj_failure_collection_20261001/v6_masks/hnsd001_011',
        baseline_kind='Historical saved npj; current frozen-code control is pixel-identical'))
    return frames


def main():
    assert not OUT.exists(),'Use a new output name'
    OUT.mkdir(parents=True)
    frames=sources();deps={}
    for p in [Path(__file__),Path(renderer.__file__),ROOT/'scripts/benchmark/probe_defect_context_preprocessing.py',
              ROOT/'scripts/benchmark/package_saved_npj_role_separated_review.py',ROOT/'scripts/benchmark/build_npj_v6_comparison.py']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    for folder in [TRANSFER,UNSEEN,renderer.OUT]:
        frozen=json.loads((folder/'complete.json').read_text())
        for rel,digest in frozen['outputs'].items():assert sha(folder/rel)==digest,rel
        deps[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json')
    for c in frames:
        for key in ['original','saved_npj','legacy_output','original_crop','npj_red']:
            deps[c[key]]=sha(ROOT/c[key])
        for key in ['red','extra','protect','hold','edit']:
            p=ROOT/c['v6_roles']/(key+'.png');deps[str(p.relative_to(ROOT))]=sha(p)
    dump(OUT/'manifest.json',dict(settings=SETTINGS,frames=frames,dependencies=deps,source_sha256=sha(Path(__file__))))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    records=[]
    for c in frames:
        original=read(ROOT/c['original']);base=read(ROOT/c['saved_npj']);legacy=read(ROOT/c['legacy_output'])
        x,y,w,h=c['xywh'];s=np.s_[y:y+h,x:x+w];reference=original[s]
        assert np.array_equal(reference,read(ROOT/c['original_crop']))
        masks,layout=detect_frame(original);drop,b=damage_features(original)
        roles={k:read(ROOT/c['v6_roles']/(k+'.png'),True)[s] for k in ['red','extra','protect','hold','edit']}
        red=cv2.imread(str(ROOT/c['npj_red']),0)[s]<127
        mandatory=roles['red']|roles['extra']|red
        exterior=masks['exterior'][s];interior=masks['interior'][s]
        update,components=select_update(base,drop[s],b[s],exterior,mandatory)
        known,threshold=renderer.paper_reference(base,update,exterior,mandatory)
        repaired,applied,used,solves=renderer.fill(base,update,known)
        assert not (update&interior).any() and not (update&mandatory).any()
        assert np.array_equal(repaired[interior|mandatory],base[interior|mandatory])
        assert np.array_equal(repaired[~applied],base[~applied])
        poison=base.copy();poison[~known]=[255,0,255]
        poisoned,pa,_,_=renderer.fill(poison,update,known)
        assert np.array_equal(poisoned[applied],repaired[applied]) and np.array_equal(pa,applied)
        native=OUT/'native'/c['id']
        for key,a in dict(original=reference,baseline=base,legacy=legacy,repaired=repaired,
            interior=interior,exterior=exterior,mandatory=mandatory,update=update,applied=applied,
            reference=known,used_reference=used).items():write(native/(key+'.png'),a)
        page_image=original.copy()
        if layout['chosen']:
            for line in layout['chosen']['lines']:
                cv2.line(page_image,tuple(line[:2]),tuple(line[2:]),(0,0,255),max(2,round(original.shape[1]/600)))
        page_image=overlay(page_image,[('blue',masks['exterior'])])
        figure([original,page_image],['原ページ','赤＝支持した枠線、青＝紙面内の枠外'],OUT/(c['id']+'_page_layout.png'),
            c['page']+'：枠の判定は原ページ全体から',
            ['横線2本と縦線2本を照合。判定不能なら編集領域を作らない。',
             '軸に沿った外包で本文を保持。枠外にも文字・印影はあり得る。'],overview=True)
        figure([reference,base,legacy,repaired],['原画像','保存したnpj基準','前回の枠外試作','原ページの枠外だけ'],
            OUT/(c['id']+'_comparison.png'),c['page']+'：本文保持と汚れ除去を同時に確認',
            ['追加DDRMなし。紙面補間は前回と同じ、変えるのは枠外候補の文脈。',
             '本文の外包・朱候補・暖色保留・更新外はnpj基準を維持。筆画補完なし。',
             '既知の失敗例を含む開発段階。新しい独立冊の検証とは数えない。'],overview=True)
        keep=roles['protect']|roles['hold']|mandatory
        L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        after=cv2.cvtColor(repaired.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        record=dict(id=c['id'],page=c['page'],layout=layout,components=components,solves=solves,
            update_pixels=int(update.sum()),applied_pixels=int(applied.sum()),changed_pixels=int(np.any(repaired!=base,2).sum()),
            update_overlap_v6_keep=int((update&keep).sum()),removed_dark_pixels=int((applied&(L<30)&(after>60)).sum()),
            kept_interior_and_mandatory_exact=True,reference_L_threshold=threshold,excluded_reference_poison_invariant=True)
        records.append(record);print(c['id'],'frame',layout['chosen']['box'] if layout['chosen'] else None,
            'update',record['update_pixels'],'applied',record['applied_pixels'],flush=True)
        del original,masks,drop,b;gc.collect()
    dump(OUT/'diagnostics.json',dict(cases=records))
    dump(OUT/'verification.json',dict(interior_and_mandatory_exact=True,outside_applied_exact=True,
        panels_exact=True,excluded_reference_poison_invariant=True,thresholds_fixed_before_run=True,
        no_stroke_generation=True,pixel_GT=False))


if __name__=='__main__':main()
