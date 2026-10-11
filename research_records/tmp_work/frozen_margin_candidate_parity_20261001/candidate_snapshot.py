"""Array-only composition of the frozen development margin-mask conditions.

No page IDs, ROI coordinates, file reads, cached masks, or semantic labels enter
infer(). Inputs are the original BGR image, fixed-v6 masks, and their original
image features. This remains a development candidate, not a damage classifier.
"""
import cv2
import numpy as np
import probe_paper_rim_lowfrequency_v2 as rim
import probe_margin_top_completion as top
import probe_thin_margin_components as thin
import probe_anchored_pale_margin_masks as pale
import diagnose_unanchored_upper_cores as upper
import diagnose_directional_paper_candidates as direction

SETTINGS=dict(thin_peak=3.5,fragment_distance=8,pale_drop=6,new_upper_horizontal_ratio=3,
              core_halo_radius=6,halo_drop=2,recursive=False,ROI_gating=False,
              direction_settings=direction.SETTINGS['provisional_match'],development=True)


def fragments(connected,margin,seed,anchor):
    if not anchor.any():return np.zeros_like(anchor),[]
    H,W=margin.shape
    n,labels,stats,_=cv2.connectedComponentsWithStats(connected.astype(np.uint8),connectivity=8)
    seeds=np.bincount(labels.ravel(),weights=(seed&margin).ravel(),minlength=n)
    dt=cv2.distanceTransform(connected.astype(np.uint8),cv2.DIST_L2,5)
    peak=np.zeros(n,np.float32);np.maximum.at(peak,labels.ravel(),dt.ravel())
    away=cv2.distanceTransform((~anchor).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    gap=np.full(n,np.inf);np.minimum.at(gap,labels.ravel(),away.ravel())
    inner=cv2.erode(margin.astype(np.uint8),np.ones((3,3),np.uint8))>0
    boundary=np.bincount(labels.ravel(),weights=(connected&~inner).ravel(),minlength=n)
    selected=np.zeros(n,bool);notes=[]
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if 16<=area<128 and seeds[k]>=8 and peak[k]>=2 and 0<gap[k]<=8 and boundary[k]==0 and max(w,h)<=.2*min(H,W):
            selected[k]=True
            notes.append(dict(id=k,xywh=[x,y,w,h],area=area,brown_seeds=int(seeds[k]),peak=float(peak[k]),distance=float(gap[k])))
    return selected[labels]&connected,notes


def halo(core):
    return cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0


def infer(original,v6,features):
    shape=original.shape[:2]
    assert original.dtype==np.uint8 and original.ndim==3 and original.shape[2]==3
    assert all(v.shape==shape and v.dtype==bool for v in v6.values())
    assert all(features[k].shape==shape for k in ['light_drop','color_seed','warm_anchor_guard'])
    modes,maps,margin_notes=rim.MARGIN.infer(original,v6,features)
    completions,upper_maps,completion_notes=top.infer(original,features,modes['dark_support'],maps)
    domain=upper_maps['safe_domain']
    guard=upper_maps['warm_anchor_guard']|upper_maps['long_line_guard']|v6['red']|v6['extra']
    initial_core=maps['accepted_core']|upper_maps['completed_core']
    rim_edit,_,_=rim.rim_mask(completions['completed']['edit'],initial_core,features['light_drop'],domain,guard)
    strict,_=thin.select(maps['dark_support'],maps['top_margin'],features['color_seed'],6)
    assert np.array_equal(strict,maps['accepted_core'])
    relaxed,thin_notes=thin.select(maps['dark_support'],maps['top_margin'],features['color_seed'],3.5)
    new_thin=relaxed&~strict
    allowed=domain&~guard
    thin_edit=rim_edit|(halo(new_thin)&allowed&(features['light_drop']>=2))
    fragment_core,fragment_notes=fragments(maps['dark_support'],maps['top_margin'],features['color_seed'],new_thin)
    near_edit=thin_edit|(halo(fragment_core)&allowed&(features['light_drop']>=2))
    core=initial_core|new_thin|fragment_core
    connected,pale_notes=pale.candidate(original,features,core,domain,guard,6)
    pale_edit=near_edit|connected
    _,strong_notes=upper.infer(original,features,core,domain,guard)
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    support=(features['light_drop']>=10)&(lab[:,:,0]<=80)&allowed
    support=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    support&=allowed
    _,labels=cv2.connectedComponents(support.astype(np.uint8),connectivity=8)
    ids=[n['id'] for n in strong_notes if n['accepted'] and n['old_core_overlap']==0 and n['xywh'][2]<=3*n['xywh'][3]]
    strong=np.isin(labels,ids)&support
    linked,upper_pale_notes=pale.candidate(original,features,core|strong,domain,guard,6)
    edit=pale_edit|((linked|(halo(strong)&(features['light_drop']>=2)))&allowed)
    assert not np.any((edit&~v6['edit'])&~allowed)
    return dict(edit=edit,core=core,strong_core=strong,domain=domain,guard=guard,
        near_edit=near_edit,pale_edit=pale_edit,strict_core=strict,new_thin_core=new_thin,fragment_core=fragment_core),dict(
        margin=margin_notes,completion=completion_notes,thin=thin_notes,fragments=fragment_notes,
        pale=pale_notes,upper=strong_notes,upper_pale=upper_pale_notes)


def infer_hold(original,masks,known):
    assert known.shape==original.shape[:2] and not np.any(known&masks['edit'])
    light=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
    eligible=masks['edit']&masks['domain']&~masks['guard']
    return direction.infer(light,known,eligible,masks['core']|masks['strong_core'],masks['domain'],masks['guard'])
