"""Frame v3: real directional line support and fitted corner intersections.

V2 merged text alignments as well as rules; their remote endpoints rejected
valid frames. V3 checks actual line occupancy and uses intersections of the
four fitted rules. Damage thresholds and renderer remain unchanged.
"""
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import probe_npj_full_page_layout as core
from probe_npj_full_page_layout_v2 import group_lines
from build_npj_v6_comparison import ROOT,sha,dump

OUT=ROOT/'tmp_work/npj_full_page_layout_v3_20261004'


def fit_group(group,H,W):
    kind=group['kind']; slopes=[]
    for x0,y0,x1,y1 in group['members']:
        slopes.append((y1-y0)/max(x1-x0,1) if kind=='h' else (x1-x0)/max(y1-y0,1))
    return dict(kind=kind,slope=float(np.median(slopes)),center=group['center'],members=group['members'],
                bounds=group['bounds'],H=H,W=W)


def intersection(horizontal,vertical):
    m,n=horizontal['slope'],vertical['slope']
    a=horizontal['center']-m*.5*horizontal['W']
    b=vertical['center']-n*.5*vertical['H']
    y=(m*b+a)/(1-m*n);x=n*y+b
    return np.array([x,y],np.float64)


def occupancy(mask,p0,p1):
    length=max(2,round(float(np.linalg.norm(p1-p0))))
    points=np.linspace(p0,p1,length)
    H,W=mask.shape;seen=np.zeros(length,bool)
    for dy in range(-3,4):
        for dx in range(-3,4):
            x=np.rint(points[:,0]).astype(int)+dx;y=np.rint(points[:,1]).astype(int)+dy
            valid=(x>=0)&(x<W)&(y>=0)&(y<H)
            seen[valid]|=mask[y[valid],x[valid]]
    return float(seen.mean())


def detect_frame(original):
    H,W=original.shape[:2]
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    L=lab[:,:,0]
    local=cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    ink=(local-L)>=8
    directional={k:cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,
                 np.ones((1,31) if k=='h' else (31,1),np.uint8))>0 for k in ['h','v']}
    lines=[]
    for direction,min_length in [('h',round(.45*W)),('v',round(.25*H))]:
        detected=cv2.HoughLinesP(directional[direction].astype(np.uint8)*255,1,np.pi/360,
            threshold=max(80,round(.08*W)),minLineLength=min_length,maxLineGap=max(16,round(.025*max(W,H))))
        if detected is None:continue
        for raw in detected[:,0]:
            x0,y0,x1,y1=map(int,raw);dx=x1-x0;dy=y1-y0
            if direction=='h' and abs(dx)>=.45*W and abs(dy)<=.04*abs(dx):
                if x1<x0:x0,y0,x1,y1=x1,y1,x0,y0
                lines.append(dict(kind='h',xyxy=[x0,y0,x1,y1]))
            elif direction=='v' and abs(dy)>=.25*H and abs(dx)<=.04*abs(dy):
                if y1<y0:x0,y0,x1,y1=x1,y1,x0,y0
                lines.append(dict(kind='v',xyxy=[x0,y0,x1,y1]))
    hg=group_lines(lines,'h',H,W);vg=group_lines(lines,'v',H,W)
    hs=[fit_group(g,H,W) for g in hg];vs=[fit_group(g,H,W) for g in vg]
    candidates=[]
    for top in hs:
        for bottom in hs:
            yt,yb=top['center'],bottom['center']
            if yb-yt<.25*H:continue
            xl=max(top['bounds'][0],bottom['bounds'][0]);xr=min(top['bounds'][2],bottom['bounds'][2])
            if xr-xl<.45*W:continue
            sides=[]
            for v in vs:
                overlap=max(0,min(v['bounds'][3],yb)-max(v['bounds'][1],yt))/(yb-yt)
                if overlap>=.60:sides.append(v)
            left=[v for v in sides if abs(v['center']-xl)<=.04*W]
            right=[v for v in sides if abs(v['center']-xr)<=.04*W]
            for lv in left:
                for rv in right:
                    tl,tr=intersection(top,lv),intersection(top,rv)
                    bl,br=intersection(bottom,lv),intersection(bottom,rv)
                    corners=np.array([tl,tr,br,bl])
                    minimum=corners.min(0);maximum=corners.max(0)
                    xmin,ymin=map(int,np.floor(minimum));xmax,ymax=map(int,np.ceil(maximum))
                    if xmax-xmin<.45*W:continue
                    if xmin<.02*W or xmax>.98*W or ymin<.02*H or ymax>.98*H:continue
                    supports=[occupancy(directional['h'],tl,tr),occupancy(directional['h'],bl,br),
                              occupancy(directional['v'],tl,bl),occupancy(directional['v'],tr,br)]
                    if min(supports)<.45:continue
                    interior_ink=ink[ymin+8:ymax-8,xmin+8:xmax-8]
                    fraction=float(interior_ink.mean()) if interior_ink.size else 0.
                    if fraction<.015:continue
                    drawn=[np.rint(np.concatenate([a,b])).astype(int).tolist()
                           for a,b in [(tl,tr),(bl,br),(tl,bl),(tr,br)]]
                    candidates.append(dict(box=[xmin,ymin,xmax,ymax],corners=corners.tolist(),
                         lines=drawn,support_fraction=supports,ink_inside=fraction,
                         score=float((xmax-xmin)*(ymax-ymin)),grouped=True,
                         raw_support_members=top['members']+bottom['members']+lv['members']+rv['members']))
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
         supported_frames=len(candidates),horizontal_lines=sum(s['kind']=='h' for s in lines),
         vertical_lines=sum(s['kind']=='v' for s in lines),horizontal_groups=len(hg),vertical_groups=len(vg))


if __name__=='__main__':
    digest=sha(Path(__file__))
    core.OUT=OUT;core.SETTINGS=copy.deepcopy(core.SETTINGS)
    core.SETTINGS['frame']['fragment_grouping']='Mid-page intercept complete-link span <= .012*orthogonal page size'
    core.SETTINGS['frame']['directional_filter']='31px horizontal/vertical morphology opening before Hough'
    core.SETTINGS['frame']['corners']='Intersections of robust-median slope and intercept of four rule groups'
    core.SETTINGS['frame']['support_gate']='Each fitted rule requires >=45% directional ink occupancy within 3px'
    core.detect_frame=detect_frame
    core.main()
    assert sha(Path(__file__))==digest
    manifest=json.loads((OUT/'manifest.json').read_text())
    for p in [Path(__file__),ROOT/'scripts/benchmark/probe_npj_full_page_layout_v2.py']:
        manifest['dependencies'][str(p.relative_to(ROOT))]=sha(p)
    manifest['entrypoint']=str(Path(__file__).relative_to(ROOT))
    manifest['entrypoint_sha256_recorded_before_run']=digest
    dump(OUT/'manifest.json',manifest);shutil.copyfile(__file__,OUT/'entrypoint_snapshot.py')
