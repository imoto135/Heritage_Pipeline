"""Full-page frame v2: merge collinear fragments before matching corners.

Keep the first version for its partial-column-frame counterexample. Numeric
damage selection and paper rendering remain identical to version 1.
"""
import copy
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import probe_npj_full_page_layout as core
from build_npj_v6_comparison import ROOT,sha,dump

OUT=ROOT/'tmp_work/npj_full_page_layout_v2_20261004'


def group_lines(lines,kind,H,W):
    members=[s for s in lines if s['kind']==kind]
    # Cluster by projected mid-page intercept, not by a fragment's own midpoint.
    for s in members:
        x0,y0,x1,y1=s['xyxy']
        if kind=='h':s['intercept']=y0+(y1-y0)*(.5*W-x0)/max(x1-x0,1)
        else:s['intercept']=x0+(x1-x0)*(.5*H-y0)/max(y1-y0,1)
    groups=[];tolerance=.012*(H if kind=='h' else W)
    for s in sorted(members,key=lambda s:s['intercept']):
        # Prevent chaining across several unrelated parallel rules.
        if not groups or s['intercept']-groups[-1][0]['intercept']>tolerance:groups.append([s])
        else:groups[-1].append(s)
    result=[]
    for group in groups:
        points=np.array([s['xyxy'] for s in group]).reshape(-1,2)
        x0,y0=points.min(0);x1,y1=points.max(0)
        item=dict(kind=kind,members=[s['xyxy'] for s in group],bounds=[int(x0),int(y0),int(x1),int(y1)],
                  center=float(np.median([s['intercept'] for s in group])))
        if kind=='h':item.update(x0=int(x0),x1=int(x1))
        else:item.update(y0=int(y0),y1=int(y1))
        result.append(item)
    return result


def detect_frame(original):
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
                lines.append(dict(kind='h',xyxy=[x0,y0,x1,y1]))
            elif direction=='v' and abs(dy)>=.25*H and abs(dx)<=.04*abs(dy):
                if y1<y0:x0,y0,x1,y1=x1,y1,x0,y0
                lines.append(dict(kind='v',xyxy=[x0,y0,x1,y1]))
    horizontal=group_lines(lines,'h',H,W);vertical=group_lines(lines,'v',H,W)
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
            for lv in left:
                for rv in right:
                    if rv['center']-lv['center']<.45*W:continue
                    members=top['members']+bottom['members']+lv['members']+rv['members']
                    points=np.array(members).reshape(-1,2)
                    xmin,ymin=points.min(0);xmax,ymax=points.max(0)
                    if xmin<.02*W or xmax>.98*W or ymin<.02*H or ymax>.98*H:continue
                    inside=ink[int(yt)+8:int(yb)-8,round(lv['center'])+8:round(rv['center'])-8]
                    fraction=float(inside.mean()) if inside.size else 0.
                    if fraction<.015:continue
                    candidates.append(dict(box=[int(xmin),int(ymin),int(xmax),int(ymax)],
                        lines=members,ink_inside=fraction,score=float((xr-xl)*(yb-yt)),
                        matched_side_centers=[lv['center'],rv['center']],grouped=True))
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
        vertical_lines=sum(s['kind']=='v' for s in lines),horizontal_groups=len(horizontal),vertical_groups=len(vertical))


if __name__=='__main__':
    digest=sha(Path(__file__))
    core.OUT=OUT
    core.SETTINGS=copy.deepcopy(core.SETTINGS)
    core.SETTINGS['frame']['fragment_grouping']='Projected mid-page intercept, complete-link span <= .012*orthogonal page size'
    core.SETTINGS['frame']['pairing']='Try all supported left/right side groups; choose largest supported full frame'
    core.detect_frame=detect_frame
    core.main()
    assert sha(Path(__file__))==digest
    manifest=json.loads((OUT/'manifest.json').read_text())
    manifest['dependencies'][str(Path(__file__).relative_to(ROOT))]=digest
    manifest['entrypoint']=str(Path(__file__).relative_to(ROOT))
    manifest['entrypoint_sha256_recorded_before_run']=digest
    dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'entrypoint_snapshot.py')
