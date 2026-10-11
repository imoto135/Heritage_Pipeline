"""SWT/paired-edge inspired diagnostic, NOT exact Xiong/SSP reproduction.

Entire original frame, no evaluation ROIs in inference. Adaptive median ray
width sets a morphology background scale. Opposing gradient pairs are detected
again after compensation. Evidence of stroke geometry is not material GT.
"""
import json,shutil
from pathlib import Path
import cv2,numpy as np
from build_npj_v6_comparison import ROOT,read,write,sha,dump
TRIAL=ROOT/'tmp_work/npj_literature_trial_20261002'
OUT=TRIAL/'structural'
SETTINGS=dict(preblur_sigma=.8,canny_high_gradient_quantile=.85,canny_low_high_ratio=.4,
 max_ray_px=32,min_ray_px=3,endpoint_opposition_cosine=.866,midpoint_darkness_min_gray=4,
 background_radius='ceil(1.5*median initial opposing-ray width), clipped [3,48]',
 compensated='255 - (morphology close(original gray) - original gray)',
 second_ray_width='[max(3,.5*initial_median),min(32,2*initial_median)]',
 protect='Accepted second-pass ray interiors and endpoints, 1px halo; not a true text mask',
 exact_paper_reimplementation=False,new_restoration=False,ROI_gating=False,training=False,pixel_GT=False)

def paired(gray,min_width=3,max_width=32):
 sm=cv2.GaussianBlur(gray.astype(np.float32),(0,0),.8)
 gx=cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3)
 mag=np.hypot(gx,gy);positive=mag[mag>0]
 high=float(np.quantile(positive,.85)) if positive.size else 255.
 high=max(high,10.)
 edges=cv2.Canny(np.rint(sm).clip(0,255).astype(np.uint8),.4*high,high,L2gradient=True)>0
 ys,xs=np.nonzero(edges);norm=np.maximum(mag[ys,xs],1e-6)
 dx=-gx[ys,xs]/norm;dy=-gy[ys,xs]/norm
 active=np.ones(len(xs),bool);accepted=[];h,w=gray.shape
 for step in range(1,int(max_width)+2):
  ids=np.flatnonzero(active)
  if not len(ids):break
  xx=np.rint(xs[ids]+step*dx[ids]).astype(int);yy=np.rint(ys[ids]+step*dy[ids]).astype(int)
  inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h);active[ids[~inside]]=False
  ids=ids[inside];xx=xx[inside];yy=yy[inside]
  hit=edges[yy,xx]&(step>=3)
  candidates=ids[hit];ex=xx[hit];ey=yy[hit]
  if len(candidates):
   active[candidates]=False
   dist=np.hypot(ex-xs[candidates],ey-ys[candidates])
   dot=(-dx[candidates]*gx[ey,ex]-dy[candidates]*gy[ey,ex])/np.maximum(mag[ey,ex],1e-6)
   mx=np.rint((xs[candidates]+ex)/2).astype(int);my=np.rint((ys[candidates]+ey)/2).astype(int)
   dark=(sm[ys[candidates],xs[candidates]]+sm[ey,ex])/2-sm[my,mx]
   good=(dot<=-.866)&(dist>=min_width)&(dist<=max_width)&(dark>=4)
   for i,a,b,d in zip(candidates[good],ex[good],ey[good],dist[good]):accepted.append([int(xs[i]),int(ys[i]),int(a),int(b),float(d)])
 mask=np.zeros(gray.shape,np.uint8)
 for x,y,ex,ey,_ in accepted:cv2.line(mask,(x,y),(ex,ey),1,1)
 return mask>0,edges,np.asarray(accepted,np.float32).reshape(-1,5),dict(edge_pixels=int(edges.sum()),canny_high=high,pairs=len(accepted))

def infer(original):
 gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
 first,edges,rays,diag1=paired(gray)
 median=float(np.median(rays[:,4])) if len(rays)>=32 else None
 if median is None:
  return dict(ink_support=np.zeros(gray.shape,bool),first_support=first,edges=edges,background=gray,compensated=gray),dict(status='insufficient_ray_pairs',first=diag1,median_width=None)
 radius=int(np.clip(np.ceil(1.5*median),3,48));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
 paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel)
 compensated=255-(paper.astype(np.int16)-gray.astype(np.int16));compensated=np.clip(compensated,0,255).astype(np.uint8)
 support,edges2,rays2,diag2=paired(compensated,max(3,.5*median),min(32,2*median))
 halo=cv2.dilate(support.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))>0
 return dict(ink_support=halo,paired_support=support,first_support=first,edges=edges,compensated_edges=edges2,
             background=paper,compensated=compensated),dict(status='diagnostic',first=diag1,second=diag2,median_width=median,background_radius=radius,support_pixels=int(halo.sum()))

def main():
 assert not (OUT/'manifest.json').exists(),'Use a new version'
 selection=json.loads((TRIAL/'selection.json').read_text());OUT.mkdir(parents=True,exist_ok=True)
 deps={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),TRIAL/'selection.json',TRIAL/'selection_complete.json']}
 dump(OUT/'manifest.json',dict(settings=SETTINGS,dependencies=deps,source_sha=sha(Path(__file__))))
 shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[]
 for c in selection['frames']:
  a=read(TRIAL/'inputs'/c['id']/'original.png');images,diag=infer(a)
  for k,v in images.items():write(OUT/c['id']/(k+'.png'),v)
  dump(OUT/c['id']/'diagnostics.json',diag);rows.append(dict(frame=c['id'],**diag));print(c['id'],json.dumps(diag),flush=True)
 dump(OUT/'diagnostics.json',rows)
 dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()}))
if __name__=='__main__':main()
