from pathlib import Path
import sys
sys.path.insert(0,str(Path('scripts/benchmark').resolve()))
import cv2,numpy as np,json
from build_npj_v6_comparison import read,write,ROOT
from probe_defect_context_preprocessing import paper_estimate
src=ROOT/'tmp_work/saved_npj_hnsd_input_scope_diagnostic_20261002/native/reference.png'
a=read(src);lab=cv2.cvtColor(a.astype(np.float32)/255,cv2.COLOR_BGR2LAB);L=lab[:,:,0]
paper=paper_estimate(lab);drop=paper[:,:,0]-L
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
close=cv2.morphologyEx(L,cv2.MORPH_CLOSE,k)
# Long horizontal structures from local contrast; broad paper shadows discarded.
ink=(close-L)>8
hor=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,93),np.uint8))
print('row horizontal count maxima',sorted(enumerate(hor.sum(1).tolist()),key=lambda p:-p[1])[:20])
print('L/drop/chroma quantiles whole',np.quantile(L,[0,.1,.5,.9,1]),np.quantile(drop,[0,.1,.5,.9,1]))
for name,box in [('text',(30,50,160,260)),('lower',(170,400,350,170)),('stain',(230,420,150,100)),('paper',(10,450,140,80)),('outside',(0,570,620,50))]:
 x,y,w,h=box;s=np.s_[y:y+h,x:x+w]
 print(name,'L',np.quantile(L[s],[.1,.5,.9]),'drop',np.quantile(drop[s],[.1,.5,.9]),'ab',np.median(lab[s,1:],axis=(0,1)) if False else np.median(lab[s][:,:,1:],axis=(0,1)))
write(ROOT/'tmp_work/ruled_margin_explore_20261002/ink.png',ink)
write(ROOT/'tmp_work/ruled_margin_explore_20261002/horizontal.png',hor.astype(bool))
