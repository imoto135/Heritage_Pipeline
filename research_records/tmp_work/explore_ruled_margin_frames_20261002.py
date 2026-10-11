from pathlib import Path
import sys
sys.path.insert(0,str(Path('scripts/benchmark').resolve()))
import cv2,numpy as np,json
from build_npj_v6_comparison import read,ROOT
s=json.loads((ROOT/'tmp_work/saved_npj_paper_followup_selection_20261002/selection.json').read_text())
for c in s['cases']:
 if c['id']=='crop_hnsd':a=read(ROOT/'tmp_work/saved_npj_hnsd_input_scope_diagnostic_20261002/native/reference.png')
 else:a=read(ROOT/c['input'])
 h,w=a.shape[:2];L=cv2.cvtColor(a.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
 ink=(cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))-L)>8
 hor=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,max(81,round(.15*w))|1),np.uint8))
 rows=hor.sum(1)>.65*w
 n,lab,stats,_=cv2.connectedComponentsWithStats(rows.astype(np.uint8)[:,None],8)
 print(c['id'],a.shape,'horizontal',[(int(v[1]),int(v[3]),int(hor[int(v[1]):int(v[1]+v[3])].max(1).sum())) for v in stats[1:]])
 for v in stats[1:]:
  y0=int(v[1]);y1=y0+int(v[3]);r=max(16,round(.12*h));above=ink[max(0,y0-r-8):max(0,y0-8)];below=ink[min(h,y1+8):min(h,y1+r+8)]
  print('rule',y0,y1,'ink before/after',above.mean() if above.size else None,below.mean() if below.size else None)
