from pathlib import Path
import sys,importlib.util,json,time
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tmp_work/npj_contact_research_20261007'))
from common import read,write,dump,sha
from render_full import records
def main():
 cv2.setNumThreads(2)
 p=ROOT/'modules/ddrm_npj/scripts/csc.py';spec=importlib.util.spec_from_file_location('audit_old_csc',p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 summary=[]
 boxes={'T01':[812,421,450,450],'N01':[428,347,190,220],'N02':[520,890,520,560]}
 for c in records():
  id=c['id']
  if id not in boxes:continue
  md=ROOT/c['source']['npj_black'];ip=md.parent/'00_input.png';ace=cv2.imread(str(ip));start=time.monotonic();r=module.color_space_clustering(ace)
  dest=OUT/'mask_audit'/id;dest.mkdir(parents=True,exist_ok=True)
  fresh={'black':r.black_text_mask>127,'red':r.red_text_mask>127,'degradation':r.degradation_mask>127,'background':r.background_mask>127}
  v2=Path(str(md.parent).replace('split_dataset_csc_mask/','split_dataset_csc_mask_v2/'))
  stored={'black':cv2.imread(str(md),0)<127,'red':cv2.imread(str(md.parent/'03_text_red.png'),0)<127,'degradation':cv2.imread(str(md.parent/'04_degradation.png'),0)>127,'background':cv2.imread(str(md.parent/'05_background.png'),0)>127}
  x,y,w,h=boxes[id];roi=np.s_[y:y+h,x:x+w];stats={}
  for role,mask in fresh.items():
   write(dest/(role+'_regenerated.png'),mask*255);stats[role]=dict(stored=int(stored[role].sum()),regenerated=int(mask.sum()),changed=int(np.count_nonzero(mask!=stored[role])),ROI_stored=int(stored[role][roi].sum()),ROI_regenerated=int(mask[roi].sum()))
  raw=read(ROOT/c['source']['original']);a=read(ip);assert raw.shape==a.shape
  v2b=cv2.imread(str(v2/'02_text_black.png'),0)<127;v2r=cv2.imread(str(v2/'03_text_red.png'),0)<127
  stats['v2_ROI']=dict(black=int(v2b[roi].sum()),red=int(v2r[roi].sum()))
  summary.append(dict(id=id,page=c['page'],ROI=boxes[id],stats=stats,input_equals_ACE=True,input_raw_MAE_255=float(np.abs(raw.astype(float)-a).mean()),source=str(ip.relative_to(ROOT)),source_sha256=sha(ip),csc_source_sha256=sha(p),seconds=time.monotonic()-start,material_GT=False))
  print('mask audit',id,stats,flush=True)
 dump(OUT/'mask_audit/summary.json',summary)
if __name__=='__main__':main()
