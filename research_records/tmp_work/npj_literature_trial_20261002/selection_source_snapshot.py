"""Freeze authentic npj frames and evaluation ROIs before external-model outputs."""
import json,shutil
from pathlib import Path
from build_npj_v6_comparison import ROOT,read,write,sha,dump
from package_saved_npj_inference_v6_review import check
OUT=ROOT/'tmp_work/npj_literature_trial_20261002'
FOLLOW=ROOT/'tmp_work/saved_npj_paper_only_followup_pilot_20261002'
OLD=ROOT/'tmp_work/saved_npj_inference_v6_pilot_20261002'
SELECTION=ROOT/'tmp_work/saved_npj_paper_followup_selection_20261002/selection.json'

def main():
 assert not (OUT/'selection.json').exists(),'Selection fixed already'
 OUT.mkdir(parents=True,exist_ok=True);check(FOLLOW);old=check(OLD)
 cases=json.loads(SELECTION.read_text())['cases']
 frames=[];deps={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SELECTION]}
 for c in cases:
  frames.append(dict(id=c['id'],page=c['page'],input=c['input'],original=c['reference_original'],box=c['reference_xywh'],
   saved_npj=c['saved_npj'],baseline_label=c['baseline_label'],roles=str((FOLLOW/'native'/c['id']).relative_to(ROOT)),
   regions=c['rois'],illustration=False))
 for c in old['frames']:
  fid='old_'+c['id'];regions=[dict(id='old_'+r['id'],title=r['title'],xywh=r['xywh']) for r in old['regions'] if r['frame']==c['id']]
  frames.append(dict(id=fid,page=c['page'],input=c['saved_input'],original=c['original'],box=c['frame_xywh'],
   saved_npj=c['saved_npj'],baseline_label=c['baseline_label'],roles=str((OLD/'native'/c['id']).relative_to(ROOT)),regions=regions,illustration=False))
 for folder in [FOLLOW,OLD]:
  for name in ['manifest.json','complete.json']:deps[str((folder/name).relative_to(ROOT))]=sha(folder/name)
 for c in frames:
  a=read(ROOT/c['original'])
  if c['box']:
   x,y,w,h=c['box'];a=a[y:y+h,x:x+w]
  n=read(ROOT/c['saved_npj']);observed=read(ROOT/c['input']);assert a.shape==n.shape==observed.shape
  dest=OUT/'inputs'/c['id'];write(dest/'original.png',a);write(dest/'saved_input.png',observed);write(dest/'saved_npj.png',n)
  for k in ['edit','protect','hold','red','extra']:
   p=ROOT/c['roles']/(k+'.png');v=read(p,True);assert v.shape==a.shape[:2];write(dest/(k+'.png'),v);deps[str(p.relative_to(ROOT))]=sha(p)
  for k in ['original','input','saved_npj']:deps[c[k]]=sha(ROOT/c[k])
  c['input_reference_exact']=bool((a==observed).all());c['shape']=list(a.shape)
 selection=dict(frames=frames,dependencies=deps,scope='6 saved frames from 5 pages in known development books; not 6 independent documents',
  ROIs=sum(len(c['regions']) for c in frames),selection_before_new_model_output=True,original_coordinates_only_display=True,illustrations=False,
  methods=['Background/SWT/paired-edge inspired structural diagnostic, not exact Xiong/SSP reproduction','StainRestorer StainDoc weight','DocRes binarization'])
 dump(OUT/'selection.json',selection);shutil.copyfile(__file__,OUT/'selection_source_snapshot.py')
 dump(OUT/'selection_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'inputs').rglob('*')) if p.is_file()},selection_sha=sha(OUT/'selection.json')))
 print('Fixed',len(frames),'frames',selection['ROIs'],'ROIs',flush=True)
if __name__=='__main__':main()
