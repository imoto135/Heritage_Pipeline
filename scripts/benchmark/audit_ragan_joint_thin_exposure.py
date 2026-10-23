"""Audit fine-stroke representation; never promote held-out labels into training."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from compare_v5_aligned_fade_control import dump,sha
import diagnose_ragan_discriminator as d
OUT=ROOT/'tmp_work/ragan_joint_replay_20261022'
def main():
 patch='200017197_00172_0_p00'
 rows=[json.loads(s) for s in (ROOT/'tmp_work/makura_contact_stain_train_control_20261022/ragan_contact/val_manifest.jsonl').read_text().splitlines()]
 row=next(r for r in rows if r['patch_id']==patch)
 tracepath=ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png');trace=np.asarray(Image.open(tracepath).convert('L'))>0
 paths={k:ROOT/row['labels'][k] for k in ['target','valid']};target=np.asarray(Image.open(paths['target']).convert('L'))<128;valid=np.asarray(Image.open(paths['valid']).convert('L'))>0
 a=dict(patch=patch,trace_n=int(trace.sum()),known_ink=int((trace&target&valid).sum()),known_paper=int((trace&~target&valid).sum()),unknown=int((trace&~valid).sum()),trace_sha256=sha(tracepath),label_hashes={k:sha(p) for k,p in paths.items()},interpretation='source-only trace versus provisional held-out labels; not used for G updates')
 index=json.loads((d.OUT/'index.json').read_text())['rows']
 row=next(r for r in index if r['patch']==patch and r['split']=='val');x,y=row['crop_xy'];aa=dict(np.load(d.OUT/row['cache']));sub=trace[y:y+256,x:x+256]
 a.update(native256_crop_xy=[x,y],trace_in_native256_crop=int(sub.sum()),trace_valid_in_native256_crop=int((sub&aa['valid']).sum()))
 dump(OUT/'fine_trace_label_audit.json',a)
 reviews=ROOT/'tmp_work/makura_additional_thin_cores_20261021';selected={r['patch']:r for r in index if r['split']=='train'};out=[]
 for r in json.loads((reviews/'audit.json').read_text())['rows']:
  patch=r['patch'];path=reviews/'masks'/(patch+'_trace.png');trace=np.asarray(Image.open(path).convert('L'))>0
  row=dict(patch=patch,trace_n=int(trace.sum()),patch_in_fixed24=patch in selected,trace_in_crop=0,trace_valid_ink_in_crop=0,mask_sha256=sha(path))
  if patch in selected:
   entry=selected[patch];x,y=entry['crop_xy'];sub=trace[y:y+256,x:x+256];aa=dict(np.load(d.OUT/entry['cache']))
   row.update(trace_in_crop=int(sub.sum()),trace_valid_ink_in_crop=int((sub&aa['valid']&(aa['target']<.5)).sum()),crop_xy=[x,y])
  out.append(row)
 dump(OUT/'training_reviewed_thin_exposure_audit.json',dict(rows=out,previous_review_set='five additional training thin traces; not all possible fine text',interpretation='fixed24 subset does not retain full reviewed fine-stroke exposure of previous full-data training'))
 print('Held-out trace:',a['known_ink'],'ink',a['unknown'],'unknown. Training reviewed traces in fixed crops:',sum(r['trace_in_crop'] for r in out))
if __name__=='__main__':main()
