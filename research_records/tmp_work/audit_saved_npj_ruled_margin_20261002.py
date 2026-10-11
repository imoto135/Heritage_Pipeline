from pathlib import Path
import sys,json,zipfile
import cv2,numpy as np
sys.path.insert(0,str(Path('scripts/benchmark').resolve()))
from build_npj_v6_comparison import ROOT,read,sha,dump
from package_saved_npj_inference_v6_review import check
out=ROOT/'tmp_work/saved_npj_ruled_margin_pilot_20261002';m=check(out)
assert sha(out/'source_snapshot.py')==m['source_sha256']
records=[]
for c in m['cases']:
 d=out/'native'/c['id'];base=read(d/'saved_npj.png');assert np.array_equal(base,read(ROOT/c['saved_npj']))
 margin=read(d/'margin.png',True);mandatory=read(d/'mandatory.png',True);guard=read(d/'narrow_guard.png',True)
 for mode in ['margin','guarded']:
  update=read(d/(mode+'_update.png'),True);v=read(d/(mode+'.png'))
  assert not (update&(~margin|mandatory)).any()
  if mode=='guarded':assert not (update&guard).any()
  assert np.array_equal(base[~update],v[~update])
  if update.any():
   y,x=np.nonzero(update);a=v.astype(np.int16)
   residual=4*a[y,x]-a[y-1,x]-a[y+1,x]-a[y,x-1]-a[y,x+1]
   maxres=int(np.abs(residual).max());assert maxres<=4
  else:maxres=0
  records.append(dict(frame=c['id'],mode=mode,update=int(update.sum()),changed=int(np.any(v!=base,axis=2).sum()),rounded_equation_max_residual=maxres))
with zipfile.ZipFile(out/'comparison_pngs.zip') as z:
 assert z.testzip() is None
 assert len(z.namelist())==19
 for p in out.glob('*.png'):assert z.read(p.name)==p.read_bytes()
dump(ROOT/'tmp_work/saved_npj_ruled_margin_independent_audit_20261002.json',dict(status='passed',source_dependencies_outputs_hash_exact=True,rounded_harmonic_equation=True,zip_19_exact=True,records=records,pixel_GT=False,adopted=False))
print(json.dumps(records),flush=True)
