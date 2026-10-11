"""Frozen local CSC v2 for independent inputs; never label it authentic NPJ reproduction."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from prepare_fullpage_baselines import module

def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();cv2.setNumThreads(2)
 protocol=OUT/'independent_evaluation_protocol_v2.json';pr=json.loads(protocol.read_text());pages={r['page']:r for r in json.loads((OUT/pr['unified_cases']).read_text())['cases']}
 v2path=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2/scripts/csc.py';acepath=ROOT/'modules/ddrm_npj/scripts/ace.py';v2=module('independent_fixed_csc_v2',v2path);ace=module('independent_fixed_ace',acepath)
 dest=OUT/'independent_predictions/csc_v2';dest.mkdir(parents=True,exist_ok=True)
 for idx,(page,r) in enumerate(pages.items()):
  if idx%a.shards!=a.shard:continue
  d=dest/page;d.mkdir(exist_ok=True);done=d/'complete.json'
  if done.exists():
   rec=json.loads(done.read_text());assert rec['source_sha256']==r['source_sha256']
   for f,sha in rec['outputs'].items():assert file_hash(d/f)==sha
   continue
  start=time.monotonic();source=ROOT/r['original'];assert file_hash(source)==r['source_sha256'];rgb=np.asarray(Image.open(source).convert('RGB'));cached=OUT/'fullpage_baselines'/page/'complete.json';cacheorigin=False
  if cached.exists():
   cr=json.loads(cached.read_text());assert cr['source_sha256']==r['source_sha256'];masks={k:np.asarray(Image.open(cached.parent/('csc_v2_'+k+'.png')).convert('L'))>127 for k in ['red','black','stain']};cacheorigin=True
  else:
   bgr=rgb[:,:,::-1].copy();print('CSC input',page,flush=True);eq=ace.automatic_color_equalization(bgr,alpha=5.,n_samples=200,seed=0);cv2.imwrite(str(d/'ace.png'),eq);c=v2.color_space_clustering(eq,seed=42,max_samples=50000,border_margin_ratio=.04);masks={'red':c.red_text_mask>127,'black':c.black_text_mask>127,'stain':c.degradation_mask>127}
  for k,m in masks.items():assert m.shape==rgb.shape[:2];Image.fromarray(m.astype(np.uint8)*255).save(d/(k+'_mask.png'))
  rec=dict(page=page,source_sha256=r['source_sha256'],protocol_sha256=file_hash(protocol),script_sha256=file_hash(__file__),csc_source_sha256=file_hash(v2path),ace_source_sha256=file_hash(acepath),fixed_settings={'ACE_alpha':5.,'ACE_n_samples':200,'ACE_seed':0,'CSC_seed':42,'max_samples':50000,'border_margin_ratio':.04},local_v2_not_verified_original_paper=True,reused_verified_source_identical_cache=cacheorigin,counts={k:int(m.sum()) for k,m in masks.items()},seconds=time.monotonic()-start,outputs={f.name:file_hash(f) for f in d.glob('*.png')});done.write_text(json.dumps(rec,indent=2));print('Independent CSC',page,round(rec['seconds'],1),flush=True)
 (dest/('shard%d_complete.json'%a.shard)).write_text(json.dumps(dict(shard=a.shard,shards=a.shards,pages=list(pages))))
if __name__=='__main__':main()
