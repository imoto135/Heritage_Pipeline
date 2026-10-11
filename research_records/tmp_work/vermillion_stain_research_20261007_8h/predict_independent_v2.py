"""Frozen LAST checkpoints on independent local/prospective/external originals."""
import argparse,json,time,pickle
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from evaluate_synthetic_selectors import load_selector
from predict_native_selectors import predict
from fit_sparse_color_context import feature_maps
from prepare_fullpage_baselines import heuristic_masks,sauvola

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');p.add_argument('--models',nargs='*');p.add_argument('--classical-only',action='store_true');a=p.parse_args();cv2.setNumThreads(2);torch.set_num_threads(4)
 protofile=OUT/'independent_evaluation_protocol_v2.json';proto=json.loads(protofile.read_text());manifest=OUT/proto['unified_cases'];assert file_hash(manifest)==proto['unified_cases_sha256']
 pages={r['page']:r for r in json.loads(manifest.read_text())['cases']};dest=OUT/'independent_predictions';dest.mkdir(exist_ok=True)
 models={} if a.classical_only else {n:r for n,r in proto['model_resources'].items() if not a.models or n in a.models}
 for name,resource in models.items():
  path=ROOT/resource['path'];assert file_hash(path)==resource['sha256'];forest=path.suffix=='.pkl'
  if forest:model=pickle.loads(path.read_bytes());cfg={'mode':'spatial'}
  else:model,cfg,ck=load_selector(path,a.device);assert ck['step']==resource['step']
  for page,r in pages.items():
   d=dest/name/page;d.mkdir(parents=True,exist_ok=True);done=d/'complete.json'
   if done.exists():
    old=json.loads(done.read_text());assert old['resource_sha256']==resource['sha256'] and old['source_sha256']==r['source_sha256']
    for f,sha in old['outputs'].items():assert file_hash(d/f)==sha
    continue
   start=time.monotonic();assert file_hash(ROOT/r['original'])==r['source_sha256'];rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));h,w=rgb.shape[:2]
   if forest:
    maps,names=feature_maps(rgb,True);prob=np.zeros((1,h,w),np.float32)
    for y in range(0,h,96):prob[0,y:y+96]=model.predict_proba(np.stack([m[y:y+96].reshape(-1) for m in maps],1))[:,1].reshape(min(96,h-y),w)
   else:prob=predict(model,rgb,cfg['mode'],a.device)
   assert np.isfinite(prob).all()
   for i,kind in enumerate(['mark'] if forest else ['mark','stain']):
    Image.fromarray((prob[i]>=.5).astype(np.uint8)*255).save(d/(kind+'_mask.png'));Image.fromarray(np.rint(prob[i]*255).astype(np.uint8)).save(d/(kind+'_probability_preview.png'))
   rec=dict(name=name,page=page,book=r['book'],role=r['role'],source_sha256=r['source_sha256'],resource_sha256=resource['sha256'],protocol_sha256=file_hash(protofile),script_sha256=file_hash(__file__),native_inference_source_sha256=file_hash(OUT/'predict_native_selectors.py'),threshold=.5,mark_pixels=int((prob[0]>=.5).sum()),seconds=time.monotonic()-start,not_for_training=True,no_pixel_or_reading_GT=True,outputs={f.name:file_hash(f) for f in d.glob('*.png')});done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('Independent',name,page,round(rec['seconds'],1),flush=True)
  del model
 # Frozen source-only baselines; no adaptations to these pages.
 for page,r in pages.items():
  d=dest/'source_heuristics'/page;d.mkdir(parents=True,exist_ok=True);done=d/'complete.json'
  if done.exists():continue
  rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));masks=heuristic_masks(rgb);masks['sauvola_black']=sauvola(rgb)
  for name,mask in masks.items():Image.fromarray(mask.astype(np.uint8)*255).save(d/(name+'.png'))
  done.write_text(json.dumps(dict(page=page,source_sha256=r['source_sha256'],protocol_sha256=file_hash(protofile),source_sha256_code=file_hash(OUT/'prepare_fullpage_baselines.py'),script_sha256=file_hash(__file__),counts={n:int(m.sum()) for n,m in masks.items()},outputs={f.name:file_hash(f) for f in d.glob('*.png')}),indent=2));print('Independent heuristics',page,flush=True)
 (dest/('classical_complete.json' if a.classical_only else 'queue_complete.json')).write_text(json.dumps(dict(pages=list(pages),models=list(models),protocol_sha256=file_hash(protofile)),indent=2))
if __name__=='__main__':main()
