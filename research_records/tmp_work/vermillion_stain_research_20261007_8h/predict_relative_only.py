"""Relative-only deployment, with explicit correct loader and book exclusion guards."""
import argparse,json,time
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from relative_only_selector import load_relative_only
from predict_native_selectors import predict

def main():
 p=argparse.ArgumentParser();p.add_argument('--exclude-dev-book');p.add_argument('--device',default='cuda:1');p.add_argument('--independent',action='store_true');a=p.parse_args();cv2.setNumThreads(2);torch.set_num_threads(4);suffix='_exclude'+a.exclude_dev_book if a.exclude_dev_book else ''
 trial=OUT/'native_partial_trials'/('dense_native_relative_only_seed715'+suffix);assert (trial/'complete.json').exists();path=trial/'last.pth';model,cfg,ck=load_relative_only(path,a.device);assert ck['step']==8000;proto=json.loads((trial/'protocol.json').read_text())
 raw=json.loads((OUT/('independent_cases_v2.json' if a.independent else 'cases_frozen.json')).read_text())['cases'];pages={r['page']:r for r in raw if a.independent or r['role']=='development'}
 if a.exclude_dev_book:pages={p:r for p,r in pages.items() if r['book']==a.exclude_dev_book}
 method='partial_dense_native_relative_only'+suffix;dest=OUT/('independent_predictions' if a.independent else 'native_predictions')/method
 for page,r in pages.items():
  if a.exclude_dev_book:assert r['book'] not in proto['actual_training_books']
  d=dest/page;d.mkdir(parents=True,exist_ok=True);done=d/'complete.json'
  if done.exists():
   rec=json.loads(done.read_text());assert rec['resource_sha256']==file_hash(path)
   for f,sha in rec['outputs'].items():assert file_hash(d/f)==sha
   continue
  start=time.monotonic();assert file_hash(ROOT/r['original'])==r['source_sha256'];rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));prob=predict(model,rgb,'relative',a.device)
  for i,k in enumerate(['mark','stain']):Image.fromarray((prob[i]>=.5).astype(np.uint8)*255).save(d/(k+'_mask.png'));Image.fromarray(np.rint(prob[i]*255).astype(np.uint8)).save(d/(k+'_probability_preview.png'))
  rec=dict(name=method,page=page,book=r['book'],role=r['role'],source_sha256=r['source_sha256'],resource_sha256=file_hash(path),checkpoint_sha256=file_hash(path),threshold=.5,loader='RelativeOnlySelector: absoluteRGB/paper channels zeroed for every forward',loader_sha256=file_hash(OUT/'relative_only_selector.py'),training_protocol_sha256=file_hash(trial/'protocol.json'),registered_design_sha256=file_hash(OUT/'relative_only_ablation_registered.json'),script_sha256=file_hash(__file__),excluded_development_books=proto['excluded_development_books'],no_pixel_or_reading_GT=True,seconds=time.monotonic()-start,outputs={f.name:file_hash(f) for f in d.glob('*.png')});done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('Relative-only',page,suffix,round(rec['seconds'],1),flush=True)
if __name__=='__main__':main()
