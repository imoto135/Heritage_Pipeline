"""Strong direct restorers on frozen independent cases with source-identical cache reuse."""
import argparse,json,time,shutil
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from run_direct_literature_baselines import invoke,tiles,load,prompts,SETTINGS

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--method',required=True,choices=['stain','stain_global256','stain_global512','docres_binarization','docres_appearance_native','docres_appearance_author']);a=p.parse_args();cv2.setNumThreads(2);torch.set_num_threads(4)
 protocol=OUT/'independent_evaluation_protocol_v2.json';pr=json.loads(protocol.read_text());pages={r['page']:r for r in json.loads((OUT/pr['unified_cases']).read_text())['cases']};method=a.method;stain=method.startswith('stain');author=method=='docres_appearance_author';globalstain=method.startswith('stain_global')
 dest=OUT/'independent_direct'/method;dest.mkdir(parents=True,exist_ok=True);model=None;pns=prompts() if not stain else None
 resources=None
 for page,r in pages.items():
  d=dest/page;d.mkdir(exist_ok=True);done=d/'complete.json'
  if done.exists():
   rec=json.loads(done.read_text());assert rec['source_sha256']==r['source_sha256']
   for f,sha in rec['outputs'].items():assert file_hash(d/f)==sha
   continue
  start=time.monotonic();cached=OUT/'direct_literature_baselines'/method/page/'complete.json';reuse=False
  if cached.exists():
   old=json.loads(cached.read_text());assert old['source_sha256']==r['source_sha256']
   cached_outputs=old.get('outputs')
   if cached_outputs is None:
    assert file_hash(cached.parent/'result.png')==old['result_sha256'];cached_outputs={f.name:file_hash(f) for f in cached.parent.glob('*.png')}
   for f,sha in cached_outputs.items():
    assert file_hash(cached.parent/f)==sha;shutil.copy2(cached.parent/f,d/f)
   reuse=True;resourceprovenance={'cached_complete_sha256':file_hash(cached),'prior_protocol_sha256':file_hash(cached.parent.parent/'protocol.json')}
  else:
   if model is None:
    model,resources=load('stain' if stain else 'docres',a.device)
    if author:model=model.half()
   assert file_hash(ROOT/r['original'])==r['source_sha256'];rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));bgr=rgb[:,:,::-1].copy();h,w=rgb.shape[:2]
   if stain:inp=bgr
   else:
    prompt=pns['binarization_promptv2'](bgr) if method=='docres_binarization' else pns['appearance_prompt'](bgr);Image.fromarray(prompt[:,:,::-1]).save(d/'prompt.png');inp=np.concatenate([bgr,prompt],2)
   if author:
    assert max(w,h)>=1600;v=invoke(model,cv2.resize(inp,(1600,1600)),method,a.device,half=True);pred=(v*255).astype(np.uint8);pred[pred==0]=1;ratio=cv2.resize(bgr,(1600,1600)).astype(float)/pred.astype(float);ratio=cv2.resize(ratio,(w,h));ratio[ratio==0]=.00001;result=np.clip(bgr.astype(float)/ratio,0,255).astype(np.uint8)
   elif globalstain:
    side=256 if method=='stain_global256' else 512;v=invoke(model,cv2.resize(inp,(side,side),interpolation=cv2.INTER_LINEAR),'stain',a.device);low=np.rint(v*255).astype(np.uint8);Image.fromarray(low[:,:,::-1]).save(d/'model_output_RGB.png');result=cv2.resize(low,(w,h),interpolation=cv2.INTER_LINEAR)
   else:
    v=tiles(model,inp,method,a.device)
    if method=='docres_binarization':np.save(d/'background_probability.npy',v[:,:,0]);Image.fromarray((v[:,:,0]<.5).astype(np.uint8)*255).save(d/'ink_mask.png');result=np.repeat((v[:,:,0]>=.5)[:,:,None]*255,3,2).astype(np.uint8)
    else:result=np.rint(v*255).astype(np.uint8)
   Image.fromarray(result[:,:,::-1]).save(d/'result.png');resourceprovenance={'resources':{str(f):file_hash(f) for f in resources}}
  rec=dict(page=page,source_sha256=r['source_sha256'],method=method,role=r['role'],book=r['book'],protocol_sha256=file_hash(protocol),script_sha256=file_hash(__file__),adapter_sha256=file_hash(OUT/'run_direct_literature_baselines.py'),source_identical_cache=reuse,resource_provenance=resourceprovenance,seconds=time.monotonic()-start,channels='Stain input RGB; internal array BGR; PNG RGB; DocRes BGR tensor',exact_author_benchmark=False,docres_mirror_provenance=SETTINGS['provenance']['docres'],no_pixel_or_reading_GT=True,outputs={f.name:file_hash(f) for f in d.iterdir() if f.is_file() and f.name!='complete.json'});done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('Independent direct',method,page,round(rec['seconds'],1),'cache' if reuse else 'new',flush=True)
 (dest/'queue_complete.json').write_text(json.dumps(dict(method=method,pages=list(pages))))
if __name__=='__main__':main()
