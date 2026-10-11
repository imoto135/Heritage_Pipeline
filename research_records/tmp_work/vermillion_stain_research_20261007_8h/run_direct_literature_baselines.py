"""Fixed original-only direct baselines; retain channel and tiling provenance."""
import argparse,json,time,sys,shutil
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_npj_literature_models import load,prompts,EXTERNAL,SETTINGS

def invoke(model,a,method,device,half=False):
    # StainRestorer author expects RGB; DocRes author expects OpenCV BGR.
    t=torch.from_numpy((a[:,:,::-1].copy() if method=='stain' else a).transpose(2,0,1).copy())[None].to(device).float()/255
    if half:t=t.half()
    with torch.inference_mode():
        p=model(t)
        if method=='stain':p=p[0]
        elif method=='docres_binarization':p=torch.softmax(p[:,:2],1)[:,1:2]
        p=p.clamp(0,1);assert torch.isfinite(p).all();v=p[0].permute(1,2,0).float().cpu().numpy()
    return v[:,:,::-1].copy() if method=='stain' else v

def tiles(model,a,method,device):
    h,w=a.shape[:2];core=384;halo=64;pad=cv2.copyMakeBorder(a,halo,halo+core,halo,halo+core,cv2.BORDER_REFLECT_101)
    channels=1 if method=='docres_binarization' else 3;result=np.empty((h,w,channels),np.float32);coverage=np.zeros((h,w),np.uint8)
    for y in range(0,h,core):
        for x in range(0,w,core):
            hh,ww=min(core,h-y),min(core,w-x);v=invoke(model,pad[y:y+512,x:x+512],method,device)
            result[y:y+hh,x:x+ww]=v[halo:halo+hh,halo:halo+ww];coverage[y:y+hh,x:x+ww]+=1
    assert np.all(coverage==1);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['stain','docres_binarization','docres_appearance_native','docres_appearance_author'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--external',action='store_true');a=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4)
    raw=json.loads((OUT/'cases_frozen.json').read_text())['cases'];pages={r['page']:{**r,'file':str(ROOT/r['original'])} for r in raw if r['role'] in a.roles}
    if a.external:
        for r in json.loads((OUT/'external_originals/frozen_cases/cases_frozen.json').read_text())['cases']:pages['external_'+r['page']]={**r,'file':str(OUT/r['original'])}
    method=a.method;dest=OUT/'direct_literature_baselines'/method;dest.mkdir(parents=True,exist_ok=True)
    model,resources=load('stain' if method=='stain' else 'docres',a.device)
    pns=prompts() if method!='stain' else None
    author=method=='docres_appearance_author'
    if author:model=model.half()
    protocol=dict(method=method,resources={str(f):file_hash(f) for f in resources},original_inputs_only=True,
        source_sha256=file_hash(__file__),adapter_source_sha256=file_hash(ROOT/'scripts/benchmark/run_npj_literature_models.py'),
        docres_weight_provenance=SETTINGS['provenance']['docres'],stain_weight='Author StainDoc normal-stain release, not Seal/Mark-removal weights',
        tiling='512 native tiles, halo64, disjoint384 cores; complete original-size prompt' if not author else 'Author appearance geometry: 1600x1600 global float16 forward then ratio-map interpolation',
        exact_author_benchmark=False,GT=False,not_adopted=True,
        color='Stain RGB tensor -> BGR array -> RGB PNG; DocRes BGR tensor/output -> RGB PNG',
        interpretation='DocRes binarization is a foreground candidate, not color restoration')
    pp=dest/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==protocol
    else:pp.write_text(json.dumps(protocol,ensure_ascii=False,indent=2));shutil.copy2(__file__,dest/'source_snapshot.py')
    for page,r in pages.items():
        d=dest/page;d.mkdir(exist_ok=True);done=d/'complete.json'
        if done.exists():
            old=json.loads(done.read_text());assert old['source_sha256']==r['source_sha256']
            for f,sha in old['outputs'].items():assert file_hash(d/f)==sha
            continue
        assert file_hash(r['file'])==r['source_sha256'];rgb=np.asarray(Image.open(r['file']).convert('RGB'));bgr=rgb[:,:,::-1].copy();h,w=rgb.shape[:2];start=time.monotonic()
        if method=='stain':inp=bgr
        else:
            prompt=pns['binarization_promptv2'](bgr) if method=='docres_binarization' else pns['appearance_prompt'](bgr)
            Image.fromarray(prompt[:,:,::-1]).save(d/'prompt.png');inp=np.concatenate([bgr,prompt],2)
        if author:
            # All selected native pages exceed1600. Match the published global
            # branch and uint8 truncation before ratio-map interpolation.
            assert max(w,h)>=1600
            small=cv2.resize(inp,(1600,1600));v=invoke(model,small,method,a.device,half=True)
            pred=(v*255).astype(np.uint8);pred[pred==0]=1
            ratio=cv2.resize(bgr,(1600,1600)).astype(float)/pred.astype(float)
            ratio=cv2.resize(ratio,(w,h));ratio[ratio==0]=.00001
            result=np.clip(bgr.astype(float)/ratio,0,255).astype(np.uint8)
        else:
            v=tiles(model,inp,method,a.device)
            if method=='docres_binarization':
                np.save(d/'background_probability.npy',v[:,:,0]);Image.fromarray((v[:,:,0]<.5).astype(np.uint8)*255).save(d/'ink_mask.png')
                result=np.repeat((v[:,:,0]>=.5)[:,:,None]*255,3,2).astype(np.uint8)
            else:result=np.rint(v*255).astype(np.uint8)
        Image.fromarray(result[:,:,::-1]).save(d/'result.png')
        rec=dict(page=page,book=r['book'],role=r['role'],original=r['file'],source_sha256=r['source_sha256'],method=method,
            seconds=time.monotonic()-start,shape=list(rgb.shape),finite=True,protocol_sha256=file_hash(pp),
            source_pixel_GT=False,inputs_have_no_masks_or_annotations=True,outputs={f.name:file_hash(f) for f in d.iterdir() if f.is_file()})
        done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('Direct baseline',method,page,round(rec['seconds'],1),'sec',flush=True)
    (dest/'queue_complete.json').write_text(json.dumps(dict(pages=list(pages),method=method),indent=2))

if __name__=='__main__':main()
