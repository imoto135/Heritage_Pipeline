"""Fixed native-resolution inference; saves masks without inventing ink pixels."""
import argparse
from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch
from PIL import Image
from synthetic_layers import OUT,ROOT,input_features,file_hash
from train_layer_selector import CONFIGS
from evaluate_synthetic_selectors import load_selector

@torch.no_grad()
def predict(model,rgb,mode,device):
    features=input_features(rgb,mode);_,h,w=features.shape
    side=256;stride=128
    ph=max(0,side-h);pw=max(0,side-w)
    if ph or pw:features=np.pad(features,((0,0),(0,ph),(0,pw)),mode='reflect')
    hh,ww=features.shape[1:]
    ys=list(range(0,hh-side+1,stride));xs=list(range(0,ww-side+1,stride))
    if ys[-1]!=hh-side:ys.append(hh-side)
    if xs[-1]!=ww-side:xs.append(ww-side)
    coords=[(y,x) for y in ys for x in xs]
    weight=np.outer(np.hanning(side),np.hanning(side)).astype(np.float32)
    weight=np.maximum(weight,.03)
    total=np.zeros((2,hh,ww),np.float32);norm=np.zeros((hh,ww),np.float32)
    for off in range(0,len(coords),8):
        batch_coords=coords[off:off+8]
        batch=np.stack([features[:,y:y+side,x:x+side] for y,x in batch_coords])
        z=model(torch.from_numpy(batch).to(device)).sigmoid().cpu().numpy()
        for (y,x),p in zip(batch_coords,z):
            total[:,y:y+side,x:x+side]+=p*weight
            norm[y:y+side,x:x+side]+=weight
    assert np.all(norm>0)
    return (total/norm[None])[:,:h,:w]

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=list(CONFIGS));p.add_argument('--device',default='cuda:0')
    p.add_argument('--roles',nargs='+',default=['development']);a=p.parse_args();cv2.setNumThreads(2);torch.set_num_threads(4)
    protocol=json.loads((OUT/'cases_frozen.json').read_text())
    pages={r['page']:r for r in protocol['cases'] if r['role'] in a.roles}
    dest=OUT/'native_predictions';dest.mkdir(exist_ok=True)
    setting=dict(checkpoint='fixed 12000-step last.pth',native=True,patch=256,overlap=128,
        averaging='Hann weight floor .03; float probabilities averaged before .5 threshold',
        paper='Input-derived whole-page morphology; no known clean paper or GT',
        overlay='These files are predictions only; genuine restoration separate',
        source_sha256=file_hash(__file__),cases_sha256=file_hash(OUT/'cases_frozen.json'))
    pp=dest/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==setting,'Changed inference requires separate version'
    else:pp.write_text(json.dumps(setting,ensure_ascii=False,indent=2))
    for name in a.names:
        path=OUT/'learned_trials'/f'{name}_seed413'/'last.pth'
        assert (path.parent/'complete.json').exists()
        model,cfg,ck=load_selector(path,a.device);assert ck['step']==12000
        for page,r in pages.items():
            d=dest/name/page;d.mkdir(parents=True,exist_ok=True);done=d/'complete.json'
            if done.exists():
                old=json.loads(done.read_text());assert old['checkpoint_sha256']==file_hash(path)
                assert old['source_sha256']==r['source_sha256']
                for f,sha in old['outputs'].items():assert file_hash(d/f)==sha
                continue
            start=time.monotonic();assert file_hash(ROOT/r['original'])==r['source_sha256']
            rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'))
            prob=predict(model,rgb,cfg['mode'],a.device);assert prob.shape==(2,*rgb.shape[:2])
            assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
            for i,layer in enumerate(['mark','stain']):
                mask=prob[i]>=.5
                Image.fromarray(mask.astype(np.uint8)*255).save(d/(layer+'_mask.png'))
                Image.fromarray(np.rint(prob[i]*255).astype(np.uint8)).save(d/(layer+'_probability_preview.png'))
            record=dict(page=page,book=r['book'],name=name,role=r['role'],original=r['original'],
                source_sha256=r['source_sha256'],checkpoint_sha256=file_hash(path),seconds=time.monotonic()-start,
                shape=list(rgb.shape),mark_pixels=int((prob[0]>=.5).sum()),stain_pixels=int((prob[1]>=.5).sum()),
                overlapping_predicted_layers=int(((prob[0]>=.5)&(prob[1]>=.5)).sum()),
                mark_probability_percentiles=np.percentile(prob[0],[0,25,50,75,95,99,100]).tolist(),
                no_pixel_GT=True,probability_preview_quantized_not_used_for_masks=True,
                outputs={f.name:file_hash(f) for f in d.glob('*.png')})
            done.write_text(json.dumps(record,ensure_ascii=False,indent=2))
            print('native',name,page,'mark',record['mark_pixels'],'stain',record['stain_pixels'],round(record['seconds'],1),'sec',flush=True)
        del model

if __name__=='__main__':main()
