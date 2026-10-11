"""Native predictions from fixed tentative sparse-supervised classifiers."""
import argparse
import json
import pickle
import time
import cv2
import numpy as np
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from fit_sparse_color_context import feature_maps

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['color_logistic','color_forest','spatial_logistic','spatial_forest'])
    p.add_argument('--roles',nargs='+',default=['development']);a=p.parse_args();cv2.setNumThreads(2)
    cases=json.loads((OUT/'cases_frozen.json').read_text());pages={r['page']:r for r in cases['cases'] if r['role'] in a.roles}
    records=json.loads((OUT/'sparse_context_trials/models.json').read_text())
    models={r['name']:(pickle.loads((OUT/'sparse_context_trials'/(r['name']+'.pkl')).read_bytes()),r) for r in records if r['name'] in a.names}
    for page,r in pages.items():
        rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'));h,w=rgb.shape[:2]
        assert file_hash(ROOT/r['original'])==r['source_sha256']
        for spatial in [False,True]:
            selected=[(n,m,info) for n,(m,info) in models.items() if info['spatial']==spatial]
            if not selected:continue
            maps,names=feature_maps(rgb,spatial)
            for name,model,info in selected:
                start=time.monotonic();dest=OUT/'native_predictions'/name/page;dest.mkdir(parents=True,exist_ok=True)
                assert info['features']==names
                done=dest/'complete.json'
                if done.exists():
                    prior=json.loads(done.read_text());assert prior['model_sha256']==info['model_sha256']
                    for filename,sha in prior['outputs'].items():assert file_hash(dest/filename)==sha
                    continue
                prob=np.zeros((h,w),np.float32)
                for y in range(0,h,96):
                    features=np.stack([m[y:y+96].reshape(-1) for m in maps],1)
                    prob[y:y+96]=model.predict_proba(features)[:,1].reshape(min(96,h-y),w)
                assert np.isfinite(prob).all()
                mask=prob>=.5;Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mark_mask.png')
                Image.fromarray(np.rint(prob*255).astype(np.uint8)).save(dest/'mark_probability_preview.png')
                record=dict(page=page,name=name,book=r['book'],role=r['role'],original=r['original'],
                    source_sha256=r['source_sha256'],model_sha256=info['model_sha256'],threshold=.5,
                    script_sha256=file_hash(__file__),mark_pixels=int(mask.sum()),seconds=time.monotonic()-start,
                    training='92 tentative original-development points; no validation labels used',
                    classifier_has_no_stain_head=True,no_full_pixel_GT=True,
                    outputs={f.name:file_hash(f) for f in dest.glob('*.png')})
                done.write_text(json.dumps(record,ensure_ascii=False,indent=2))
                print('sparse native',name,page,int(mask.sum()),round(record['seconds'],1),'sec',flush=True)

if __name__=='__main__':main()
