"""Native masks from fixed-final texture-transfer trials."""
import argparse,json,time
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from evaluate_synthetic_selectors import load_selector
from predict_native_selectors import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--exclude-dev-book',default=None);a=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4)
    cases=json.loads((OUT/'cases_frozen.json').read_text())['cases'];pages={r['page']:r for r in cases if r['role'] in a.roles}
    if a.exclude_dev_book:pages={p:r for p,r in pages.items() if r['book']==a.exclude_dev_book}
    for name in a.names:
        suffix='_exclude'+a.exclude_dev_book if a.exclude_dev_book else ''
        trial=OUT/'texture_transfer_trials'/f'{name}_seed614{suffix}';assert (trial/'complete.json').exists()
        path=trial/'last.pth';model,cfg,ck=load_selector(path,a.device);assert ck['step']==10000
        method='texture_'+name+suffix
        for page,r in pages.items():
            dest=OUT/'native_predictions'/method/page;dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json'
            if done.exists():
                old=json.loads(done.read_text());assert old['checkpoint_sha256']==file_hash(path)
                for f,sha in old['outputs'].items():assert file_hash(dest/f)==sha
                continue
            start=time.monotonic();assert file_hash(ROOT/r['original'])==r['source_sha256'];rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'))
            prob=predict(model,rgb,'relative',a.device)
            for i,layer in enumerate(['mark','stain']):
                Image.fromarray((prob[i]>=.5).astype(np.uint8)*255).save(dest/(layer+'_mask.png'))
                Image.fromarray(np.rint(prob[i]*255).astype(np.uint8)).save(dest/(layer+'_probability_preview.png'))
            rec=dict(name=method,page=page,role=r['role'],source_sha256=r['source_sha256'],checkpoint_sha256=file_hash(path),
                threshold=.5,final_steps=10000,training_protocol_sha256=file_hash(trial/'protocol.json'),
                native_inference_source_sha256=file_hash(OUT/'predict_native_selectors.py'),script_sha256=file_hash(__file__),
                mark_pixels=int((prob[0]>=.5).sum()),seconds=time.monotonic()-start,real_full_pixel_GT=False,
                outputs={f.name:file_hash(f) for f in dest.glob('*.png')})
            done.write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('Texture native',method,page,rec['mark_pixels'],round(rec['seconds'],1),flush=True)

if __name__=='__main__':main()
