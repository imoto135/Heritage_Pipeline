"""Native fixed-final partial-supervision predictions; book exclusions enforced."""
import argparse,json,time
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,ROOT,file_hash
from evaluate_synthetic_selectors import load_selector
from predict_native_selectors import predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--roles',nargs='+',default=['development']);p.add_argument('--exclude-dev-book');a=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4)
    pages={r['page']:r for r in json.loads((OUT/'cases_frozen.json').read_text())['cases'] if r['role'] in a.roles}
    if a.exclude_dev_book:pages={p:r for p,r in pages.items() if r['book']==a.exclude_dev_book}
    for name in a.names:
        suffix='_exclude'+a.exclude_dev_book if a.exclude_dev_book else '';trial=OUT/'native_partial_trials'/f'{name}_seed715{suffix}'
        assert (trial/'complete.json').exists();path=trial/'last.pth';model,cfg,ck=load_selector(path,a.device);assert ck['step']==8000
        protocol=json.loads((trial/'protocol.json').read_text());method='partial_'+name+suffix
        for page,r in pages.items():
            if a.exclude_dev_book:assert r['book'] not in protocol['actual_training_books']
            dest=OUT/'native_predictions'/method/page;dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json'
            if done.exists():
                rec=json.loads(done.read_text());assert rec['checkpoint_sha256']==file_hash(path)
                for f,sha in rec['outputs'].items():assert file_hash(dest/f)==sha
                continue
            start=time.monotonic();assert file_hash(ROOT/r['original'])==r['source_sha256'];rgb=np.asarray(Image.open(ROOT/r['original']).convert('RGB'))
            prob=predict(model,rgb,'relative',a.device);assert np.isfinite(prob).all()
            for i,layer in enumerate(['mark','stain']):
                Image.fromarray((prob[i]>=.5).astype(np.uint8)*255).save(dest/(layer+'_mask.png'))
                Image.fromarray(np.rint(prob[i]*255).astype(np.uint8)).save(dest/(layer+'_probability_preview.png'))
            done.write_text(json.dumps(dict(name=method,page=page,book=r['book'],role=r['role'],original=r['original'],source_sha256=r['source_sha256'],
                checkpoint_sha256=file_hash(path),training_protocol_sha256=file_hash(trial/'protocol.json'),script_sha256=file_hash(__file__),
                native_inference_source_sha256=file_hash(OUT/'predict_native_selectors.py'),threshold=.5,final_steps=8000,
                mark_pixels=int((prob[0]>=.5).sum()),seconds=time.monotonic()-start,outputs={f.name:file_hash(f) for f in dest.glob('*.png')},
                no_real_pixel_or_reading_GT=True,excluded_development_books=protocol['excluded_development_books']),ensure_ascii=False,indent=2))
            print('Partial native',method,page,round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
