"""Fixed added-field diagnostic for matched native-partial controls."""
import argparse,json,time
import cv2,numpy as np,torch
from PIL import Image
from synthetic_layers import OUT,file_hash
from evaluate_synthetic_selectors import load_selector
from predict_native_selectors import predict
from evaluate_counterfactual_challenge import diagnose
from render_selector_diagnostics import mask_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['sparse_control','dense_native','dense_native_brown']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4);root=OUT/'counterfactual_stain_challenge';rows=json.loads((root/'inputs_frozen.json').read_text())['records'];cache={}
    names=[]
    for variant in a.names:
        trial=OUT/'native_partial_trials'/f'{variant}_seed715';assert (trial/'complete.json').exists();checkpoint=trial/'last.pth';model,cfg,ck=load_selector(checkpoint,a.device);assert ck['step']==8000;name='partial_'+variant;names.append(name)
        for r in rows:
            dest=root/'predictions'/name/r['id'];dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json'
            if done.exists():
                old=json.loads(done.read_text());assert old['input_sha256']==r['input_sha256'];assert old['checkpoint_sha256']==file_hash(checkpoint);assert old['mask_sha256']==file_hash(dest/'mark_mask.png');continue
            start=time.monotonic();assert file_hash(OUT/r['input'])==r['input_sha256'];rgb=np.asarray(Image.open(OUT/r['input']).convert('RGB'));mask=predict(model,rgb,'relative',a.device)[0]>=.5
            key=(name,r['page'])
            if key not in cache:
                original=mask_path(*key);assert original.exists(),'Original prediction must be completed before comparison';cache[key]=np.asarray(Image.open(original).convert('L'))>127
            Image.fromarray(mask.astype(np.uint8)*255).save(dest/'mark_mask.png')
            done.write_text(json.dumps(dict(id=r['id'],case=r['case'],name=name,family='native_partial',input_sha256=r['input_sha256'],checkpoint_sha256=file_hash(checkpoint),
                training_protocol_sha256=file_hash(trial/'protocol.json'),script_sha256=file_hash(__file__),diagnostic_helper_sha256=file_hash(OUT/'evaluate_counterfactual_challenge.py'),
                mask_sha256=file_hash(dest/'mark_mask.png'),threshold=.5,seconds=time.monotonic()-start,diagnostic=diagnose(r,mask,cache[key]),
                original_training_points_may_have_been_seen=True,no_real_pixel_or_reading_GT=True,perturbation='Frozen development diagnosis, not final test'),ensure_ascii=False,indent=2))
            print('Partial counterfactual',name,r['id'],round(time.monotonic()-start,1),flush=True)
    (root/'native_partial_results.json').write_text(json.dumps([json.loads(f.read_text()) for name in names for f in (root/'predictions'/name).glob('*/complete.json')],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
