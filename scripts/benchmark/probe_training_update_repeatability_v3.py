"""Repeat one real training update from a fixed state; no model selection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,hashlib,copy
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import run_paper_location_control as tr
from matched_generator_support import hooked,objective,logit_of
from ragan_pilot_support import predict,auxiliaries
from audit_ragan_inference_context import digest_state
from deterministic_reflection_support import install,checks
from compare_v5_aligned_fade_control import dump,sha,rgb
OUT=tr.ROOT/'tmp_work/training_update_repeatability_v3_20261023'
def gradient_sha(model):
    h=hashlib.sha256()
    for n,p in sorted(model.named_parameters()):
        if p.grad is not None:h.update(n.encode());h.update(p.grad.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def main():
    assert not OUT.exists();OUT.mkdir();tr.check();torch.set_num_threads(2);torch.cuda.set_device(0);device=torch.device('cuda:0');seed=1
    plan=json.loads((tr.base.OUT/'plans/main_seed1.json').read_text());data=tr.base.old.Data(tr.base.old.full.read_rows('train'));samples=[data.sample(e) for e in plan[:4]];x,t,v,pos,neg=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)];del data
    def inp(raw):return torch.from_numpy(auxiliaries(raw).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    bplan=json.loads((tr.base.OUT/'plans/background_seed1.json').read_text());bgs=[r for r in json.loads((tr.base.OUT/'background_candidates.json').read_text())['candidates'] if r['arm']=='multibook'];bg=inp(rgb(tr.ROOT/bgs[bplan[0]]['crop']))
    mplan=json.loads((tr.OLD/'plans/mixed_seed1.json').read_text());cs=json.loads((tr.OLD/'selected_sources.json').read_text())['cases'];c=cs[mplan[0]];mx=inp(rgb(tr.ROOT/c['crop']));lab=np.asarray(Image.open(tr.OLD/'labels'/f"{c['id']}.png"));ip=torch.from_numpy(lab==1).to(device);pp=torch.from_numpy(lab==2).to(device);npaper=torch.from_numpy(np.asarray(Image.open(tr.OUT/'paper_masks/near'/f"{c['id']}.png"))>0).to(device)
    defaults=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32);rows=[];errors=[]
    for kind in ['unet','ragan']:
      payload=torch.load(tr.base.old.OUT/f'{kind}_seed1/step_03024.pt',map_location='cpu',weights_only=False)
      for mode in ['strict_deterministic_adapter']:
        strict=True;torch.use_deterministic_algorithms(strict);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=strict;torch.backends.cuda.matmul.allow_tf32=False if strict else defaults['matmul_tf32'];torch.backends.cudnn.allow_tf32=False if strict else defaults['cudnn_tf32']
        for trial in [0,1]:
          try:
            torch.manual_seed(20261022);model,state=hooked(kind);reflection_names=install(model);model=model.to(device);model.load_state_dict(payload['model']);model.train();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5);opt.load_state_dict(copy.deepcopy(payload['optimizer']));before=digest_state(model)
            opt.zero_grad(set_to_none=True);prob=predict(model,kind,x);main,_=objective(prob,t,v,logit_of(kind,state['raw']),pos,neg);main.backward();predict(model,kind,bg);z=logit_of(kind,state['raw']);bg_loss=F.softplus(-z[:,:,64:192,64:192]).mean();bg_loss.backward()
            predict(model,kind,mx);z=logit_of(kind,state['raw'])[0,0];paper=F.softplus(-z[pp]).mean();ink=F.softplus(z[ip]).mean();extra=F.softplus(-z[npaper]).mean();(paper+ink+1*extra).backward();grad=gradient_sha(model);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            assert {int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}=={3024}
            rows.append(dict(kind=kind,mode=mode,trial=trial,reflection_modules=reflection_names,before_state=before,after_state=digest_state(model),gradient_sha256=grad,gradient_norm=float(norm),main=float(main.detach()),background=float(bg_loss.detach()),mixed_paper=float(paper.detach()),mixed_ink=float(ink.detach()),extra=float(extra.detach())))
            print(kind,mode,trial,'norm',float(norm),'state',rows[-1]['after_state'][:14],flush=True)
            del model,opt,state;gc.collect();torch.cuda.empty_cache()
          except RuntimeError as e:
            errors.append(dict(kind=kind,mode=mode,trial=trial,error=str(e)));print('Unsupported',kind,mode,str(e)[:250],flush=True);break
      del payload
    summaries=[]
    for kind in ['unet','ragan']:
      for mode in ['strict_deterministic_adapter']:
        rr=[r for r in rows if r['kind']==kind and r['mode']==mode];summaries.append(dict(kind=kind,mode=mode,successful_trials=len(rr),identical_gradient=len(rr)==2 and rr[0]['gradient_sha256']==rr[1]['gradient_sha256'],identical_update=len(rr)==2 and rr[0]['after_state']==rr[1]['after_state']))
    dump(OUT/'results.json',dict(reflection_checks=checks(),reflection_support_sha256=sha(tr.ROOT/'scripts/benchmark/deterministic_reflection_support.py'),rows=rows,summary=summaries,errors=errors,seed=1,arm='near',steps=1,no_new_training_checkpoint_saved=True,workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],default_tf32=defaults,source_protocol_sha256=sha(tr.OUT/'protocol.json'),limits='One fixed real minibatch/extra-context update, same process/GPU; does not prove full-run or cross-device reproducibility. Default-algorithms mode also has CUBLAS workspace config set. Strict mode changes math flags, not directly comparable performance.',probe_script_sha256=sha(__file__)))
    print(summaries,flush=True)
if __name__=='__main__':main()
