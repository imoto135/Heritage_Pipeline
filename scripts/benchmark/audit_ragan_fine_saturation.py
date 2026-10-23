"""Read-only direct supervision gradients on the two stubborn training traces."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_fine_retention_control import OUT
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss
from compare_v5_aligned_fade_control import sha,dump,rgb
import diagnose_ragan_discriminator as diag

def main():
 device=torch.device('cuda:1');torch.cuda.set_device(device);torch.set_num_threads(2)
 protocol=json.loads((OUT/'hard_fine_fit_probe/protocol.json').read_text());rows=full.read_rows('train');reviews=full.reviewed();samples=[];traces=[]
 for case in protocol['cases']:
  row=next(r for r in rows if r['patch_id']==case['patch'] and r['damage_family']==case['family'] and (case['family']=='identity' or float(r['damage_strength'])==.8));x,y=case['xy'];image=rgb(ROOT/row['input'])[y:y+256,x:x+256]
  t=full.gray(ROOT/row['labels']['target'])[y:y+256,x:x+256];v=full.gray(ROOT/row['labels']['valid'])[y:y+256,x:x+256]
  samples.append((torch.from_numpy(auxiliaries(image).copy()).permute(2,0,1).float()/255,torch.from_numpy(t.copy()).unsqueeze(0).float()/255,torch.from_numpy(v.copy()).unsqueeze(0).float()/255));traces.append(reviews[case['patch']][1][y:y+256,x:x+256])
 x,t,v=[torch.stack([s[i] for s in samples]).to(device) for i in range(3)];result=[]
 checkpoints={'initial':diag.CHECKPOINT,'exposure20_order0':OUT/'exposure20_order0/step_00864.pt','exposure20_order1':OUT/'exposure20_order1/step_00864.pt'}
 for name,path in checkpoints.items():
  G=make_model('ragan').to(device);G.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);G.eval();captured={}
  handle=G.corase_c6s.gate01.gate01.register_forward_hook(lambda m,i,o:captured.update(raw=o))
  p=predict(G,'ragan',x);raw=captured['raw'];a,b=raw.chunk(2,dim=1)
  logits=a+b-torch.logsumexp(torch.stack([torch.zeros_like(a),a,b]),dim=0)
  torch.testing.assert_close(torch.sigmoid(logits),p,rtol=2e-5,atol=2e-6)
  old=supervised_loss(p,t,v);gp,gr=torch.autograd.grad(old,[p,raw],retain_graph=True)
  stable=(F.binary_cross_entropy_with_logits(logits,t,reduction='none')*v).sum()/v.sum()
  stable_raw=torch.autograd.grad(stable,raw)[0]
  for i,case in enumerate(protocol['cases']):
   if case['family']!='targeted_fade':continue
   tr=torch.from_numpy(traces[i]).to(device);values=p[i,0][tr];g=gp[i,0][tr];zgrad=gr[i,:,tr];sg=stable_raw[i,:,tr]
   result.append(dict(condition=name,patch=case['patch'],n=int(tr.sum()),paper_min=float(values.min().detach()),paper_median=float(values.median().detach()),paper_max=float(values.max().detach()),
    paper_at_or_above_clip=int((values>=1-1e-6).sum()),paper_exact1=int((values==1).sum()),zero_direct_p_gradient=int((g==0).sum()),
    zero_direct_raw_head_gradient=int((zgrad.abs().sum(0)==0).sum()),nonzero_stable_BCE_head_gradient=int((sg.abs().sum(0)>0).sum()),checkpoint_sha256=sha(path)))
  handle.remove();del G;torch.cuda.empty_cache()
 dump(OUT/'fine_saturation_audit.json',dict(rows=result,read_only=True,same_product_sigmoid_model=True,stable_logit='a+b-logsumexp(0,a,b)',not_a_training_result=True))
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
