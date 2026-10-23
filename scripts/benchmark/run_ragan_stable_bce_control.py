"""Numerical BCE fix only: exact logit of final sigmoid(a)*sigmoid(b).

Keep the public forward and non-BCE supervised terms unchanged.
"""
import argparse,json,shutil,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as base
from ragan_pilot_support import make_model,supervised_loss
from compare_v5_aligned_fade_control import sha,dump
OUT=ROOT/'tmp_work/ragan_stable_bce_control_20261022'
PARENT=base.OUT
CONDITIONS={f'stable_bce_order{n}':dict(gan=False,batch_order_seed=base.SEED+n,order=n) for n in (0,1)}

def product_logit(raw):
 a,b=raw.float().chunk(2,dim=1)
 return a+b-torch.logsumexp(torch.stack([torch.zeros_like(a),a,b]),dim=0)

def stable_loss(paper_probability,target,valid,raw):
 # Preserve exactly the old L1/Dice/Jaccard; replace the BCE path alone.
 p=paper_probability.float().clamp(1e-6,1-1e-6);t=target.float();v=valid.float();n=v.sum().clamp_min(1)
 l1=((p-t).abs()*v).sum()/n
 bce=(F.binary_cross_entropy_with_logits(product_logit(raw),t,reduction='none')*v).sum()/n
 ink=(1-p)*v;truth=(1-t)*v;intersection=(ink*truth).sum()
 dice=1-(2*intersection+1e-6)/(ink.sum()+truth.sum()+1e-6)
 jaccard=1-(intersection+1e-6)/(ink.sum()+truth.sum()-intersection+1e-6)
 return 10*l1+bce+dice+jaccard

def checks():
 torch.manual_seed(17);raw=torch.randn(2,2,8,8,requires_grad=True);a,b=raw.chunk(2,dim=1);p=torch.sigmoid(a)*torch.sigmoid(b)
 t=torch.randint(0,2,p.shape).float();v=(torch.rand_like(p)>.25).float()
 torch.testing.assert_close(torch.sigmoid(product_logit(raw)),p,rtol=2e-6,atol=1e-7)
 torch.testing.assert_close(stable_loss(p,t,v,raw),supervised_loss(p,t,v),rtol=2e-6,atol=1e-6)
 ga=torch.autograd.grad(stable_loss(p,t,v,raw),raw,retain_graph=True)[0];gb=torch.autograd.grad(supervised_loss(p,t,v),raw)[0]
 torch.testing.assert_close(ga,gb,rtol=2e-5,atol=1e-7)
 z=torch.full((1,2,2,2),30.,requires_grad=True);a,b=z.chunk(2,1);p=torch.sigmoid(a)*torch.sigmoid(b);t=torch.zeros_like(p);v=torch.ones_like(p)
 old=supervised_loss(p,t,v);oldg=torch.autograd.grad(old,z,retain_graph=True)[0];new=stable_loss(p,t,v,z);newg=torch.autograd.grad(new,z)[0]
 assert (oldg==0).all() and torch.isfinite(new) and torch.isfinite(newg).all() and (newg>0).all()
 z2=torch.randn(1,2,4,4,requires_grad=True);a,b=z2.chunk(2,1);p2=torch.sigmoid(a)*torch.sigmoid(b);t2=torch.zeros_like(p2);v2=torch.ones_like(p2);v2[:,:,:2]=0
 unknown_target=torch.where(v2>0,t2,1-t2)
 torch.testing.assert_close(stable_loss(p2,t2,v2,z2),stable_loss(p2,unknown_target,v2,z2),rtol=0,atol=0)
 g2=torch.autograd.grad(stable_loss(p2,t2,v2,z2),z2)[0];assert (g2[:,:,:2]==0).all()
 return dict(exact_probability_equivalence=True,unsaturated_loss_equivalence=True,unsaturated_gradient_equivalence=True,
  saturated_wrong_ink_old_gradient_zero=True,saturated_wrong_ink_new_gradient_nonzero=True,unknown_target_invariant=True,unknown_gradient_zero=True,
  formula='logit(sigmoid(a)*sigmoid(b))=a+b-logsumexp(0,a,b)',saturated_example_old_loss=float(old.detach()),saturated_example_new_loss=float(new.detach()))

def freeze():
 assert not OUT.exists();OUT.mkdir()
 for name in ['source_hashes.json','crop_schedule.json','fine_exposure_audit.json']:
  shutil.copy2(PARENT/name,OUT/name)
 p=json.loads((PARENT/'protocol.json').read_text());p.update(conditions=CONDITIONS,parent_protocol_sha256=sha(PARENT/'protocol.json'),
  changed_factor='BCE numerical formulation only; non-BCE terms and public forward unchanged',
  GAN=False,reference_D_weights_and_Adam_restored=False,paired_with_existing_noGAN=True,exploratory_after_fine_retention_readout=True)
 dump(OUT/'protocol.json',p);dump(OUT/'evaluation_protocol.json',json.loads((PARENT/'evaluation_protocol.json').read_text()));dump(OUT/'loss_checks.json',checks())
 for n in (0,1):
  shutil.copy2(PARENT/'batch_schedules'/f'no_gan_order{n}.json',OUT/'batch_schedules'/f'stable_bce_order{n}.json') if (OUT/'batch_schedules').exists() else None
 (OUT/'batch_schedules').mkdir(exist_ok=True)
 for n in (0,1):shutil.copy2(PARENT/'batch_schedules'/f'no_gan_order{n}.json',OUT/'batch_schedules'/f'stable_bce_order{n}.json')
 snapshot=OUT/'source_snapshot';snapshot.mkdir()
 for name in [Path(__file__).name,'run_ragan_full_continuation.py','ragan_pilot_support.py','evaluate_ragan_fine_retention_control.py','evaluate_ragan_joint_continuation.py']:
  shutil.copy2(ROOT/'scripts/benchmark'/name,snapshot/name)
 print('Frozen exact-logit BCE only control, two unchanged schedules',flush=True)

def train(condition,device):
 captured={}
 def hooked_model(kind):
  model=make_model(kind)
  model.corase_c6s.gate01.gate01.register_forward_hook(lambda m,i,o:captured.update(raw=o))
  return model
 def loss(p,t,v):return stable_loss(p,t,v,captured['raw'])
 base.OUT=OUT;base.CONDITIONS=CONDITIONS;base.make_model=hooked_model;base.supervised_loss=loss
 base.train(condition,device)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--condition',choices=list(CONDITIONS));a.add_argument('--device',default='cuda:0');args=a.parse_args()
 if args.action=='freeze':freeze()
 else:train(args.condition,args.device)
