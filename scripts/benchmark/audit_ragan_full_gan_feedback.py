"""Read-only local gradient scale on preselected full-training batches."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_full_continuation import OUT,ROOT,read_rows,metadata,coordinates,reviewed,gray
from compare_v5_aligned_fade_control import dump,sha,rgb
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss,critic_input
import diagnose_ragan_discriminator as diag

def freeze():
 path=OUT/'gradient_protocol.json';assert not path.exists()
 records=read_rows('train');patches=sorted(reviewed())[:4];batches=[]
 for family in ('identity','targeted_fade','stain_overlap','mixed'):
  ids=[]
  for patch in patches:
   candidates=[i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']==family]
   ids.append(max(candidates,key=lambda i:float(records[i]['damage_strength'])))
  batches.append(dict(family=family,indices=ids,epoch=10))
 dump(path,dict(selection='lexicographically first4 of existing7 reviewed train patches; highest strength per family',
  batches=batches,conditions=['initial','reference_gan_order0','reference_gan_order1'],checkpoint_step=864,
  no_G_or_D_updates=True,losses='shared supervised loss versus .1 BCE(D(G),1)',
  interpretation='local parameter-gradient norm ratio only, not restoration accuracy or global bound',
  formal_accuracy=False))

def main(device_name):
 protocol=json.loads((OUT/'gradient_protocol.json').read_text());records=read_rows('train');meta=metadata();batches=[]
 for row in protocol['batches']:
  samples=[]
  for index in row['indices']:
   r=records[index];x,y=coordinates(index,row['epoch'],r,meta)
   a=rgb(ROOT/r['input'])[y:y+256,x:x+256];t=gray(ROOT/r['labels']['target'])[y:y+256,x:x+256];v=gray(ROOT/r['labels']['valid'])[y:y+256,x:x+256]
   samples.append((torch.from_numpy(auxiliaries(a).copy()).permute(2,0,1).float()/255,
    torch.from_numpy(t.copy()).unsqueeze(0).float()/255,torch.from_numpy(v.copy()).unsqueeze(0).float()/255))
  batches.append((row,[torch.stack([s[j] for s in samples]) for j in range(3)]))
 torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device);result=[]
 for condition in protocol['conditions']:
  ck=diag.CHECKPOINT if condition=='initial' else OUT/condition/'step_00864.pt'
  payload=torch.load(ck,map_location='cpu',weights_only=False)
  G=make_model('ragan').to(device);G.load_state_dict(payload['model']);G.eval()
  D=diag.get_discriminator(payload,'sigmoid_scalar_hinge',device);D.eval()
  for parameter in D.parameters():parameter.requires_grad_(False)
  parameters=[p for p in G.parameters() if p.requires_grad]
  for row,parts in batches:
   x,t,v=[p.to(device) for p in parts];p=predict(G,'ragan',x)
   sup=supervised_loss(p,t,v);score=D(critic_input(p,v),x[:,:3]).clamp(1e-6,1-1e-6)
   adv=.1*F.binary_cross_entropy(score,torch.ones_like(score))
   gs=torch.autograd.grad(sup,parameters,retain_graph=True,allow_unused=True)
   ga=torch.autograd.grad(adv,parameters,allow_unused=True)
   ns=float(torch.sqrt(sum(g.float().square().sum() for g in gs if g is not None)))
   na=float(torch.sqrt(sum(g.float().square().sum() for g in ga if g is not None)))
   result.append(dict(condition=condition,family=row['family'],supervised_loss=float(sup.detach()),weighted_adversarial_loss=float(adv.detach()),supervised_gradient_norm=ns,weighted_adversarial_gradient_norm=na,ratio=na/max(ns,1e-30),checkpoint_sha256=sha(ck)))
  del G,D,payload;torch.cuda.empty_cache()
 dump(OUT/'gradient_feedback.json',dict(rows=result,protocol_sha256=sha(OUT/'gradient_protocol.json'),local_audit_only=True,no_updates=True))
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.action=='freeze':freeze()
 else:main(a.device)
