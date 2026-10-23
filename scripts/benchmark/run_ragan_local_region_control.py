"""Stable BCE + fixed20% inputs: normalized core BCE, with/without local paper."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as base
from run_ragan_stable_bce_control import stable_loss,product_logit
from run_ragan_stable_exposure_control import OUT as PARENT
from ragan_local_region_support import region_masks,local_terms,PADDING
from ragan_pilot_support import make_model
from compare_v5_aligned_fade_control import sha,dump
OUT=ROOT/'tmp_work/ragan_local_region_control_20261022'
CONDITIONS={f'{kind}_order{n}':dict(gan=False,batch_order_seed=20261021+n,order=n,kind=kind) for n in (0,1) for kind in ['inkmean','inkpaper']}
OriginalCrops=base.Crops

def checks():
 z=torch.tensor([[[[1.,-2.],[3.,-4.]]]],requires_grad=True)
 pos=torch.tensor([[[[True,False],[False,False]]]]);neg=torch.tensor([[[[False,True],[False,False]]]])
 a,b=local_terms(z,pos,neg);torch.testing.assert_close(a,torch.nn.functional.softplus(z[0,0,0,0]));torch.testing.assert_close(b,torch.nn.functional.softplus(-z[0,0,0,1]))
 g=torch.autograd.grad(a+b,z)[0];assert g[0,0,0,0]>0 and g[0,0,0,1]<0 and (g[0,0,1]==0).all()
 empty=torch.zeros_like(pos);c,d=local_terms(z,empty,empty);assert c==0 and d==0
 return dict(ink_gradient_reduces_paper_logit=True,paper_gradient_increases_paper_logit=True,excluded_positions_zero_gradient=True,no_region_returns_zero=True)

def freeze():
 assert not (OUT/'protocol.json').exists();OUT.mkdir(exist_ok=True)
 records=base.read_rows('train');regions=region_masks(records);maskfolder=OUT/'region_masks';maskfolder.mkdir();maskaudit=[]
 from PIL import Image
 for patch,m in regions.items():
  for k in ['core','paper','known_paper','roi']:Image.fromarray(m[k].astype(np.uint8)*255).save(maskfolder/(patch+'_'+k+'.png'))
  maskaudit.append(dict(patch=patch,core_n=int(m['core'].sum()),paper_proxy_n=int(m['paper'].sum()),box=m['box'],
   mask_hashes={k:sha(maskfolder/(patch+'_'+k+'.png')) for k in ['core','paper','known_paper','roi']}))
 dump(OUT/'region_mask_audit.json',maskaudit);dump(OUT/'loss_checks.json',checks())
 schedules_audit=[]
 for c,conf in CONDITIONS.items():
  source=PARENT/'batch_schedules'/f"stable_exposure_order{conf['order']}.json";dest=OUT/'batch_schedules'/(c+'.json');dest.parent.mkdir(exist_ok=True);shutil.copy2(source,dest)
  active=0;core_count=0;paper_count=0;minimum_paper=10**9;occurrences=[]
  for step,batch in enumerate(json.loads(source.read_text())):
   for idx in batch['indices']:
    row=records[idx];patch=row['patch_id']
    if patch not in regions or row['damage_family']!='targeted_fade' or float(row['damage_strength'])!=.8:continue
    x,y=base.coordinates(idx,batch['epoch'],row,base.metadata());m=regions[patch];pos=m['core'][y:y+256,x:x+256];neg=m['paper'][y:y+256,x:x+256]
    assert int(pos.sum())==int(m['core'].sum()) and int(neg.sum())>=16
    active+=1;core_count+=int(pos.sum());paper_count+=int(neg.sum());minimum_paper=min(minimum_paper,int(neg.sum()));occurrences.append(dict(step=step,patch=patch,core_n=int(pos.sum()),paper_n=int(neg.sum())))
  assert active==173*4
  schedules_audit.append(dict(condition=c,active_examples=active,core_point_exposures=core_count,paper_proxy_exposures=paper_count,minimum_paper_n=minimum_paper))
 dump(OUT/'region_exposure_audit.json',schedules_audit)
 for name in ['source_hashes.json','crop_schedule.json','fine_exposure_audit.json','evaluation_protocol.json']:shutil.copy2(PARENT/name,OUT/name)
 p=json.loads((PARENT/'protocol.json').read_text());p.update(conditions=CONDITIONS,parent_local_baseline_protocol_sha256=sha(PARENT/'protocol.json'),
  changed_factor='extra region-normalized positive BCE; second arm adds equally weighted local-paper BCE',
  baseline='stable BCE + identical20% schedule',extra_ink_coefficient=1.,extra_paper_coefficient={'inkmean':0.,'inkpaper':1.},
  extra_scope='only existing7 reviewed targeted_fade.80 examples; each active example normalized independently',
  extra_ink_mask='existing source-reviewed valid-positive core',extra_paper_mask='unchanged valid paper intersect source remote-paper proxy intersect core bounding box+16px',
  source_proxy_is_not_GT=True,ROI_padding=PADDING,no_label_changes=True,no_RGB_changes=True,no_threshold_changes=True,
  auxiliary_objective_composite_not_novelty_claim=True,exploratory_after_residual44_audit=True)
 dump(OUT/'protocol.json',p);snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
 for name in [Path(__file__).name,'ragan_local_region_support.py','run_ragan_full_continuation.py','run_ragan_stable_bce_control.py','ragan_pilot_support.py','evaluate_ragan_fine_retention_control.py','evaluate_ragan_joint_continuation.py']:
  shutil.copy2(ROOT/'scripts/benchmark'/name,snapshot/name)
 print('Frozen normalized positive BCE vs positive+paper BCE; unchanged stable20% baseline,2 orders',flush=True)

def train(condition,device):
 conf=CONDITIONS[condition];state={'raw':None,'masks':[],'losses':[]}
 class LocalCrops(OriginalCrops):
  def __init__(self,records):super().__init__(records);self.regions=region_masks(records)
  def get(self,idx,epoch):
   sample=super().get(idx,epoch);row=self.rows[idx];patch=row['patch_id'];active=patch in self.regions and row['damage_family']=='targeted_fade' and float(row['damage_strength'])==.8
   if active:
    x,y=base.coordinates(idx,epoch,row,self.meta);m=self.regions[patch];positive=m['core'][y:y+256,x:x+256];negative=m['paper'][y:y+256,x:x+256]
   else:positive=negative=np.zeros((256,256),bool)
   state['masks'].append((torch.from_numpy(positive.copy()).unsqueeze(0),torch.from_numpy(negative.copy()).unsqueeze(0)))
   return sample
 def hooked_model(kind):
  model=make_model(kind);model.corase_c6s.gate01.gate01.register_forward_hook(lambda m,i,o:state.update(raw=o));return model
 def objective(p,t,v):
  assert len(state['masks'])==4
  positive=torch.stack([m[0] for m in state['masks']]).to(p.device);negative=torch.stack([m[1] for m in state['masks']]).to(p.device);state['masks'].clear()
  assert not (positive&((v==0)|(t>.5))).any() and not (negative&((v==0)|(t<.5))).any()
  raw=state['raw'];original=stable_loss(p,t,v,raw);ink,paper=local_terms(product_logit(raw),positive,negative)
  total=original+ink+(paper if conf['kind']=='inkpaper' else 0)
  state['losses'].append(dict(global_loss=float(original.detach()),normalized_ink_BCE=float(ink.detach()),normalized_paper_BCE=float(paper.detach()),
   active_regions=int(positive.flatten(1).any(1).sum()),positive_pixels=int(positive.sum()),paper_pixels=int(negative.sum()),total=float(total.detach())))
  return total
 base.OUT=OUT;base.CONDITIONS=CONDITIONS;base.Crops=LocalCrops;base.make_model=hooked_model;base.supervised_loss=objective
 base.train(condition,device)
 dump(OUT/condition/'loss_components.json',state['losses'])
 assert len(state['losses'])==864 and not state['masks']

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train']);p.add_argument('--condition',choices=list(CONDITIONS));p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.action=='freeze':freeze()
 else:train(a.condition,a.device)
