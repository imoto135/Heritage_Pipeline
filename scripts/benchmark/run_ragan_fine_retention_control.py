"""No-GAN continuation: isolate reviewed-core weight and strong-fade frequency.

Reuse the frozen full-data baseline; preserve inputs, teachers, G and Adam.
"""
from __future__ import annotations
import argparse,json,shutil,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_v5_exposure_loss_control import core_masks
from ragan_pilot_support import make_model,predict,supervised_loss
from run_ragan_joint_continuation import evaluate_native
from compare_v5_aligned_fade_control import sha,dump
import diagnose_ragan_discriminator as diag
OUT=ROOT/'tmp_work/ragan_fine_retention_control_20261022'
CONDITIONS={f'{factor}_order{n}':dict(factor=factor,order=n,core_weight=8 if factor=='core8' else 1)
 for n in (0,1) for factor in ('core8','exposure20')}
BASELINES={f'baseline_order{n}':full.OUT/f'no_gan_order{n}' for n in (0,1)}
STEPS=864

def weighted_loss(p,t,w):
 p=p.float().clamp(1e-6,1-1e-6);t=t.float();w=w.float();n=w.sum().clamp_min(1)
 l1=((p-t).abs()*w).sum()/n;bce=(F.binary_cross_entropy(p,t,reduction='none')*w).sum()/n
 ink=1-p;truth=1-t;intersection=(ink*truth*w).sum();ip=(ink*w).sum();it=(truth*w).sum()
 dice=1-(2*intersection+1e-6)/(ip+it+1e-6)
 jaccard=1-(intersection+1e-6)/(ip+it-intersection+1e-6)
 return 10*l1+bce+dice+jaccard

def loss_checks():
 torch.manual_seed(11);p=torch.rand(2,1,8,8).clamp(.01,.99).requires_grad_(True);t=torch.randint(0,2,p.shape).float();v=(torch.rand_like(p)>.25).float()
 torch.testing.assert_close(weighted_loss(p,t,v),supervised_loss(p,t,v),rtol=1e-6,atol=1e-6)
 w=v*(1+7*(t==0));alternate=torch.where(v>0,t,1-t)
 torch.testing.assert_close(weighted_loss(p,t,w),weighted_loss(p,alternate,w),rtol=0,atol=0)
 g=torch.autograd.grad(weighted_loss(p,t,w),p)[0];assert (g[v==0]==0).all()
 # All-ink, weight8 must normalize to the same loss, without squaring weights.
 t0=torch.zeros_like(t);torch.testing.assert_close(weighted_loss(p,t0,8*v),weighted_loss(p,t0,v),rtol=1e-6,atol=1e-6)
 return dict(binary_weights_match_baseline=True,unknown_target_invariant=True,unknown_gradient_zero=True,uniform_weight_rescaling_invariant=True)

def freeze():
 assert not OUT.exists();OUT.mkdir()
 rows=full.read_rows('train');masks=core_masks();cores={patch:full.gray(path)>0 for patch,path in masks.items()};assert len(cores)==7
 audited=[]
 for patch,core in cores.items():
  r=next(r for r in rows if r['patch_id']==patch);valid=full.gray(ROOT/r['labels']['valid'])>0;t=full.gray(ROOT/r['labels']['target'])<128
  assert np.array_equal(core,core&valid&t);audited.append(dict(patch=patch,n=int(core.sum()),sha256=sha(masks[patch]),path=str(masks[patch].relative_to(ROOT))))
 assert sum(r['n'] for r in audited)==1558
 targets=[i for i,r in enumerate(rows) if r['patch_id'] in cores and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8]
 assert len(targets)==7
 schedule_audit=[]
 for name,conf in CONDITIONS.items():
  n=conf['order'];base=json.loads((full.OUT/'batch_schedules'/f'no_gan_order{n}.json').read_text());schedule=[]
  for epoch in full.EPOCHS:
   batches=[b for b in base if b['epoch']==epoch];indices=np.array([i for b in batches for i in b['indices']]);assert len(indices)==864
   before=indices.copy();replaced=[]
   if conf['factor']=='exposure20':
    rng=np.random.default_rng(full.SEED+n+epoch*100000+709)
    eligible=np.flatnonzero(~np.isin(indices,targets));replaced=rng.choice(eligible,166,replace=False)
    additions=np.concatenate([rng.permutation(targets) for _ in range(24)])[:166]
    indices[replaced]=additions
   assert int(np.isin(indices,targets).sum())==(173 if conf['factor']=='exposure20' else 7)
   schedule.extend(dict(epoch=epoch,indices=indices[k:k+4].tolist()) for k in range(0,864,4))
   target_draws=[int(i) for i in indices if i in targets]
   schedule_audit.append(dict(condition=name,epoch=epoch,target_draws=len(target_draws),changed_positions=int((indices!=before).sum()),
    per_target={rows[i]['patch_id']:int((indices==i).sum()) for i in targets},
    family_counts=dict(Counter(rows[int(i)]['damage_family'] for i in indices)),unique_input_records=len(set(indices.tolist()))))
  dump(OUT/'batch_schedules'/(name+'.json'),schedule)
  if conf['factor']=='core8':assert sha(OUT/'batch_schedules'/(name+'.json'))==sha(full.OUT/'batch_schedules'/f'no_gan_order{n}.json')
 checks=loss_checks();dump(OUT/'weighted_loss_checks.json',checks);dump(OUT/'core_mask_audit.json',audited);dump(OUT/'sampling_audit.json',schedule_audit)
 for filename in ('source_hashes.json','crop_schedule.json','fine_exposure_audit.json'):
  shutil.copy2(full.OUT/filename,OUT/filename)
 conf_all={**{k:dict(factor='baseline',order=n) for n,k in enumerate(BASELINES)},**CONDITIONS}
 dump(OUT/'protocol.json',dict(conditions=conf_all,new_conditions=CONDITIONS,baselines={k:str(v.relative_to(ROOT)) for k,v in BASELINES.items()},
  checkpoint=str(diag.CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=sha(diag.CHECKPOINT),
  parent_protocol_sha256=sha(full.OUT/'protocol.json'),train_manifest_sha256=sha(full.PARENT/'train_manifest.jsonl'),
  train_records=864,train_patches=96,train_page_groups=24,RGB_and_labels_unchanged=True,
  G_Adam_restored_step=2160,G_lr=3e-5,gradient_clip=1.,GAN=False,FP32=True,steps=STEPS,primary_step=864,secondary_step=432,
  saved_steps=[216,432,648,864],eval_steps=[0,216,432,648,864],threshold=.5,
  core_weight=8,core_weight_scope='all damage families, existing7 source-reviewed positive core masks only,1558 valid ink pixels',
  weighted_loss='same masked L1/BCE/inkDice/inkJaccard with one weight factor per pixel in each term; normalize by weighted sums',
  exposure='replace166 random non-target positions per epoch, keep original7 target draws;173/864 strong-fade target draws, roughly20%; remaining positions and crops unchanged',
  frequency_changes_family_mixture=True,exposure_repeated_crops=True,core_loss_batch_order_identical_to_baseline=True,
  two_batch_orders_not_independent_model_seeds=True,known_development=True,labels_provisional=True,formal_accuracy=False,
  official_reproduction=False,not_a_hyperparameter_search=True,knobs_chosen='8 and20% reused from prior short pilot; no tuning after this readout'))
 prior=full.OUT/'evaluation_protocol.json';evalprotocol=json.loads(prior.read_text());evalprotocol['parent_evaluation_protocol_sha256']=sha(prior)
 evalprotocol['comparison']='initial10epochs + existing noGAN baseline2 + core8/frequency20 two orders; fixedfinal14epochs primary,12epochs secondary'
 dump(OUT/'evaluation_protocol.json',evalprotocol)
 snapshot=OUT/'source_snapshot';snapshot.mkdir()
 for filename in (Path(__file__).name,'run_ragan_full_continuation.py','ragan_pilot_support.py','run_v5_exposure_loss_control.py','evaluate_ragan_joint_continuation.py'):
  shutil.copy2(ROOT/'scripts/benchmark'/filename,snapshot/filename)
 print('Frozen4 new controls; core1558 pixels; exposure173/864 target draws per epoch; same10epoch G/Adam',flush=True)

class WeightedCrops(full.Crops):
 def __init__(self,rows):
  super().__init__(rows);self.cores={patch:full.gray(path)>0 for patch,path in core_masks().items()}
 def core(self,index,epoch):
  row=self.rows[index];patch=row['patch_id'];x,y=full.coordinates(index,epoch,row,self.meta)
  a=self.cores[patch][y:y+256,x:x+256] if patch in self.cores else np.zeros((256,256),bool)
  return torch.from_numpy(a.copy()).unsqueeze(0)

def train(condition,device_name):
 p=json.loads((OUT/'protocol.json').read_text());conf=CONDITIONS[condition];dest=OUT/condition;assert not dest.exists();dest.mkdir()
 assert sha(diag.CHECKPOINT)==p['checkpoint_sha256'] and sha(full.PARENT/'train_manifest.jsonl')==p['train_manifest_sha256']
 for path,digest in json.loads((OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/path)==digest
 for r in json.loads((OUT/'core_mask_audit.json').read_text()):assert sha(ROOT/r['path'])==r['sha256']
 torch.set_num_threads(2);torch.manual_seed(full.SEED+conf['order']);device=torch.device(device_name);torch.cuda.set_device(device)
 payload=torch.load(diag.CHECKPOINT,map_location='cpu',weights_only=False);G=make_model('ragan').to(device);G.load_state_dict(payload['model'])
 opt=torch.optim.Adam([q for q in G.parameters() if q.requires_grad],lr=3e-5);opt.load_state_dict(payload['optimizer'])
 assert {int(s['step']) for s in opt.state.values() if 'step' in s}=={2160}
 data=WeightedCrops(full.read_rows('train'));schedule=json.loads((OUT/'batch_schedules'/(condition+'.json')).read_text());history=[];running=[];start=time.monotonic()
 for step in range(STEPS+1):
  if step in p['eval_steps']:
   entry=dict(step=step,validation=evaluate_native(G,device),seconds=time.monotonic()-start)
   if running:entry['training_mean']={k:float(np.mean([r[k] for r in running])) for k in running[0]};running=[]
   history.append(entry);dump(dest/'history.json',history);print(condition,step,entry['validation']['total'],'seconds',round(entry['seconds'],1),flush=True)
   if step in p['saved_steps']:
    torch.save(dict(model=G.state_dict(),optimizer=opt.state_dict(),condition=condition,step=2160+step,additional_steps=step,
     discriminator=None,d_optimizer=None,protocol_sha256=sha(OUT/'protocol.json')),dest/f'step_{step:05d}.pt')
   if step==STEPS:break
  batch=schedule[step];samples=[data.get(i,batch['epoch']) for i in batch['indices']];x,t,v=[torch.stack([a[j] for a in samples]).to(device) for j in range(3)]
  G.train();opt.zero_grad(set_to_none=True);pred=predict(G,'ragan',x)
  if conf['factor']=='core8':
   core=torch.stack([data.core(i,batch['epoch']) for i in batch['indices']]).to(device);assert not (core&((v==0)|(t>.5))).any()
   w=v*(1+7*core.float());loss=weighted_loss(pred,t,w)
  else:loss=supervised_loss(pred,t,v)
  assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(G.parameters(),1.);assert torch.isfinite(norm);opt.step()
  running.append(dict(objective=float(loss.detach())))
 dump(dest/'completion.json',dict(condition=condition,additional_steps=STEPS,initial_optimizer_steps=[2160],
  final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
  schedule_sha256=sha(OUT/'batch_schedules'/(condition+'.json')),checkpoint_sha256=sha(diag.CHECKPOINT),
  no_discriminator=True,seconds=time.monotonic()-start,formal_accuracy=False))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','train']);a.add_argument('--condition',choices=list(CONDITIONS));a.add_argument('--device',default='cuda:0');args=a.parse_args()
 if args.action=='freeze':freeze()
 else:train(args.condition,args.device)
