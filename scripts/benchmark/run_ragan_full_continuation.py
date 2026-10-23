"""Resume matched G/Adam on all 96 patches, with audited fine-stroke exposure.

Two batch-order seeds are sensitivity checks, not independently trained models.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_v5_aligned_fade_control import SEED,OLD,CORES,trace_centers,crop_xy
from run_contact_stain_training_control import OUT as CONTACT,make_training_field
from compare_v5_aligned_fade_control import sha,dump,rgb
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss,critic_input
from run_ragan_joint_continuation import evaluate_native
from evaluate_ridge_gate_independent_val import proxies
import diagnose_ragan_discriminator as diag
OUT=ROOT/'tmp_work/ragan_full_continuation_20261022'
PARENT=ROOT/'tmp_work/ragan_convergence_audit_20261022/ragan_contact'
EPOCHS=list(range(10,14));STEPS=864
CONDITIONS={f'{kind}_order{n}':dict(gan=kind=='reference_gan',batch_order_seed=SEED+n) for n in (0,1) for kind in ('no_gan','reference_gan')}

def read_rows(split):return [json.loads(s) for s in (PARENT/(split+'_manifest.jsonl')).read_text().splitlines()]
def gray(path):return np.asarray(Image.open(path).convert('L'))
def metadata():
 centers=trace_centers()
 generic={r['patch']:r['selected_generic_xy'] for r in json.loads((OLD/'input_audit.json').read_text())['rows'] if r['strength']==.8}
 anchors={r['patch']:r['anchor_xy'] for r in json.loads((CONTACT/'input_audit.json').read_text())['rows']}
 return centers,generic,anchors

def coordinates(idx,epoch,row,meta):
 centers,generic,anchors=meta
 if row['damage_family']=='targeted_fade':return crop_xy(idx,epoch,row,centers,generic)
 rng=np.random.default_rng(SEED+epoch*100000+idx)
 if row['damage_family']=='stain_overlap':
  dx,dy=map(int,rng.integers(-16,17,size=2));cx,cy=anchors[row['patch_id']]
  return int(np.clip(cx-128+dx,0,256)),int(np.clip(cy-128+dy,0,256))
 return tuple(map(int,rng.integers(0,257,size=2)))

def reviewed():
 result={}
 for patch in trace_centers():
  path=CORES/'masks'/(patch+'_trace.png')
  if not path.exists():path=ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png')
  result[patch]=(path,gray(path)>0)
 return result

def freeze():
 assert not (OUT/'protocol.json').exists();OUT.mkdir(exist_ok=True)
 rows=read_rows('train');val=read_rows('val');meta=metadata();reviews=reviewed()
 assert len(rows)==864 and len({r['patch_id'] for r in rows})==96
 train_groups={r['group'] for r in rows};val_groups={r['group'] for r in val};assert train_groups.isdisjoint(val_groups)
 assert not any(r['patch_id']=='200017197_00172_0_p00' for r in rows)
 hashes={};labels={};sources={}
 for row in rows+val:
  pairs=[('input','input_sha256')]
  for pk,hk in pairs:
   path=ROOT/row[pk]
   if row[pk] not in hashes:hashes[row[pk]]=sha(path)
   assert hashes[row[pk]]==row[hk]
  for key in ('target','valid'):
   path=ROOT/row['labels'][key]
   if row['labels'][key] not in hashes:hashes[row['labels'][key]]=sha(path)
   assert hashes[row['labels'][key]]==row['labels'][key+'_sha256']
  if row['patch_id'] not in sources:
   path=ROOT/row['clean'];sources[row['patch_id']]=rgb(path);hashes[row['clean']]=sha(path)
  if row['patch_id'] not in labels:labels[row['patch_id']]=(gray(ROOT/row['labels']['target']),gray(ROOT/row['labels']['valid'])>0)
 trace_audit=[];crop_schedule=[];stain_audit=[]
 fields={}
 for patch,source in sources.items():
  if patch not in {r['patch_id'] for r in rows}:continue
  _,core,remote,_=proxies(source);field,_=make_training_field(core);fields[patch]=(core,remote,field>=.5)
 for epoch in EPOCHS:
  for idx,row in enumerate(rows):
   x,y=coordinates(idx,epoch,row,meta);patch=row['patch_id']
   crop_schedule.append(dict(index=idx,epoch=epoch,xy=[x,y],patch=patch,family=row['damage_family']))
   if patch in reviews and row['damage_family']=='targeted_fade':
    path,trace=reviews[patch];sub=trace[y:y+256,x:x+256];target,valid=labels[patch]
    n=int(trace.sum());inside=int(sub.sum());known=int((trace&valid&(target<128)).sum())
    damaged=rgb(ROOT/row['input']);delta=cv2.cvtColor(damaged,cv2.COLOR_RGB2GRAY).astype(np.int16)-cv2.cvtColor(sources[patch],cv2.COLOR_RGB2GRAY).astype(np.int16)
    changed=int((trace&(np.abs(delta)>=10)).sum())
    assert inside==n and known==n,(patch,epoch,inside,known,n)
    changed_any=int((trace&(np.abs(delta)>=1)).sum());assert changed_any==n
    if float(row['damage_strength'])==.8:assert changed==n,(patch,row['damage_strength'],changed,n)
    trace_audit.append(dict(patch=patch,index=idx,epoch=epoch,strength=row['damage_strength'],n=n,in_crop=inside,valid_ink=known,changed_ge1=changed_any,changed_ge10=changed,minimum_gray_change=int(np.abs(delta)[trace].min()),trace_sha256=sha(path)))
   if row['damage_family']=='stain_overlap':
    core,remote,field=fields[patch];affected=core&field;paper=remote&field
    ink_n=int(affected[y:y+256,x:x+256].sum());paper_n=int(paper[y:y+256,x:x+256].sum())
    assert ink_n>=20 and paper_n>=200
    stain_audit.append(dict(index=idx,epoch=epoch,patch=patch,core_in_stain_crop=ink_n,remote_paper_in_stain_crop=paper_n))
 assert len(trace_audit)==7*3*4 and sum(r['in_crop'] for r in trace_audit)==455*3*4
 dump(OUT/'source_hashes.json',hashes);dump(OUT/'crop_schedule.json',crop_schedule)
 dump(OUT/'fine_exposure_audit.json',dict(rows=trace_audit,unique_trace_points=455,unique_patches=7,all_reviewed_points_valid_ink=True,all_reviewed_points_inside_targeted_fade_crops=True,all_reviewed_points_changed_ge1=True,strong_fade_all_reviewed_points_changed_ge10=True,all_strengths_changed_ge10=False,exposures=len(trace_audit),repeated_point_exposures=sum(r['in_crop'] for r in trace_audit)))
 dump(OUT/'contact_crop_audit.json',dict(rows=stain_audit,source_proxy_not_GT=True))
 for name,conf in CONDITIONS.items():
  schedule=[]
  for epoch in EPOCHS:
   generator=torch.Generator().manual_seed(conf['batch_order_seed']+100000*epoch)
   indices=torch.randperm(len(rows),generator=generator).tolist()
   schedule.extend(dict(epoch=epoch,indices=indices[k:k+4]) for k in range(0,len(rows),4))
  assert len(schedule)==STEPS;dump(OUT/'batch_schedules'/(name+'.json'),schedule)
 for n in (0,1):assert sha(OUT/'batch_schedules'/f'no_gan_order{n}.json')==sha(OUT/'batch_schedules'/f'reference_gan_order{n}.json')
 dump(OUT/'protocol.json',dict(conditions=CONDITIONS,checkpoint=str(diag.CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=sha(diag.CHECKPOINT),
  train_manifest_sha256=sha(PARENT/'train_manifest.jsonl'),val_manifest_sha256=sha(PARENT/'val_manifest.jsonl'),
  train_patches=96,train_pages=24,train_groups=24,val_groups=8,train_records=864,
  families=dict(Counter(r['damage_family'] for r in rows)),input_and_labels_unchanged=True,
  crops='original epoch-index formula; continue crop epochs10..13; targeted fade centers7 / stain anchors96 / otherwise random',
  steps=STEPS,epochs=EPOCHS,batch_size=4,saved_steps=[216,432,648,864],primary_step=864,threshold=.5,
  common_G_weights=True,common_G_Adam_restored_step=2160,G_lr=3e-5,gradient_clip=1.,FP32=True,
  reference_D_weights_and_Adam_restored=True,D_lr=1e-6,D_loss='sigmoid scalar mean hinge',G_adv_weight=.1,
  shuffle='new frozen stateless per-epoch permutations; not exact persisted original loader state',
  two_batch_order_sensitivity_checks=True,independent_model_replicates=False,
  crop_formula_seed=SEED,eval_steps=[0,216,432,648,864],formal_accuracy=False,labels_provisional=True,
  known_development=True,no_val_trace_in_training=True,official_reproduction=False,
  purpose='full training composition control; GAN versus no-GAN from common trained generator; not a scratch model comparison'))
 prior=ROOT/'tmp_work/ragan_convergence_audit_20261022/evaluation_protocol.json'
 dump(OUT/'evaluation_protocol.json',dict(inherit_evaluation_from=str(prior.relative_to(ROOT)),inherit_sha256=sha(prior),threshold=.5,
  primary_step=864,secondary_step=432,all_conditions_reported=True,known_development=True,formal_accuracy=False,
  fixed_native_val8='same critic-diagnostic caches; supplementary only',full512_synthetic=True,
  real_cases=['D01','D03','D04','T07','C01'],thin_trace49_separate=True,
  training_reviewed_fine_traces='all7,identity and targeted fade.80,full512; training fit only',
  adoption='retain thin text and improve stain paper; no selection by provisional aggregate F1'))
 snapshot=OUT/'source_snapshot';snapshot.mkdir()
 for filename in (Path(__file__).name,'ragan_pilot_support.py','run_contact_stain_training_control.py','run_v5_aligned_fade_control.py'):
  shutil.copy2(ROOT/'scripts/benchmark'/filename,snapshot/filename)
 print('Frozen:',len(rows),'records,7 reviewed traces/455 points,84 fade crops,5460 supervised point exposures per run',flush=True)

class Crops:
 def __init__(self,rows):
  self.rows=rows;self.meta=metadata();self.labels={};self.inputs={}
  for row in rows:
   if row['patch_id'] not in self.labels:self.labels[row['patch_id']]=(gray(ROOT/row['labels']['target']),gray(ROOT/row['labels']['valid']))
   self.inputs[row['input']]=rgb(ROOT/row['input'])
 def get(self,index,epoch):
  row=self.rows[index];x,y=coordinates(index,epoch,row,self.meta)
  raw=self.inputs[row['input']][y:y+256,x:x+256];t,v=self.labels[row['patch_id']]
  return (torch.from_numpy(auxiliaries(raw).copy()).permute(2,0,1).float()/255,
   torch.from_numpy(t[y:y+256,x:x+256].copy()).unsqueeze(0).float()/255,
   torch.from_numpy(v[y:y+256,x:x+256].copy()).unsqueeze(0).float()/255)

def train(condition,device_name):
 conf=CONDITIONS[condition];protocol=json.loads((OUT/'protocol.json').read_text());dest=OUT/condition;assert not dest.exists();dest.mkdir()
 assert sha(diag.CHECKPOINT)==protocol['checkpoint_sha256'] and sha(PARENT/'train_manifest.jsonl')==protocol['train_manifest_sha256']
 for path,digest in json.loads((OUT/'source_hashes.json').read_text()).items():assert sha(ROOT/path)==digest
 torch.set_num_threads(2);torch.manual_seed(conf['batch_order_seed']);device=torch.device(device_name);torch.cuda.set_device(device)
 payload=torch.load(diag.CHECKPOINT,map_location='cpu',weights_only=False)
 model=make_model('ragan').to(device);model.load_state_dict(payload['model'])
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5);opt.load_state_dict(payload['optimizer'])
 assert {int(s['step']) for s in opt.state.values() if 'step' in s}=={2160}
 D=diag.get_discriminator(payload,'sigmoid_scalar_hinge',device) if conf['gan'] else None;dopt=None
 if D is not None:
  dopt=torch.optim.Adam(D.parameters(),lr=1e-6);dopt.load_state_dict(payload['d_optimizer'])
  assert {int(s['step']) for s in dopt.state.values() if 'step' in s}=={2160}
 trainset=Crops(read_rows('train'));schedule=json.loads((OUT/'batch_schedules'/(condition+'.json')).read_text())
 history=[];running=[];start=time.monotonic()
 for step in range(STEPS+1):
  if step in protocol['eval_steps']:
   entry=dict(step=step,validation=evaluate_native(model,device),seconds=time.monotonic()-start)
   if running:entry['training_mean']={k:float(np.mean([r[k] for r in running])) for k in running[0]};running=[]
   history.append(entry);dump(dest/'history.json',history)
   print(condition,step,entry['validation']['total'],'seconds',round(entry['seconds'],1),flush=True)
   if step in protocol['saved_steps']:
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=2160+step,additional_steps=step,
     discriminator=D.state_dict() if D is not None else None,d_optimizer=dopt.state_dict() if dopt else None,
     condition=condition,protocol_sha256=sha(OUT/'protocol.json')),dest/f'step_{step:05d}.pt')
   if step==STEPS:break
  batch=schedule[step];samples=[trainset.get(i,batch['epoch']) for i in batch['indices']]
  x,t,v=[torch.stack([a[j] for a in samples]).to(device) for j in range(3)]
  model.train();opt.zero_grad(set_to_none=True);p=predict(model,'ragan',x);adv=p.new_zeros(());dl=p.new_zeros(())
  if D is not None:
   D.train()
   for par in D.parameters():par.requires_grad_(True)
   dopt.zero_grad(set_to_none=True)
   rs=D(critic_input(t,v),x[:,:3]).mean();fs=D(critic_input(p.detach(),v),x[:,:3]).mean()
   dl=F.relu(1-rs)+F.relu(1+fs);assert torch.isfinite(dl);dl.backward();dopt.step();D.eval()
   for par in D.parameters():par.requires_grad_(False)
   judgment=D(critic_input(p,v),x[:,:3]).clamp(1e-6,1-1e-6)
   adv=F.binary_cross_entropy(judgment,torch.ones_like(judgment))
  sup=supervised_loss(p,t,v);loss=sup+.1*adv;assert torch.isfinite(loss);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
  running.append(dict(supervised=float(sup.detach()),adversarial=float(adv.detach()),D=float(dl.detach())))
 dump(dest/'completion.json',dict(condition=condition,additional_steps=STEPS,initial_optimizer_steps=[2160],
  final_optimizer_steps=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),
  D_optimizer_restored=conf['gan'],checkpoint_sha256=sha(diag.CHECKPOINT),
  schedule_sha256=sha(OUT/'batch_schedules'/(condition+'.json')),seconds=time.monotonic()-start,formal_accuracy=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train']);p.add_argument('--condition',choices=list(CONDITIONS));p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.action=='freeze':freeze()
 else:train(a.condition,a.device)
