"""Matched joint G/D continuation on the frozen critic-diagnostic training crops.

A small stationary diagnostic, not full-data training or a paper reproduction.
"""
from __future__ import annotations
import argparse,json,sys,time,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import diagnose_ragan_discriminator as diag
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss,critic_input
from compare_v5_aligned_fade_control import sha,dump
OUT=ROOT/'tmp_work/ragan_joint_continuation_20261022'
CONDITIONS={
 'no_gan':dict(objective=None,lr=None),
 'reference_gan':dict(objective='sigmoid_scalar_hinge',lr=1e-6),
 'bce_native':dict(objective='sigmoid_bce',lr=1e-4),
 'bce_balanced':dict(objective='sigmoid_bce',lr=1e-4,balanced=True),
}
STEPS=432

def freeze(replay=False):
 assert not OUT.exists();OUT.mkdir()
 index=json.loads((diag.OUT/'index.json').read_text())['rows']
 dump(OUT/'protocol.json',dict(conditions=CONDITIONS,seed=diag.SEED,steps=STEPS,batch_size=4,
  checkpoint=str(diag.CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=sha(diag.CHECKPOINT),
  dataset_index_sha256=sha(diag.OUT/'index.json'),train_examples=24,val_examples=8,
  training='fixed 24 pages/crops; clean and .70 fade replay added' if replay else 'fixed 24 dark-stain crops from prior D diagnosis; no augmentation or val updates',
  training_families=['stain','clean','fade070'] if replay else ['stain'],
  replay_family_schedule='family=(step+batch_position)%3' if replay else None,
  common_G_weights=True,common_G_Adam_restored=True,G_lr=3e-5,G_gradient_clip=1.,G_adversarial_weight=.1,
  common_D_weights=True,all_D_Adam_new=True,D_update_before_G=True,
  balanced='one each current generator / added stain / erased overlap ink / same-area relocation per batch; cyclic positions',
  native_fake='current generator, detached for D; never frozen cached generator output',
  eval_steps=[0,108,216,324,432],saved_steps=[216,432],primary_step=432,threshold=.5,
  real_cases=['D01','D03','D04','T07','C01'],known_development=True,formal_accuracy=False,
  labels_provisional=True,fixed_crops_repeated=True,one_seed=True,
  inference='per image original tensor construction, after-crop auxiliaries',
  control_scope='BCE native versus balanced isolates negative distribution; reference versus BCE changes objective and D lr',
  rows=index))
 shutil.copy2(__file__,OUT/Path(__file__).name)
 if replay:
  _,aa=arrays('train');dump(OUT/'replay_inputs.json',[dict(index=i,family=f,sha256=hashlib.sha256(a.tobytes()).hexdigest()) for i,d in enumerate(aa) for f,a in replay_images(d).items()])
 print('Frozen joint protocol',flush=True)

def arrays(split):
 rows=[r for r in json.loads((diag.OUT/'index.json').read_text())['rows'] if r['split']==split]
 result=[]
 for r in rows:
  path=diag.OUT/r['cache'];assert sha(path)==r['cache_sha256'];result.append(dict(np.load(path)))
 return rows,result

def replay_images(a):
 from evaluate_ridge_gate_independent_val import proxies
 source=a['source'];paper,*_=proxies(source)
 fade=np.rint(source.astype(np.float32)*.3+paper.astype(np.float32)*.7).astype(np.uint8)
 return dict(stain=a['image'],clean=source,fade070=fade)

def evaluate_native(model,device):
 model.eval();rows,aa=arrays('val');result=[]
 with torch.inference_mode():
  for row,a in zip(rows,aa):
   x=torch.from_numpy(auxiliaries(a['image'])).permute(2,0,1).unsqueeze(0).float().to(device)/255
   p=predict(model,'ragan',x)[0,0].cpu().numpy();ink=p<.5;t=a['target']<.5;v=a['valid']
   tp=int((ink&t&v).sum());fp=int((ink&~t&v).sum());fn=int((~ink&t&v).sum())
   result.append(dict(patch=row['patch'],tp=tp,fp=fp,fn=fn,
    overlap_ink_kept=int((ink&a['erase_region']).sum()),overlap_ink_n=int(a['erase_region'].sum()),
    stain_paper_ink=int((ink&a['add_region']).sum()),stain_paper_n=int(a['add_region'].sum())))
 totals={k:sum(r[k] for r in result) for k in result[0] if k!='patch'}
 totals['ink_f1_to_provisional']=2*totals['tp']/max(1,2*totals['tp']+totals['fp']+totals['fn'])
 return dict(rows=result,total=totals)

def train(condition,device_name):
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['checkpoint_sha256']==sha(diag.CHECKPOINT)
 conf=CONDITIONS[condition];dest=OUT/condition;assert not dest.exists();dest.mkdir()
 torch.set_num_threads(2);torch.manual_seed(diag.SEED);device=torch.device(device_name);torch.cuda.set_device(device)
 payload=torch.load(diag.CHECKPOINT,map_location='cpu',weights_only=False)
 model=make_model('ragan').to(device);model.load_state_dict(payload['model'])
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=3e-5);opt.load_state_dict(payload['optimizer'])
 assert all(g['lr']==3e-5 for g in opt.param_groups)
 initial_optimizer_steps=sorted(set(int(s['step']) for s in opt.state.values() if 'step' in s));assert initial_optimizer_steps==[2160]
 D=diag.get_discriminator(payload,conf['objective'],device) if conf['objective'] else None
 dopt=torch.optim.Adam(D.parameters(),lr=conf['lr']) if D is not None else None
 data=diag.dataset('train',device);_,aa=arrays('train')
 replay=len(protocol['training_families'])>1
 if replay:
  imgs=[replay_images(a) for a in aa]
  frozen=json.loads((OUT/'replay_inputs.json').read_text())
  assert all(hashlib.sha256(imgs[r['index']][r['family']].tobytes()).hexdigest()==r['sha256'] for r in frozen)
  inputs=torch.from_numpy(np.stack([[auxiliaries(a[f]) for a in imgs] for f in protocol['training_families']])).permute(0,1,4,2,3).float().to(device)/255
 else:inputs=torch.from_numpy(np.stack([auxiliaries(a['image']) for a in aa])).permute(0,3,1,2).float().to(device)/255
 rng=np.random.default_rng(diag.SEED);history=[];start=time.monotonic();losses=[];schedule=[]
 for step in range(STEPS+1):
  if step in protocol['eval_steps']:
   entry=dict(step=step,validation=evaluate_native(model,device),seconds=time.monotonic()-start)
   if losses:entry['training_mean']={k:float(np.mean([r[k] for r in losses])) for k in losses[0]};losses=[]
   history.append(entry);dump(dest/'history.json',history)
   print(condition,step,entry['validation']['total'],'seconds',round(entry['seconds'],1),flush=True)
   if step in protocol['saved_steps']:
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=2160+step,additional_steps=step,
     discriminator=D.state_dict() if D is not None else None,d_optimizer=dopt.state_dict() if dopt else None,
     condition=condition,protocol_sha256=sha(OUT/'protocol.json')),dest/f'step_{step:05d}.pt')
   if step==STEPS:break
  idx=rng.choice(24,4,replace=False);schedule.append(idx.tolist());model.train();opt.zero_grad(set_to_none=True)
  x=inputs[(step+np.arange(4))%3,idx] if replay else inputs[idx]
  target=data['real'][idx];valid=data['valid'][idx];p=predict(model,'ragan',x)
  adv=p.new_zeros(());dl=p.new_zeros(())
  if D is not None:
   D.train()
   for par in D.parameters():par.requires_grad_(True)
   dopt.zero_grad(set_to_none=True)
   rs=D(critic_input(target,valid),x[:,:3])
   if conf.get('balanced'):
    keys=[None,'added_stain','erased_overlap_ink','same_area_relocation']
    fake=torch.stack([p[i].detach() if keys[(step+i)%4] is None else data[keys[(step+i)%4]][j] for i,j in enumerate(idx)])
   else:fake=p.detach()
   fs=D(critic_input(fake,valid),x[:,:3])
   if conf['objective']=='sigmoid_scalar_hinge':dl=F.relu(1-rs.mean())+F.relu(1+fs.mean())
   else:dl=F.binary_cross_entropy(rs,torch.ones_like(rs))+F.binary_cross_entropy(fs,torch.zeros_like(fs))
   assert torch.isfinite(dl);dl.backward();dopt.step();D.eval()
   for par in D.parameters():par.requires_grad_(False)
   judgment=D(critic_input(p,valid),x[:,:3]).clamp(1e-6,1-1e-6)
   adv=F.binary_cross_entropy(judgment,torch.ones_like(judgment))
  sup=supervised_loss(p,target,valid);loss=sup+.1*adv;assert torch.isfinite(loss);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
  losses.append(dict(supervised=float(sup.detach()),adversarial=float(adv.detach()),D=float(dl.detach())))
 dump(dest/'schedule.json',schedule)
 dump(dest/'completion.json',dict(condition=condition,additional_steps=STEPS,initial_optimizer_steps=initial_optimizer_steps,
  final_optimizer_steps=sorted(set(int(s['step']) for s in opt.state.values() if 'step' in s)),
  checkpoint_sha256=sha(diag.CHECKPOINT),schedule_sha256=sha(dest/'schedule.json'),
  seconds=time.monotonic()-start,formal_accuracy=False))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','train']);parser.add_argument('--condition',choices=list(CONDITIONS));parser.add_argument('--device',default='cuda:0');parser.add_argument('--replay',action='store_true');a=parser.parse_args()
 if a.replay:
  OUT=ROOT/'tmp_work/ragan_joint_replay_20261022';CONDITIONS={k:v for k,v in CONDITIONS.items() if k!='reference_gan'}
 if a.action=='freeze':freeze(a.replay)
 else:train(a.condition,a.device)
