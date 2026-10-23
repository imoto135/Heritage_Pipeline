"""Fixed four training images: can the two remaining weak traces be fitted?

Local memorization probe only; never an adopted model or held-out score.
"""
import json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_fine_retention_control import OUT as PARENT
from ragan_pilot_support import make_model,predict,auxiliaries,supervised_loss
from compare_v5_aligned_fade_control import dump,sha,rgb,save,coverage,mask
import diagnose_ragan_discriminator as diag
import apply_ragan_real_regions as real
OUT=PARENT/'hard_fine_fit_probe'
STEPS=864

def main():
 assert not OUT.exists();OUT.mkdir()
 records=full.read_rows('train');reviews=full.reviewed();meta=full.metadata();cases=[];samples=[]
 patches=['200017197_00103_1_p00','200017197_00214_1_p00']
 for patch in patches:
  idx=next(i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8)
  x,y=full.coordinates(idx,13,records[idx],meta);r=records[idx]
  t=full.gray(ROOT/r['labels']['target'])[y:y+256,x:x+256];v=full.gray(ROOT/r['labels']['valid'])[y:y+256,x:x+256];trace=reviews[patch][1][y:y+256,x:x+256]
  assert trace.sum()==reviews[patch][1].sum() and (trace&(v>0)&(t==0)).sum()==trace.sum()
  for family in ('identity','targeted_fade'):
   row=next(r for r in records if r['patch_id']==patch and r['damage_family']==family and (family=='identity' or float(r['damage_strength'])==.8))
   image=rgb(ROOT/row['input'])[y:y+256,x:x+256];source=rgb(ROOT/row['clean'])[y:y+256,x:x+256]
   cases.append(dict(patch=patch,family=family,input=image,source=source,target=t,valid=v,trace=trace,xy=[x,y],input_sha256=sha(ROOT/row['input']),trace_sha256=sha(reviews[patch][0])))
   samples.append((torch.from_numpy(auxiliaries(image).copy()).permute(2,0,1).float()/255,
    torch.from_numpy(t.copy()).unsqueeze(0).float()/255,torch.from_numpy(v.copy()).unsqueeze(0).float()/255))
 dump(OUT/'protocol.json',dict(checkpoint_sha256=sha(diag.CHECKPOINT),G_Adam_restored_step=2160,
  cases=[{k:c[k] for k in ('patch','family','xy','input_sha256','trace_sha256')} for c in cases],
  selection='two original reviewed train traces still weak after20% exposure; source selected after main readout',
  exploratory=True,one_fixed_batch4=True,auxiliaries_after_crop=True,steps=STEPS,eval_steps=[0,108,216,432,648,864],
  G_lr=3e-5,gradient_clip=1.,loss='unchanged shared supervised loss, no weighting',GAN=False,
  same_crop_for_clean_and_damaged=True,RGB_and_teacher_values_unchanged=True,
  heldout_data_used=False,memorization_only=True,not_generalization=True,not_a_candidate_model=True,formal_accuracy=False))
 shutil.copy2(__file__,OUT/Path(__file__).name)
 torch.set_num_threads(2);torch.manual_seed(20261022);device=torch.device('cuda:0');torch.cuda.set_device(device)
 payload=torch.load(diag.CHECKPOINT,map_location='cpu',weights_only=False);G=make_model('ragan').to(device);G.load_state_dict(payload['model'])
 opt=torch.optim.Adam([p for p in G.parameters() if p.requires_grad],lr=3e-5);opt.load_state_dict(payload['optimizer'])
 assert {int(s['step']) for s in opt.state.values() if 'step' in s}=={2160}
 x,t,v=[torch.stack([s[i] for s in samples]).to(device) for i in range(3)];history=[];start=time.monotonic()
 for step in range(STEPS+1):
  if step in (0,108,216,432,648,864):
   G.eval()
   with torch.inference_mode():prob=predict(G,'ragan',x);pred=prob[:,0].cpu().numpy()<.5;loss=float(supervised_loss(prob,t,v))
   rows=[]
   for i,(case,pp) in enumerate(zip(cases,pred)):
    save(OUT/'masks'/f'{step:05d}'/(case['patch']+'_'+case['family']+'.png'),pp)
    valid=case['valid']>0;target=case['target']<128
    tp=int((pp&target&valid).sum());fp=int((pp&~target&valid).sum());fn=int((~pp&target&valid).sum())
    rows.append(dict(patch=case['patch'],family=case['family'],tp=tp,fp=fp,fn=fn,**coverage(pp,case['trace'])))
   entry=dict(step=step,loss=loss,rows=rows,seconds=time.monotonic()-start);history.append(entry);dump(OUT/'history.json',history)
   print(step,'strong-fade r2',[(r['patch'],r['r2'],r['n']) for r in rows if r['family']=='targeted_fade'],'loss',loss,flush=True)
   if step==STEPS:break
  G.train();opt.zero_grad(set_to_none=True);p=predict(G,'ragan',x);loss=supervised_loss(p,t,v);assert torch.isfinite(loss);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(G.parameters(),1.);assert torch.isfinite(norm);opt.step()
 torch.save(dict(model=G.state_dict(),optimizer=opt.state_dict(),additional_steps=864,step=3024,memorization_probe=True),OUT/'last.pt')
 figures=[]
 for case in cases:
  teacher=case['target'].copy();teacher[case['valid']==0]=128;key=case['patch']+'_'+case['family']
  items=[('損傷前原本',case['source']),('固定入力 '+case['family'],case['input']),('暫定教師 / 不明は灰色',teacher)]
  for step in [0,216,432,864]:items.append((f'局所診断 +{step}更新',np.where(mask(OUT/'masks'/f'{step:05d}'/(key+'.png')),0,255).astype(np.uint8)))
  path=OUT/'figures'/(key+'_固定4枚への当てはまり.png');real.panel(path,'局所的な当てはまりのみ / '+key,items);figures.append(str(path.relative_to(PARENT)))
 dump(OUT/'results.json',dict(history=history,figures=figures,final_optimizer_step=sorted({int(s['step']) for s in opt.state.values() if 'step' in s}),initial_checkpoint_unchanged=sha(diag.CHECKPOINT)==json.loads((OUT/'protocol.json').read_text())['checkpoint_sha256'],not_generalization=True,not_candidate=True))
if __name__=='__main__':main()
