"""Fixed full512/real readout, plus fit of the seven reviewed training traces."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_full_continuation import OUT,read_rows,reviewed,CONDITIONS
from compare_v5_aligned_fade_control import sha,dump,rgb,save,coverage,mask
from ragan_pilot_support import make_model
from audit_v5_probability_thresholds import ink_probability
import evaluate_ragan_joint_continuation as common
import diagnose_ragan_discriminator as diag
import apply_ragan_real_regions as real
LABELS={f'{kind}_order{n}':label+f' 順序{n+1}' for n in (0,1) for kind,label in [('no_gan','GANなし'),('reference_gan','従来GAN')]}

def main(step,device_name):
 protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
 assert sha(ROOT/protocol['inherit_evaluation_from'])==protocol['inherit_sha256']
 assert step in (protocol['primary_step'],protocol['secondary_step'])
 common.OUT=OUT;common.LABELS.update(LABELS)
 destination='evaluation' if step==protocol['primary_step'] else f'evaluation_{step:05d}'
 common.main(destination_name=destination,primary_step_override=step,device_name=device_name)
 torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device)
 inputs=[];rows=read_rows('train');reviews=reviewed()
 for patch,(path,trace) in reviews.items():
  row=next(r for r in rows if r['patch_id']==patch)
  for family in ('identity','targeted_fade'):
   r=next(r for r in rows if r['patch_id']==patch and r['damage_family']==family and (family=='identity' or float(r['damage_strength'])==.8))
   inputs.append(dict(patch=patch,family=family,image=rgb(ROOT/r['input']),source=rgb(ROOT/r['clean']),trace=trace,input_sha256=sha(ROOT/r['input']),trace_sha256=sha(path)))
 checkpoints={'initial':diag.CHECKPOINT,**{k:OUT/k/f'step_{step:05d}.pt' for k in CONDITIONS}}
 measures=[];dest=OUT/destination
 for condition,path in checkpoints.items():
  model=make_model('ragan').to(device);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval()
  for case in inputs:
   pred=ink_probability(model,'ragan',case['image'],device)>=.5
   save(dest/'training_reviewed_masks'/condition/(case['patch']+'_'+case['family']+'.png'),pred)
   measures.append(dict(condition=condition,patch=case['patch'],family=case['family'],input_sha256=case['input_sha256'],trace_sha256=case['trace_sha256'],**coverage(pred,case['trace'])))
  del model;torch.cuda.empty_cache()
 figures=[]
 for case in inputs:
  if case['family']!='targeted_fade':continue
  items=[('学習原本',case['source']),('学習入力 退色.80',case['image'])]
  for condition in checkpoints:
   label='再開前' if condition=='initial' else LABELS[condition]
   items.append((label,np.where(mask(dest/'training_reviewed_masks'/condition/(case['patch']+'_'+case['family']+'.png')),0,255).astype(np.uint8)))
  path=dest/'figures'/(case['patch']+'_学習細線比較.png');real.panel(path,'学習側の確認済み細線 / '+case['patch'],items);figures.append(str(path.relative_to(OUT)))
 dump(dest/'training_reviewed_results.json',dict(rows=measures,figures=figures,training_fit_only=True,not_generalization=True,formal_accuracy=False))
 print('Reviewed training traces evaluated',step,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[432,864],default=864);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.step,a.device)
