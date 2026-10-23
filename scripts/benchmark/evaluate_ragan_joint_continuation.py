"""Fixed readout of matched joint continuations, including collateral real cases."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_joint_continuation import OUT,CONDITIONS,arrays
import diagnose_ragan_discriminator as diag
from ragan_pilot_support import make_model,predict,auxiliaries
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,save,coverage
from audit_v5_probability_thresholds import ink_probability
from evaluate_ridge_gate_independent_val import proxies
from evaluate_contact_stain_counterfactual import make_field
import apply_ragan_real_regions as real
import evaluate_classical_partial_labels as scoring
PARENT=ROOT/'tmp_work/makura_contact_stain_train_control_20261022'
LABELS={'initial':'再開前','no_gan':'GANなし','reference_gan':'従来GAN','bce_native':'BCE / 生成器負例','bce_balanced':'BCE / 改変負例追加'}

def main(checkpoints_override=None,destination_name='evaluation',primary_step_override=None,device_name='cuda:0'):
 primary_step=primary_step_override
 protocol=json.loads((OUT/'protocol.json').read_text());primary_step=primary_step or protocol.get('primary_step',432);dest=OUT/destination_name;assert not (dest/'results.json').exists()
 torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device)
 classic={c['id']:c for c in json.loads((real.CLASSIC/'protocol.json').read_text())['cases']}
 realcases=[classic[k] for k in ('D01','D03','D04')]+json.loads((PARENT/'other_book_transfer/protocol.json').read_text())['cases']
 pages={c['id']:rgb(ROOT/c['source']) for c in realcases}
 for c in realcases:assert sha(ROOT/c['source'])==c['source_sha256']
 regions=json.loads((real.PARTIAL/'source_regions.json').read_text());ann={r['id']:r for r in json.loads((real.PARTIAL/'annotations_source_only.json').read_text())['regions']}
 prior=json.loads((ROOT/'tmp_work/ragan_convergence_audit_20261022/evaluation_protocol.json').read_text())
 synthetic=[]
 for patch in prior['synthetic_patches']:
  source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/patch/'clean.png')
  paper,core,remote,_=proxies(source);field,_=make_field(core)
  for family,inp in [('contact_stain',rgb(PARENT/'evaluation/inputs'/f'{patch}_stain.png')),('clean',source),
   ('uniform_fade_070',np.rint(source.astype(np.float32)*.3+paper.astype(np.float32)*.7).astype(np.uint8))]:
   active=field>=.5 if family=='contact_stain' else np.ones(core.shape,bool)
   synthetic.append(dict(id=patch+'_'+family,input=inp,source=source,family=family,core=core&active,paper=remote&active))
 trace=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(prior['known_traced_fade_patch']+'_high_centers.png')).convert('L'))>0
 fade=rgb(ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70/input.png')
 native_rows,native_arrays=arrays('val')
 records=[];synrows=[];thinrows=[];figures=[]
 checkpoints=checkpoints_override or {'initial':diag.CHECKPOINT,**{k:OUT/k/f'step_{primary_step:05d}.pt' for k in protocol['conditions']}}
 for condition,checkpoint in checkpoints.items():
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False);model=make_model('ragan').to(device);model.load_state_dict(payload['model']);model.eval()
  for row,a in zip(native_rows,native_arrays):
   inp=torch.from_numpy(auxiliaries(a['image'])).permute(2,0,1).unsqueeze(0).float().to(device)/255
   with torch.inference_mode():p=predict(model,'ragan',inp)[0,0].cpu().numpy()<.5
   save(dest/'native'/condition/(row['patch']+'.png'),p)
  for case in synthetic:
   pred=ink_probability(model,'ragan',case['input'],device)>=.5;dist=ndi.distance_transform_edt(~pred)
   save(dest/'synthetic'/condition/(case['id']+'.png'),pred)
   synrows.append(dict(condition=condition,id=case['id'],family=case['family'],core_n=int(case['core'].sum()),
    core_r2=int(((dist<=2)&case['core']).sum()),paper_n=int(case['paper'].sum()),paper_ink=int((pred&case['paper']).sum())))
  p=ink_probability(model,'ragan',fade,device)>=.5;thinrows.append(dict(condition=condition,**coverage(p,trace)));save(dest/'traced_fade'/(condition+'.png'),p)
  for case in realcases:
   key=case['id'];box=case.get('roi',case.get('crop_xywh'));x,y,w,h=box
   pred=real.infer_roi(model,'ragan',pages[key],box,device)<.5;path=dest/'real'/condition/(key+'.png');save(path,pred)
   if key in ('D01','D03'):
    for reg in [r for r in regions if r['case']==key]:
     rx,ry,rw,rh=reg['roi'];lab,paths=scoring.raster(reg,ann[reg['id']]);m=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
     records.append(dict(condition=condition,case=key,region=reg['id'],**scoring.measure(m,lab,paths)))
   else:
    if key=='D04':rbox=json.loads((PARENT/'d04_spot_control/protocol.json').read_text())['spot_box_xywh']
    else:rbox=json.loads((PARENT/'other_book_transfer/protocol.json').read_text())['T07_stain_paper_box_xywh' if key=='T07' else 'C01_spot_box_xywh']
    rx,ry,rw,rh=rbox;records.append(dict(condition=condition,case=key,region='selected_spot_or_stain',nonink_n=rw*rh,nonink_false=int(pred[ry:ry+rh,rx:rx+rw].sum())))
  print(condition,'evaluated',flush=True);del model,payload;torch.cuda.empty_cache()
 for case in realcases:
  key=case['id'];x,y,w,h=case.get('roi',case.get('crop_xywh'));items=[('原画像',pages[key][y:y+h,x:x+w])]
  hp=real.CLASSIC/'crops'/key/'howe_fixed.png'
  if not hp.exists():hp=PARENT/'other_book_transfer'/'masks'/key/'howe_fixed.png'
  if hp.exists():items.append(('Howe固定',np.where(mask(hp),0,255).astype(np.uint8)))
  for condition in checkpoints:items.append((LABELS[condition],np.where(mask(dest/'real'/condition/(key+'.png')),0,255).astype(np.uint8)))
  path=dest/'figures'/(key+'_原本比較.png');real.panel(path,key+f' / 同じ生成器から{primary_step}更新',items);figures.append(str(path.relative_to(OUT)))
 for row,a in zip(native_rows,native_arrays):
  items=[('損傷前原本',a['source']),('合成入力',a['image'])]
  teacher=np.rint(a['target']*255).astype(np.uint8);teacher[~a['valid']]=128;items.append(('暫定教師 / 不明は灰色',teacher))
  for condition in checkpoints:items.append((LABELS[condition],np.where(mask(dest/'native'/condition/(row['patch']+'.png')),0,255).astype(np.uint8)))
  path=dest/'figures'/(row['patch']+'_原本比較.png');real.panel(path,row['patch']+' / 学習外ページ・既知開発例',items);figures.append(str(path.relative_to(OUT)))
 for case in synthetic:
  if case['family']!='contact_stain':continue
  items=[('損傷前原本',case['source']),('合成入力',case['input'])]
  for condition in checkpoints:items.append((LABELS[condition],np.where(mask(dest/'synthetic'/condition/(case['id']+'.png')),0,255).astype(np.uint8)))
  path=dest/'figures'/(case['id']+'_512比較.png');real.panel(path,case['id']+' / 512画素での波及確認',items);figures.append(str(path.relative_to(OUT)))
 items=[('退色前原本',rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/prior['known_traced_fade_patch']/'clean.png')),('退色入力 .70',fade),('Howe固定',np.where(mask(ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70/howe_fixed.png'),0,255).astype(np.uint8))]
 for condition in checkpoints:items.append((LABELS[condition],np.where(mask(dest/'traced_fade'/(condition+'.png')),0,255).astype(np.uint8)))
 path=dest/'figures'/'退色文字_原本比較.png';real.panel(path,'既知細線49点 / 接触汚れへの学習の波及',items);figures.append(str(path.relative_to(OUT)))
 dump(dest/'results.json',dict(real=records,synthetic=synrows,traced_fade=thinrows,
  checkpoint_sha256={k:sha(v) for k,v in checkpoints.items()},figures=figures,formal_accuracy=False))
 print(json.dumps(dict(real=records,synthetic=synrows,traced_fade=thinrows),ensure_ascii=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');a=p.parse_args()
 if a.replay:OUT=ROOT/'tmp_work/ragan_joint_replay_20261022'
 main()
