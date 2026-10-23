"""Frozen readouts for fine-retention controls; reuse verified old baseline outputs."""
from __future__ import annotations
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_fine_retention_control import OUT,CONDITIONS,BASELINES
from ragan_pilot_support import make_model,predict,auxiliaries
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,save,coverage
from audit_v5_probability_thresholds import ink_probability
import evaluate_ragan_joint_continuation as common
import apply_ragan_real_regions as real
import diagnose_ragan_discriminator as diag
LABELS={'initial':'再開前10周',**{f'baseline_order{n}':f'通常学習 順序{n+1}' for n in (0,1)},
 **{f'{factor}_order{n}':label+f' 順序{n+1}' for n in (0,1) for factor,label in [('core8','墨芯重み8倍'),('exposure20','強退色の提示20%')]}}
ORDER=list(LABELS)

def copy_tree(source,dest):
 dest.mkdir(parents=True,exist_ok=True)
 for path in source.glob('*.png'):shutil.copy2(path,dest/path.name)

def inherited(step,dest,results):
 directory='evaluation' if step==864 else 'evaluation_00432';old=full.OUT/directory
 before=json.loads((old/'results.json').read_text());mapping={'initial':'initial','no_gan_order0':'baseline_order0','no_gan_order1':'baseline_order1'}
 for oldc,newc in mapping.items():
  for kind in ('native','synthetic','real'):copy_tree(old/kind/oldc,dest/kind/newc)
  (dest/'traced_fade').mkdir(exist_ok=True);shutil.copy2(old/'traced_fade'/(oldc+'.png'),dest/'traced_fade'/(newc+'.png'))
  for kind in ('real','synthetic','traced_fade'):results[kind].extend(dict(r,condition=newc) for r in before[kind] if r['condition']==oldc)
  results['checkpoint_sha256'][newc]=before['checkpoint_sha256'][oldc]
 return old,mapping

def compose(dest):
 protocol=json.loads((full.OUT/'evaluation_protocol.json').read_text());prior=json.loads((ROOT/protocol['inherit_evaluation_from']).read_text())
 classic={c['id']:c for c in json.loads((real.CLASSIC/'protocol.json').read_text())['cases']}
 cases=[classic[k] for k in ('D01','D03','D04')]+json.loads((common.PARENT/'other_book_transfer/protocol.json').read_text())['cases'];figures=[]
 def outputs(folder,key):return [(LABELS[c],np.where(mask(dest/folder/c/(key+'.png')),0,255).astype(np.uint8)) for c in ORDER]
 for case in cases:
  key=case['id'];page=rgb(ROOT/case['source']);x,y,w,h=case.get('roi',case.get('crop_xywh'))
  hp=real.CLASSIC/'crops'/key/'howe_fixed.png'
  if not hp.exists():hp=common.PARENT/'other_book_transfer/masks'/key/'howe_fixed.png'
  items=[('原画像',page[y:y+h,x:x+w]),('Howe固定',np.where(mask(hp),0,255).astype(np.uint8))]+outputs('real',key)
  path=dest/'figures'/(key+'_原本比較.png');real.panel(path,key+' / 損傷と教師を固定したGANなし対照',items);figures.append(str(path.relative_to(OUT)))
 for patch in prior['synthetic_patches']:
  source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/patch/'clean.png');inp=rgb(common.PARENT/'evaluation/inputs'/(patch+'_stain.png'));key=patch+'_contact_stain'
  path=dest/'figures'/(key+'_512比較.png');real.panel(path,key,[('原本',source),('合成汚れ入力',inp)]+outputs('synthetic',key));figures.append(str(path.relative_to(OUT)))
 for row,a in zip(*common.arrays('val')):
  teacher=np.rint(a['target']*255).astype(np.uint8);teacher[~a['valid']]=128
  path=dest/'figures'/(row['patch']+'_原本比較.png');real.panel(path,row['patch']+' / 学習外の既知開発例',[('原本',a['source']),('合成入力',a['image']),('暫定教師 / 不明は灰色',teacher)]+outputs('native',row['patch']));figures.append(str(path.relative_to(OUT)))
 folder=ROOT/'tmp_work/targeted_thin_fade_20261021/cases/alpha_0.70'
 items=[('退色前原本',rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/prior['known_traced_fade_patch']/'clean.png')),('退色入力 .70',rgb(folder/'input.png')),('Howe固定',np.where(mask(folder/'howe_fixed.png'),0,255).astype(np.uint8))]
 for c in ORDER:items.append((LABELS[c],np.where(mask(dest/'traced_fade'/(c+'.png')),0,255).astype(np.uint8)))
 path=dest/'figures'/'退色文字_原本比較.png';real.panel(path,'学習外の既知細線49点 / 教師ありの損失と提示頻度',items);figures.append(str(path.relative_to(OUT)))
 return figures

def main(step,device_name):
 p=json.loads((OUT/'evaluation_protocol.json').read_text());assert sha(ROOT/p['inherit_evaluation_from'])==p['inherit_sha256']
 destination='evaluation' if step==864 else f'evaluation_{step:05d}';dest=OUT/destination
 common.OUT=OUT;common.LABELS.update(LABELS)
 ck={c:OUT/c/f'step_{step:05d}.pt' for c in CONDITIONS}
 common.main(checkpoints_override=ck,destination_name=destination,primary_step_override=step,device_name=device_name)
 results=json.loads((dest/'results.json').read_text());old,mapping=inherited(step,dest,results)
 results['figures']=compose(dest);results['baseline_cache_source']=str(old.relative_to(ROOT));results['baseline_cache_results_sha256']=sha(old/'results.json');dump(dest/'results.json',results)
 records=full.read_rows('train');reviews=full.reviewed();inputs=[]
 for patch,(tracepath,trace) in reviews.items():
  for family in ('identity','targeted_fade'):
   r=next(r for r in records if r['patch_id']==patch and r['damage_family']==family and (family=='identity' or float(r['damage_strength'])==.8))
   inputs.append(dict(patch=patch,family=family,input=rgb(ROOT/r['input']),source=rgb(ROOT/r['clean']),trace=trace,input_sha256=sha(ROOT/r['input']),trace_sha256=sha(tracepath)))
 baseline=json.loads((old/'training_reviewed_results.json').read_text());rows=[dict(r,condition=mapping[r['condition']]) for r in baseline['rows'] if r['condition'] in mapping]
 for oldc,newc in mapping.items():copy_tree(old/'training_reviewed_masks'/oldc,dest/'training_reviewed_masks'/newc)
 torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device);native_rows=[];meta=full.metadata()
 for c,path in ck.items():
  G=make_model('ragan').to(device);G.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);G.eval()
  for case in inputs:
   pred=ink_probability(G,'ragan',case['input'],device)>=.5;save(dest/'training_reviewed_masks'/c/(case['patch']+'_'+case['family']+'.png'),pred)
   rows.append(dict(condition=c,patch=case['patch'],family=case['family'],input_sha256=case['input_sha256'],trace_sha256=case['trace_sha256'],**coverage(pred,case['trace'])))
  if step==864:
   schedule=json.loads((OUT/'batch_schedules'/(c+'.json')).read_text())
   for patch,(_,trace) in reviews.items():
    index=next(i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8)
    batch=[b for b in schedule if b['epoch']==13 and index in b['indices']][-1];position=batch['indices'].index(index);samples=[]
    for i in batch['indices']:
     r=records[i];x,y=full.coordinates(i,13,r,meta);a=rgb(ROOT/r['input'])[y:y+256,x:x+256];samples.append(torch.from_numpy(auxiliaries(a).copy()).permute(2,0,1).float()/255)
    x,y=full.coordinates(index,13,records[index],meta);sub=trace[y:y+256,x:x+256];assert sub.sum()==trace.sum();actual=torch.stack(samples).to(device)
    with torch.inference_mode():
     predictions={'actual_batch4':predict(G,'ragan',actual)[position,0].cpu().numpy()<.5,
      'single_crop1':predict(G,'ragan',actual[position:position+1].contiguous())[0,0].cpu().numpy()<.5}
    for view,pred in predictions.items():
     save(dest/'training_native_masks'/c/(patch+'_'+view+'.png'),pred)
     native_rows.append(dict(condition=c,patch=patch,view=view,xy=[x,y],epoch=13,batch_indices=batch['indices'],position=position,occurrence='last target occurrence in final epoch',**coverage(pred,sub)))
  del G;torch.cuda.empty_cache()
 figures=[]
 for case in inputs:
  if case['family']!='targeted_fade':continue
  items=[('学習原本',case['source']),('学習入力 退色.80',case['input'])]
  for c in ORDER:items.append((LABELS[c],np.where(mask(dest/'training_reviewed_masks'/c/(case['patch']+'_'+case['family']+'.png')),0,255).astype(np.uint8)))
  path=dest/'figures'/(case['patch']+'_学習細線比較.png');real.panel(path,'学習側の細線 / '+case['patch'],items);figures.append(str(path.relative_to(OUT)))
 dump(dest/'training_reviewed_results.json',dict(rows=rows,figures=figures,training_fit_only=True,formal_accuracy=False,baseline_cache_sha256=sha(old/'training_reviewed_results.json')))
 if step==864:
  old_native=json.loads((full.OUT/'training_context_audit/results.json').read_text())
  native_mapping={'initial_order0':'initial','no_gan_order0':'baseline_order0','no_gan_order1':'baseline_order1'}
  native_rows.extend(dict(r,condition=native_mapping[r['condition']]) for r in old_native['rows'] if r['condition'] in native_mapping and r['view'] in ('actual_batch4','single_crop1'))
  for oldc,newc in native_mapping.items():
   for patch in reviews:
    for view in ('actual_batch4','single_crop1'):
     source=full.OUT/'training_context_audit/masks'/oldc/(patch+'_'+view+'.png');folder=dest/'training_native_masks'/newc;folder.mkdir(parents=True,exist_ok=True);shutil.copy2(source,folder/source.name)
  native_figures=[]
  for patch,(_,trace) in reviews.items():
   index=next(i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8);r=records[index];x,y=full.coordinates(index,13,r,meta)
   source=rgb(ROOT/r['clean'])[y:y+256,x:x+256];image=rgb(ROOT/r['input'])[y:y+256,x:x+256];t=full.gray(ROOT/r['labels']['target'])[y:y+256,x:x+256].copy();v=full.gray(ROOT/r['labels']['valid'])[y:y+256,x:x+256]>0;t[~v]=128
   items=[('原本の学習切り出し',source),('強退色.80入力',image),('暫定教師 / 不明は灰色',t)]
   for c in ORDER:items.append((LABELS[c],np.where(mask(dest/'training_native_masks'/c/(patch+'_actual_batch4.png')),0,255).astype(np.uint8)))
   path=dest/'figures'/(patch+'_256学習窓比較.png');real.panel(path,'実際に提示した256画素の窓 / '+patch,items);native_figures.append(str(path.relative_to(OUT)))
  dump(dest/'training_native_results.json',dict(rows=native_rows,figures=native_figures,training_fit_only=True,not_generalization=True,baseline_cache_sha256=sha(full.OUT/'training_context_audit/results.json')))
 print('Fine-retention controls evaluated',step,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--step',type=int,choices=[432,864],default=864);a.add_argument('--device',default='cuda:0');args=a.parse_args();main(args.step,args.device)
