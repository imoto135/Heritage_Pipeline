"""Read-only comparison: exact seen crop/batch, replicated crop, single crop, full512."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_full_continuation import OUT,read_rows,metadata,coordinates,reviewed,gray
from compare_v5_aligned_fade_control import sha,dump,rgb,save,coverage,mask
from ragan_pilot_support import make_model,predict,auxiliaries
import diagnose_ragan_discriminator as diag
import apply_ragan_real_regions as real

def main():
 dest=OUT/'training_context_audit';assert not dest.exists();dest.mkdir()
 records=read_rows('train');meta=metadata();reviews=reviewed();device=torch.device('cuda:0');torch.cuda.set_device(device);torch.set_num_threads(2)
 protocol=dict(epoch=13,input_size=256,train_batch_size=4,selection='all7 reviewed patches, targeted_fade.80, actual frozen final-epoch batch',
  views=['actual_batch4','same_crop_repeated4','single_crop1','full512_single'],no_updates=True,
  all_256_batches_contiguous_NCHW=True,threshold=.5,full512='existing fixed single-image output; also changes spatial/auxiliary context',
  inference_audit_only=True,not_generalization=True,formal_accuracy=False)
 dump(dest/'protocol.json',protocol)
 allrows=[];figures=[]
 configurations=[('initial_order0',0,diag.CHECKPOINT,'initial'),('initial_order1',1,diag.CHECKPOINT,'initial')]
 configurations += [(f'{kind}_order{n}',n,OUT/f'{kind}_order{n}'/'step_00864.pt',f'{kind}_order{n}') for n in (0,1) for kind in ('no_gan','reference_gan')]
 for condition,order,ck,fullcondition in configurations:
  G=make_model('ragan').to(device);G.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);G.eval()
  schedule=json.loads((OUT/'batch_schedules'/f'no_gan_order{order}.json').read_text())
  for patch,(_,trace) in reviews.items():
   index=next(i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8)
   b=next(b for b in schedule if b['epoch']==13 and index in b['indices']);position=b['indices'].index(index);samples=[]
   for i in b['indices']:
    r=records[i];x,y=coordinates(i,13,r,meta);a=rgb(ROOT/r['input'])[y:y+256,x:x+256]
    samples.append(torch.from_numpy(auxiliaries(a).copy()).permute(2,0,1).float()/255)
   x,y=coordinates(index,13,records[index],meta);sub=trace[y:y+256,x:x+256];assert sub.sum()==trace.sum()
   actual=torch.stack(samples).to(device);single=actual[position:position+1].contiguous();repeated=single.expand(4,-1,-1,-1).contiguous()
   assert actual.is_contiguous() and repeated.is_contiguous() and single.is_contiguous()
   outputs={}
   with torch.inference_mode():
    outputs['actual_batch4']=predict(G,'ragan',actual)[position,0].cpu().numpy()<.5
    outputs['same_crop_repeated4']=predict(G,'ragan',repeated)[0,0].cpu().numpy()<.5
    outputs['single_crop1']=predict(G,'ragan',single)[0,0].cpu().numpy()<.5
   fullpath=OUT/'evaluation/training_reviewed_masks'/fullcondition/(patch+'_targeted_fade.png');outputs['full512_single']=mask(fullpath)[y:y+256,x:x+256]
   for view,pred in outputs.items():
    path=dest/'masks'/condition/(patch+'_'+view+'.png');save(path,pred)
    allrows.append(dict(condition=condition,order=order,patch=patch,view=view,batch_indices=b['indices'],position=position,xy=[x,y],checkpoint_sha256=sha(ck),**coverage(pred,sub)))
   if condition.startswith('reference_gan'):
    r=records[index];source=rgb(ROOT/r['clean'])[y:y+256,x:x+256];image=rgb(ROOT/r['input'])[y:y+256,x:x+256]
    teacher=gray(ROOT/r['labels']['target'])[y:y+256,x:x+256].copy();valid=gray(ROOT/r['labels']['valid'])[y:y+256,x:x+256]>0;teacher[~valid]=128
    items=[('原本の学習切り出し',source),('退色.80入力',image),('暫定教師 / 不明は灰色',teacher)]
    for view,label in [('actual_batch4','学習時と同じ4枚'),('same_crop_repeated4','同じ1枚を4回複製'),('single_crop1','1枚だけ / 256推論'),('full512_single','512推論の同じ場所')]:items.append((label,np.where(outputs[view],0,255).astype(np.uint8)))
    path=dest/'figures'/(condition+'_'+patch+'.png');real.panel(path,patch+' / 入力と重み固定・バッチ/範囲の比較',items);figures.append(str(path.relative_to(OUT)))
  del G;torch.cuda.empty_cache()
 dump(dest/'results.json',dict(rows=allrows,figures=figures,no_updates=True,formal_accuracy=False))
 for c,_,_,_ in configurations:
  print(c,[(v,sum(r['r2'] for r in allrows if r['condition']==c and r['view']==v)) for v in protocol['views']],flush=True)
if __name__=='__main__':main()
