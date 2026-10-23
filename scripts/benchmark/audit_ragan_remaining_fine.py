"""Audit unchanged labels and gradient interference on the residual44 example."""
import json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_stable_bce_control import stable_loss,product_logit
from run_ragan_stable_exposure_control import OUT as PARENT
from ragan_local_region_support import region_masks,local_terms
from ragan_pilot_support import make_model,predict,auxiliaries
from compare_v5_aligned_fade_control import dump,sha,rgb
import apply_ragan_real_regions as real
OUT=ROOT/'tmp_work/ragan_local_region_control_20261022'

def dot(left,right):return sum(float((a.double()*b.double()).sum()) for a,b in zip(left,right) if a is not None and b is not None)
def norm(g):return max(dot(g,g),0.)**.5

def main():
 OUT.mkdir(exist_ok=True);dest=OUT/'audit';assert not dest.exists();dest.mkdir()
 records=full.read_rows('train');meta=full.metadata();regions=region_masks(records);patch='200017197_00214_1_p00';reviews=full.reviewed()
 label_audit=[]
 for key,masks in regions.items():
  near=__import__('scipy').ndimage.distance_transform_edt(~masks['core'])<=2
  label_audit.append(dict(patch=key,core_pixels=int(masks['core'].sum()),roi_box_xyxy=masks['box'],
   existing_paper=int(masks['known_paper'].sum()),source_remote_paper_proxy=int(masks['paper'].sum()),
   dark_paper_proxy_ge15=int((masks['known_paper']&(masks['contrast']>=15)).sum()),
   near_core_dark_paper_proxy_ge15=int((masks['known_paper']&near&(masks['contrast']>=15)).sum()),
   no_labels_changed=True,dark_paper_flags_not_verified_errors=True))
 dump(dest/'label_audit.json',label_audit)
 torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);result=[];figures=[]
 index=next(i for i,r in enumerate(records) if r['patch_id']==patch and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8)
 for order in (0,1):
  ck=PARENT/f'stable_exposure_order{order}'/'step_00864.pt';payload=torch.load(ck,map_location='cpu',weights_only=False)
  G=make_model('ragan').to(device);G.load_state_dict(payload['model']);G.eval();capture={};G.corase_c6s.gate01.gate01.register_forward_hook(lambda m,i,o:capture.update(raw=o))
  schedule=json.loads((PARENT/'batch_schedules'/f'stable_exposure_order{order}.json').read_text());batch=[b for b in schedule if b['epoch']==13 and index in b['indices']][-1];position=batch['indices'].index(index)
  arrays=[];poses=[];negs=[]
  for i in batch['indices']:
   row=records[i];x,y=full.coordinates(i,13,row,meta);a=rgb(ROOT/row['input'])[y:y+256,x:x+256];t=full.gray(ROOT/row['labels']['target'])[y:y+256,x:x+256];v=full.gray(ROOT/row['labels']['valid'])[y:y+256,x:x+256]
   arrays.append((a,t,v));active=(i==index)
   poses.append(regions[patch]['core'][y:y+256,x:x+256] if active else np.zeros((256,256),bool));negs.append(regions[patch]['paper'][y:y+256,x:x+256] if active else np.zeros((256,256),bool))
  X=torch.stack([torch.from_numpy(auxiliaries(a).copy()).permute(2,0,1).float()/255 for a,t,v in arrays]).to(device)
  T=torch.stack([torch.from_numpy(t.copy()).unsqueeze(0).float()/255 for a,t,v in arrays]).to(device);V=torch.stack([torch.from_numpy(v.copy()).unsqueeze(0).float()/255 for a,t,v in arrays]).to(device)
  pos=torch.from_numpy(np.stack(poses)).unsqueeze(1).to(device);neg=torch.from_numpy(np.stack(negs)).unsqueeze(1).to(device)
  p=predict(G,'ragan',X);raw=capture['raw'];z=product_logit(raw);sup=stable_loss(p,T,V,raw)
  ink,paper=local_terms(z,pos,neg);perpixel=F.binary_cross_entropy_with_logits(z,T,reduction='none');n=V.sum();fine=(perpixel*pos).sum()/n
  parameters=[q for q in G.parameters() if q.requires_grad]
  gf=torch.autograd.grad(fine,parameters,retain_graph=True,allow_unused=True)
  gg=torch.autograd.grad(sup,parameters,retain_graph=True,allow_unused=True)
  gi=torch.autograd.grad(ink,parameters,retain_graph=True,allow_unused=True)
  gp=torch.autograd.grad(paper,parameters,allow_unused=True)
  nf,ng=norm(gf),norm(gg);rest=[None if g is None else g-(f if f is not None else 0) for f,g in zip(gf,gg)];nr=norm(rest)
  result.append(dict(order=order,checkpoint_sha256=sha(ck),target_patch=patch,batch_indices=batch['indices'],position=position,
   valid_pixels=int(n),core_pixels=int(pos.sum()),paper_proxy_pixels=int(neg.sum()),global_loss=float(sup.detach()),
   target_core_BCE_contribution=float(fine.detach()),target_core_BCE_mean=float(ink.detach()),local_paper_BCE_mean=float(paper.detach()),
   core_BCE_gradient_norm=nf,total_gradient_norm=ng,core_to_total_gradient_ratio=nf/max(ng,1e-30),
   core_vs_rest_gradient_cosine=dot(gf,rest)/max(nf*nr,1e-30),
   total_GD_step_increases_core_BCE=dot(gf,gg)<0,
   region_normalization_multiplier=float(n/pos.sum()),
   normalized_core_gradient_norm=norm(gi),normalized_paper_gradient_norm=norm(gp),
   normalized_ink_vs_paper_cosine=dot(gi,gp)/max(norm(gi)*norm(gp),1e-30),
   interpretation='local gradients of a frozen model; plain GD directional test, not Adam training result'))
  x,y=full.coordinates(index,13,records[index],meta);source=regions[patch]['source'][y:y+256,x:x+256];a,t,v=arrays[position]
  teacher=t.copy();teacher[v==0]=128;overlay=source.copy();overlay[poses[position]]=[235,45,40];overlay[negs[position]]=[40,135,235]
  trace=reviews[patch][1][y:y+256,x:x+256];trace_overlay=source.copy();trace_overlay[trace]=[235,45,40]
  path=dest/'figures'/f'remaining44_order{order+1}.png';real.panel(path,patch+' / 原本・損傷・教師・補助領域', [('原本',source),('強退色 .80',a),('現行教師 / 不明は灰色',teacher),('墨芯赤・原本明紙proxy青',overlay),('追跡44点 / 赤',trace_overlay),('現行推論',np.where(p[position,0].detach().cpu().numpy()<.5,0,255).astype(np.uint8))]);figures.append(str(path.relative_to(OUT)))
  del G,payload,gf,gg,gi,gp,rest;torch.cuda.empty_cache()
 dump(dest/'gradient_audit.json',dict(rows=result,figures=figures,no_updates=True,no_label_edits=True,formal_accuracy=False))
 shutil.copy2(__file__,dest/Path(__file__).name)
 print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
