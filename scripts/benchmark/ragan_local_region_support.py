"""Source-reviewed positive cores and source-paper proxy for auxiliary BCE.

No changes to target values, validity masks, RGB, or public generator forward.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run_ragan_full_continuation import ROOT,gray
from run_v5_exposure_loss_control import core_masks
from evaluate_ridge_gate_independent_val import proxies
from compare_v5_aligned_fade_control import rgb

PADDING=16

def region_masks(records):
 result={}
 for patch,path in core_masks().items():
  row=next(r for r in records if r['patch_id']==patch)
  core=gray(path)>0;valid=gray(ROOT/row['labels']['valid'])>0;target=gray(ROOT/row['labels']['target'])<128
  assert np.array_equal(core,core&valid&target)
  yy,xx=np.where(core);box=[max(0,int(xx.min())-PADDING),max(0,int(yy.min())-PADDING),min(512,int(xx.max())+PADDING+1),min(512,int(yy.max())+PADDING+1)]
  x0,y0,x1,y1=box;roi=np.zeros(core.shape,bool);roi[y0:y1,x0:x1]=True
  source=rgb(ROOT/row['clean']);paper,_,remote,_=proxies(source)
  known_paper=roi&valid&~target
  contrast=np.max(np.abs(source.astype(np.int16)-paper.astype(np.int16)),axis=2)
  result[patch]=dict(core=core,paper=known_paper&remote,known_paper=known_paper,roi=roi,box=box,
   contrast=contrast,core_path=str(path.relative_to(ROOT)),source=source)
 return result

def local_terms(logits,positive,negative):
 """Average each reviewed example independently; missing examples contribute zero."""
 bce_ink=F.softplus(logits);bce_paper=F.softplus(-logits)
 ink_terms=[];paper_terms=[]
 for i in range(logits.shape[0]):
  pos=positive[i].bool();neg=negative[i].bool()
  if pos.any():
   ink_terms.append(bce_ink[i][pos].mean())
   if neg.any():paper_terms.append(bce_paper[i][neg].mean())
 zero=logits.sum()*0
 return (torch.stack(ink_terms).mean() if ink_terms else zero,
         torch.stack(paper_terms).mean() if paper_terms else zero)
