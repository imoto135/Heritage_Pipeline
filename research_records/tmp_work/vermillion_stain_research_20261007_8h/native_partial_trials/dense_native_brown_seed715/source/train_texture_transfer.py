"""Controlled continued training; real paper supervises mark absence only.

All choices fixed before local validation or external selector inspection.
"""
import argparse
import json
import random
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from synthetic_layers import OUT,ROOT,LayerDataset,input_features,file_hash
from paper_texture_layers import PaperTextureLayers
from train_layer_selector import Selector,balanced_loss,validate

VARIANTS=['continue_control','paper_texture','paper_texture_sparse']

class PairedDomain(Dataset):
    def __init__(self,length,texture,excluded_books=()):
        self.normal=LayerDataset('train',length,mode='relative',training='overlap',seed=20261014)
        self.domain=PaperTextureLayers('train',length,mode='relative',training='overlap',seed=20261015,texture=texture,excluded_books=excluded_books)
    def __len__(self):return len(self.normal)
    def __getitem__(self,i):
        x,y=self.normal[i];a,b=self.domain[i];return x,y,a,b

class SparseCrops(Dataset):
    def __init__(self,length,excluded_books=()):
        self.length=length
        cases=json.loads((OUT/'cases_frozen.json').read_text())['cases']
        originals={r['page']:ROOT/r['original'] for r in cases}
        self.points=[r for r in json.loads((OUT/'sparse_original_review/points_frozen.json').read_text())['points'] if r['role']=='development' and r['book'] not in excluded_books]
        assert self.points,'No training points after exclusion'
        self.images={p:np.asarray(Image.open(originals[p]).convert('RGB')) for p in {r['page'] for r in self.points}}
    def __len__(self):return self.length
    def __getitem__(self,i):
        rng=np.random.default_rng(20261016+i*937);r=self.points[int(rng.integers(len(self.points)))];rgb=self.images[r['page']]
        px,py=r['xy_original'];h,w=rgb.shape[:2]
        x=max(0,min(w-256,px-int(rng.integers(70,186))));y=max(0,min(h-256,py-int(rng.integers(70,186))))
        crop=rgb[y:y+256,x:x+256].copy();target=np.zeros((256,256),np.float32);weight=target.copy()
        for q in self.points:
            if q['page']!=r['page']:continue
            xx,yy=q['xy_original'];xx-=x;yy-=y
            if 1<=xx<255 and 1<=yy<255:
                target[yy-1:yy+2,xx-1:xx+2]=float(q['label']=='observed_red_mark');weight[yy-1:yy+2,xx-1:xx+2]=1
        return torch.from_numpy(input_features(crop)),torch.from_numpy(target),torch.from_numpy(weight)

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=VARIANTS,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--seed',type=int,default=614);p.add_argument('--exclude-dev-book',default=None);a=p.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    torch.backends.cudnn.benchmark=True
    base=OUT/'learned_trials/relative_overlap_seed413/last.pth';ck=torch.load(base,map_location='cpu')
    assert ck['step']==12000
    excluded=[a.exclude_dev_book] if a.exclude_dev_book else []
    suffix='_exclude'+a.exclude_dev_book if a.exclude_dev_book else ''
    dest=OUT/'texture_transfer_trials'/f'{a.name}_seed{a.seed}{suffix}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(),'Already complete'
    protocol=dict(name=a.name,steps=a.steps,seed=a.seed,initial_checkpoint=str(base),initial_checkpoint_sha256=file_hash(base),
        start_utc=datetime.now(timezone.utc).isoformat(),optimizer='AdamW lr5e-5 weight_decay1e-4',
        normal_branch='4 synthetic samples per step; both heads known digital support',
        domain_branch='4 samples per step; mark head only; same digitally added geometry sequence',
        texture=a.name!='continue_control',sparse=a.name=='paper_texture_sparse',
        domain_stain_head_supervised=False,
        sparse_details='Optional additional 4 native crops with only 3x3 observed-point mark support. Tentative 92 dev points from three books. Extra exposure is not compute matched.',
        paper_regions_sha256=file_hash(OUT/'original_paper_negative_regions/regions_frozen.json'),
        sparse_points_sha256=file_hash(OUT/'sparse_original_review/points_frozen.json'),
        no_validation_or_external_used=True,real_paper_is_not_clean_GT=True,
        primary_selection='Fixed final step, threshold .5; no real-image checkpoint selection',
        historical_ink_readability_GT=False,source_sha256=file_hash(__file__),
        methods_fixed_before_real_validation=True,excluded_development_books=excluded,
        exclusion='No excluded-book paper textures or point crops enter fitting; synthetic base excludes all original development books',
        generalization_scope='Local three-book leave-one-book-out diagnosis, not independent final validation' if excluded else 'All three development books used for texture/sparse supervision')
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    source=dest/'source';source.mkdir(exist_ok=True)
    for name in ['train_texture_transfer.py','paper_texture_layers.py','synthetic_layers.py','train_layer_selector.py']:
        shutil.copy2(OUT/name,source/name)
    ds=PairedDomain(a.steps*4,a.name!='continue_control',excluded);loader=DataLoader(ds,batch_size=4,num_workers=4,pin_memory=True,persistent_workers=True)
    protocol['actual_texture_training_books']=sorted({r['book'] for r in ds.domain.texture_rows})
    assert not set(excluded)&set(protocol['actual_texture_training_books'])
    dev=DataLoader(LayerDataset('dev',256,mode='relative',training='overlap',seed=20261009),batch_size=8,num_workers=2,pin_memory=True)
    sparse_iter=None
    if a.name=='paper_texture_sparse':
        sparse_ds=SparseCrops(a.steps*4,excluded)
        protocol['actual_point_training_books']=sorted({r['book'] for r in sparse_ds.points})
        protocol['actual_point_training_count']=len(sparse_ds.points)
        assert not set(excluded)&set(protocol['actual_point_training_books'])
        sparse_iter=iter(DataLoader(sparse_ds,batch_size=4,num_workers=2,pin_memory=True,persistent_workers=True))
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    model=Selector(9).to(a.device);model.load_state_dict(ck['model']);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler();start=time.monotonic();ema=None
    for step,(x,y,d,t) in enumerate(loader,1):
        x=x.to(a.device);y=y.to(a.device);d=d.to(a.device);t=t.to(a.device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            z=model(torch.cat([x,d]));zn,zd=z.chunk(2)
            truth=(t[:,0:1]/.3).clamp(0,1)
            domain_loss=(nn.functional.binary_cross_entropy_with_logits(zd[:,0:1],truth,reduction='none')*(1+4*truth)).mean()
            loss=balanced_loss(zn,y)+domain_loss
            if sparse_iter is not None:
                sx,sy,sw=next(sparse_iter);sx=sx.to(a.device);sy=sy.to(a.device);sw=sw.to(a.device)
                sl=model(sx)[:,0];sparse_loss=(nn.functional.binary_cross_entropy_with_logits(sl,sy,reduction='none')*sw).sum()/sw.sum().clamp(min=1)
                loss=loss+.5*sparse_loss
        assert torch.isfinite(loss),'Nonfinite loss'
        scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
        ema=float(loss) if ema is None else .97*ema+.03*float(loss)
        if step%100==0:
            progress=dict(step=step,total=a.steps,loss_ema=ema,elapsed_seconds=time.monotonic()-start)
            (dest/'progress.json').write_text(json.dumps(progress,indent=2))
            if step%500==0:print(a.name,step,round(ema,4),round(progress['elapsed_seconds'],1),'sec',flush=True)
        if step%2000==0 or step==a.steps:
            scores=validate(model,dev,a.device)
            with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,synthetic_dev=scores))+'\n')
            torch.save(dict(model=model.state_dict(),config=ck['config'],protocol=protocol,step=step,dev=scores),dest/'last.pth')
    (dest/'complete.json').write_text(json.dumps(dict(status='completed',steps=a.steps,elapsed_seconds=time.monotonic()-start,last_sha256=file_hash(dest/'last.pth')),indent=2))
    print('Complete',a.name,flush=True)

if __name__=='__main__':main()
