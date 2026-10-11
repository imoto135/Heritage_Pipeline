"""Compact synthetic-layer classifiers with matched architecture ablations.

Checkpoints are experimental, not restoration or real-document annotations.
"""
from pathlib import Path
import argparse
import json
import time
import os
import random
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from synthetic_layers import OUT, LayerDataset, input_features, file_hash

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.body=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
    def forward(self,x):return self.body(x)

class Selector(nn.Module):
    def __init__(self,channels=9,pointwise=False,width=16):
        super().__init__();self.pointwise=pointwise
        if pointwise:
            self.pixel=nn.Sequential(nn.Conv2d(channels,64,1),nn.SiLU(),nn.Conv2d(64,64,1),nn.SiLU(),nn.Conv2d(64,2,1))
        else:
            self.e1=Block(channels,width);self.e2=Block(width,width*2);self.e3=Block(width*2,width*4);self.center=Block(width*4,width*6)
            self.d3=Block(width*10,width*4);self.d2=Block(width*6,width*2);self.d1=Block(width*3,width);self.head=nn.Conv2d(width,2,1)
    def forward(self,x):
        if self.pointwise:return self.pixel(x)
        a=self.e1(x);b=self.e2(nn.functional.avg_pool2d(a,2));c=self.e3(nn.functional.avg_pool2d(b,2));z=self.center(nn.functional.avg_pool2d(c,2))
        def up(z,t):return nn.functional.interpolate(z,size=t.shape[-2:],mode='bilinear',align_corners=False)
        z=self.d3(torch.cat([up(z,c),c],1));z=self.d2(torch.cat([up(z,b),b],1));z=self.d1(torch.cat([up(z,a),a],1))
        return self.head(z)

CONFIGS={
    'rgb_separated':dict(mode='rgb',training='separated',paired=False,consistency=0.,pointwise=False),
    'rgb_overlap':dict(mode='rgb',training='overlap',paired=False,consistency=0.,pointwise=False),
    'relative_overlap':dict(mode='relative',training='overlap',paired=False,consistency=0.,pointwise=False),
    'relative_pair_aug':dict(mode='relative',training='overlap',paired=True,consistency=0.,pointwise=False),
    'relative_pair_consistent':dict(mode='relative',training='overlap',paired=True,consistency=1.,pointwise=False),
    'relative_pointwise':dict(mode='relative',training='overlap',paired=False,consistency=0.,pointwise=True),
}

def balanced_loss(logits,target):
    truth=torch.clamp(target/.3,0,1)
    weights=1+4*truth[:,0:1]
    mark=nn.functional.binary_cross_entropy_with_logits(logits[:,0:1],truth[:,0:1],reduction='none')
    stain=nn.functional.binary_cross_entropy_with_logits(logits[:,1:2],truth[:,1:2])
    return (mark*weights).mean()+stain

@torch.no_grad()
def validate(model,loader,device):
    model.eval();totals=np.zeros((2,4),np.float64);losses=[]
    for x,y in loader:
        x=x.to(device);y=y.to(device);z=model(x)
        losses.append(float(balanced_loss(z,y)))
        pred=z.sigmoid()>=.5;truth=y>=.15
        for c in range(2):
            p=pred[:,c];t=truth[:,c]
            totals[c]+=[int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum()),int((~p&~t).sum())]
    model.train()
    metrics=[]
    for c,(tp,fp,fn,tn) in enumerate(totals):
        metrics.append(dict(layer=['mark','stain'][c],TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
            precision=float(tp/max(tp+fp,1)),recall=float(tp/max(tp+fn,1)),IoU=float(tp/max(tp+fp+fn,1)),
            F1=float(2*tp/max(2*tp+fp+fn,1))))
    return dict(loss=float(np.mean(losses)),metrics=metrics)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',choices=list(CONFIGS),required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=413);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--batch',type=int,default=8);parser.add_argument('--validate-every',type=int,default=1000)
    a=parser.parse_args();cfg=CONFIGS[a.name]
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=True
    dest=OUT/'learned_trials'/f'{a.name}_seed{a.seed}';dest.mkdir(parents=True,exist_ok=True)
    protocol=dict(name=a.name,config=cfg,steps=a.steps,batch=a.batch,optimizer='AdamW lr2e-4 weight_decay1e-4',
        seed=a.seed,synthetic_seed=20261008,width=16,
        real_training='None: no labels from real red or brown masks',
        synthetic_training='Two known, potentially overlapping digital layers',
        training_source_sha256=file_hash(__file__),synthesis_source_sha256=file_hash(Path(__file__).with_name('synthetic_layers.py')),
        effective_images_per_step=a.batch*(2 if cfg['paired'] else 1),
        compute_matching='Paired aug and paired consistency have identical geometry/colors/BCE exposures; other ablations have different exposure counts',
        checkpoint_selection='Highest harmonic mean of synthetic dev mark/stain F1 only; no real validation inspection',
        start_utc=datetime.now(timezone.utc).isoformat())
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    train=LayerDataset('train',length=a.steps*a.batch,seed=20261008,mode=cfg['mode'],training=cfg['training'],paired=cfg['paired'])
    dev=LayerDataset('dev',length=256,seed=20261009,mode=cfg['mode'],training='overlap')
    loader=DataLoader(train,batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=True)
    devloader=DataLoader(dev,batch_size=a.batch,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    model=Selector(3 if cfg['mode']=='rgb' else 9,cfg['pointwise']).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler();best=-1.;start=time.monotonic();ema=None
    for step,batch in enumerate(loader,1):
        if cfg['paired']:
            x,other,y=batch;x=torch.cat([x,other],0).to(a.device,non_blocking=True);y=y.to(a.device,non_blocking=True)
        else:x,y=batch;x=x.to(a.device,non_blocking=True);y=y.to(a.device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            z=model(x)
            if cfg['paired']:
                z1,z2=z.chunk(2,0);supervised=.5*(balanced_loss(z1,y)+balanced_loss(z2,y))
                consistency=(z1.sigmoid()-z2.sigmoid()).square().mean();loss=supervised+cfg['consistency']*consistency
            else:loss=balanced_loss(z,y)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
        scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5)
        scaler.step(opt);scaler.update()
        ema=float(loss) if ema is None else ema*.97+float(loss)*.03
        if step%100==0:
            progress=dict(step=step,total=a.steps,loss_ema=ema,elapsed_seconds=time.monotonic()-start,
                estimated_remaining_seconds=(time.monotonic()-start)/step*(a.steps-step),status='running')
            (dest/'progress.json').write_text(json.dumps(progress,indent=2))
            if step%500==0:print(a.name,step,round(ema,4),round(progress['elapsed_seconds'],1),'sec',flush=True)
        if step%a.validate_every==0 or step==a.steps:
            metrics=validate(model,devloader,a.device);f=[m['F1'] for m in metrics['metrics']]
            score=2*f[0]*f[1]/max(sum(f),1e-8)
            record=dict(step=step,dev=metrics,selection_score=score,elapsed_seconds=time.monotonic()-start)
            with (dest/'history.jsonl').open('a') as ff:ff.write(json.dumps(record)+'\n')
            print('dev',a.name,step,[(m['layer'],round(m['F1'],3)) for m in metrics['metrics']],flush=True)
            checkpoint=dict(model=model.state_dict(),config=cfg,protocol=protocol,step=step,dev=metrics)
            torch.save(checkpoint,dest/'last.pth')
            if score>best:best=score;torch.save(checkpoint,dest/'best.pth')
    (dest/'complete.json').write_text(json.dumps(dict(status='completed',steps=a.steps,elapsed_seconds=time.monotonic()-start,
        best_dev_selection_score=best,best_sha256=file_hash(dest/'best.pth'),last_sha256=file_hash(dest/'last.pth')),
        indent=2))
    print('Complete',a.name,flush=True)

if __name__=='__main__':main()
