"""Development-only partial-label adaptation, with matched exposure controls.

Hypothesis: dense paper absence and smooth brown perturbations reduce source-paper
overfitting without forcing unlabeled original ink to zero. Not a novelty claim.
"""
import argparse,json,random,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from synthetic_layers import OUT,ROOT,input_features,file_hash,LayerDataset
from train_texture_transfer import PairedDomain
from train_layer_selector import Selector,balanced_loss,validate

class NativePartial(Dataset):
    def __init__(self,length,dense=False,brown=False,exclude=()):
        self.length=length;self.dense=dense;self.brown=brown
        self.points=[r for r in json.loads((OUT/'sparse_original_review/points_frozen.json').read_text())['points'] if r['role']=='development' and r['book'] not in exclude]
        self.regions=[r for r in json.loads((OUT/'original_paper_negative_regions/regions_frozen.json').read_text())['regions'] if r['book'] not in exclude]
        cases=json.loads((OUT/'cases_frozen.json').read_text())['cases'];paths={r['page']:ROOT/r['original'] for r in cases}
        self.images={p:np.asarray(Image.open(paths[p]).convert('RGB')) for p in {r['page'] for r in self.points}}
    def __len__(self):return self.length
    def __getitem__(self,i):
        rng=np.random.default_rng(20261018+i*937)
        # Same crop anchor sequence for all ablations, including region anchors.
        rows=self.points+self.regions;r=rows[int(rng.integers(len(rows)))];rgb=self.images[r['page']];h,w=rgb.shape[:2]
        if 'xy_original' in r:px,py=r['xy_original']
        else:
            ax,ay,aw,ah=r['xywh_original'];px=int(rng.integers(ax,ax+aw));py=int(rng.integers(ay,ay+ah))
        x=max(0,min(w-256,px-int(rng.integers(70,186))));y=max(0,min(h-256,py-int(rng.integers(70,186))))
        crop=rgb[y:y+256,x:x+256].copy();target=np.zeros((256,256),np.float32);weight=target.copy()
        if self.dense:
            for q in self.regions:
                if q['page']!=r['page']:continue
                ax,ay,aw,ah=q['xywh_original'];x0=max(0,ax-x);x1=min(256,ax+aw-x);y0=max(0,ay-y);y1=min(256,ay+ah-y)
                if x1>x0 and y1>y0:weight[y0:y1,x0:x1]=1
        for q in self.points:
            if q['page']!=r['page']:continue
            xx,yy=q['xy_original'];xx-=x;yy-=y
            if 1<=xx<255 and 1<=yy<255:
                target[yy-1:yy+2,xx-1:xx+2]=float(q['label']=='observed_red_mark');weight[yy-1:yy+2,xx-1:xx+2]=1
        if self.brown and rng.random()<.75:
            yy,xx=np.mgrid[:256,:256].astype(np.float32);field=np.zeros((256,256),np.float32)
            for _ in range(int(rng.integers(1,4))):
                cx,cy=rng.uniform(-80,336,2);sx,sy=rng.uniform(30,210,2)
                field=np.maximum(field,np.exp(-.5*((xx-cx)/sx)**2-.5*((yy-cy)/sy)**2))
            direction=np.array([rng.uniform(.08,.55),rng.uniform(.55,.85),1.],np.float32)
            crop=np.rint(crop.astype(np.float32)*np.exp(-field[:,:,None]*rng.uniform(.03,.35)*direction)).astype(np.uint8)
        return torch.from_numpy(input_features(crop)),torch.from_numpy(target),torch.from_numpy(weight)

def partial_loss(logits,target,weight):
    loss=nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none');terms=[]
    # Prevent dense negative rectangles from overwhelming the few positive points.
    for positive in [False,True]:
        support=weight*(target>.5 if positive else target<=.5)
        if support.sum()>0:terms.append((loss*support).sum()/support.sum())
    return torch.stack(terms).mean() if terms else logits.sum()*0

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['sparse_control','dense_native','dense_native_brown'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=8000);p.add_argument('--exclude-dev-book');a=p.parse_args()
    seed=715;cv2.setNumThreads(1);torch.set_num_threads(4);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);torch.backends.cudnn.benchmark=True
    exclude=[a.exclude_dev_book] if a.exclude_dev_book else [];suffix='_exclude'+a.exclude_dev_book if a.exclude_dev_book else ''
    dest=OUT/'native_partial_trials'/f'{a.name}_seed{seed}{suffix}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists(),'Already complete; do not overwrite'
    base=OUT/'learned_trials/relative_overlap_seed413/last.pth';ck=torch.load(base,map_location='cpu');assert ck['step']==12000
    native=NativePartial(a.steps*4,a.name!='sparse_control',a.name=='dense_native_brown',exclude)
    domain=PairedDomain(a.steps*4,True,exclude)
    protocol=dict(name=a.name,steps=a.steps,seed=seed,start_utc=datetime.now(timezone.utc).isoformat(),
        initial_checkpoint=str(base),initial_checkpoint_sha256=file_hash(base),source_sha256=file_hash(__file__),
        actual_training_books=sorted({r['book'] for r in native.points}),excluded_development_books=exclude,
        point_count=len(native.points),paper_regions=len(native.regions),dense_negative=a.name!='sparse_control',brown_augmentation=a.name=='dense_native_brown',
        selection='Fixed LAST8000 and .5 threshold; developmental comparison, no local validation or external inspection',
        exposure='All conditions: 4 ordinary synthetic +4 real texture synthetic +4 native original crops per step; identical anchors and seed',
        losses='Synthetic both heads, real texture synthetic mark head, native partial mark loss averaged over positive and negative classes with coefficient .5',
        labels='Tentative original-only points and no-visible-red rectangles, not expert or chemical GT; unlabeled pixels ignored',
        perturbation='Known multiplicative gamma-RGB smooth brown fields; approximate augmentation, not physical stain model',
        points_sha256=file_hash(OUT/'sparse_original_review/points_frozen.json'),regions_sha256=file_hash(OUT/'original_paper_negative_regions/regions_frozen.json'),
        novelty='Partial labels, synthetic textures and color augmentation are prior techniques; performance hypothesis only')
    assert not set(exclude)&set(protocol['actual_training_books'])
    (dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    source=dest/'source';source.mkdir(exist_ok=True)
    for f in ['train_native_partial.py','train_texture_transfer.py','paper_texture_layers.py','synthetic_layers.py','train_layer_selector.py']:shutil.copy2(OUT/f,source/f)
    normal_loader=DataLoader(domain,batch_size=4,num_workers=4,pin_memory=True,persistent_workers=True)
    native_iter=iter(DataLoader(native,batch_size=4,num_workers=2,pin_memory=True,persistent_workers=True))
    dev=DataLoader(LayerDataset('dev',256,mode='relative',training='overlap',seed=20261009),batch_size=8,num_workers=2,pin_memory=True)
    model=Selector(9).to(a.device);model.load_state_dict(ck['model']);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler();start=time.monotonic();ema=None
    for step,(x,y,d,t) in enumerate(normal_loader,1):
        nx,ny,nw=next(native_iter);x=x.to(a.device);y=y.to(a.device);d=d.to(a.device);t=t.to(a.device);nx=nx.to(a.device);ny=ny.to(a.device);nw=nw.to(a.device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            z=model(torch.cat([x,d,nx]));zs,zd,zn=z.chunk(3);truth=(t[:,0:1]/.3).clamp(0,1)
            dl=(nn.functional.binary_cross_entropy_with_logits(zd[:,0:1],truth,reduction='none')*(1+4*truth)).mean()
            loss=balanced_loss(zs,y)+dl+.5*partial_loss(zn[:,0],ny,nw)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update()
        ema=float(loss) if ema is None else .97*ema+.03*float(loss)
        if step%100==0:
            rec=dict(step=step,total=a.steps,loss_ema=ema,elapsed_seconds=time.monotonic()-start)
            (dest/'progress.json').write_text(json.dumps(rec,indent=2))
            if step%500==0:print(a.name,suffix,step,round(ema,4),round(rec['elapsed_seconds'],1),flush=True)
        if step%2000==0 or step==a.steps:
            scores=validate(model,dev,a.device);torch.save(dict(model=model.state_dict(),config=ck['config'],protocol=protocol,step=step,dev=scores),dest/'last.pth')
            with (dest/'history.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,synthetic_dev=scores))+'\n')
    (dest/'complete.json').write_text(json.dumps(dict(status='completed',steps=a.steps,seconds=time.monotonic()-start,last_sha256=file_hash(dest/'last.pth')),indent=2));print('Complete',a.name,suffix,flush=True)

if __name__=='__main__':main()
