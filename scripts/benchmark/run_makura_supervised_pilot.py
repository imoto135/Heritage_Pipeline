"""Prepare masked provisional labels and run a small, explicitly non-GT pilot.

No original candidate, source page or frozen synthetic input is overwritten.
The pilot checks learning/implementation and creates images, not formal ranks.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from ragan_pilot_support import (ROOT, make_model, predict, reference_classes,
    stabilize_ggc, auxiliaries, supervised_loss, critic_input)

SOURCE = ROOT / 'tmp_work/makura_supervised_pilot_20261021'
OUT = ROOT / 'tmp_work/makura_supervised_smoke_20261021'
SEED = 20261021
AMP = False  # FP32 pilot: avoid scaled FP16 overflow before the first update.
SPLITS = ('train', 'val', 'pilot_eval')
# Source-only native-pixel inspection. These rectangles are exclusions, not
# inferred reconstructions or new dense ground truth.
EXCLUSIONS = {
    '200017197_00341_1_p00': [
        {'xywh':[265,480,45,32], 'reason':'isolated dark spot; uncertain provenance'},
        {'xywh':[25,413,27,33], 'reason':'isolated colored mark below left stroke'}],
    '200017197_00048_0_p00': [
        {'xywh':[35,187,55,36], 'reason':'isolated light brown mark'}],
    '200017197_00172_0_p00': [
        {'xywh':[402,236,58,28], 'reason':'colored mark to right of black text'}],
}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(path.relative_to(ROOT))


def rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def gray(path):
    return np.array(Image.open(path).convert('L'))


def rows(split):
    return [json.loads(x) for x in (OUT/(split+'_manifest.jsonl')).read_text().splitlines()]


def prepare():
    assert not (OUT/'label_protocol.json').exists(), 'Do not overwrite a frozen label version.'
    verification = json.loads((SOURCE/'verification.json').read_text())
    assert verification['pairs'] == 1440
    old_index = json.loads((SOURCE/'patch_index.json').read_text())
    index = {}
    stats = []
    for patch in old_index:
        folder = ROOT/patch['folder']
        target = gray(folder/'target_candidate.png')
        valid = gray(folder/'agreement_valid.png') > 0
        old_target_hash = sha(folder/'target_candidate.png')
        old_valid_hash = sha(folder/'agreement_valid.png')
        target = target.copy(); valid = valid.copy()
        before = int(valid.sum())
        exclusions = EXCLUSIONS.get(patch['patch_id'], [])
        for region in exclusions:
            x,y,w,h = region['xywh']
            valid[y:y+h,x:x+w] = False
            # White placeholder in unknown pixels. It is never used as truth.
            target[y:y+h,x:x+w] = 255
        dest = OUT/'labels_v2'/patch['patch_id']; dest.mkdir(parents=True)
        Image.fromarray(target).save(dest/'paper_target.png')
        Image.fromarray(valid.astype(np.uint8)*255).save(dest/'valid.png')
        info = dict(patch_id=patch['patch_id'], target=rel(dest/'paper_target.png'),
            valid=rel(dest/'valid.png'), target_sha256=sha(dest/'paper_target.png'),
            valid_sha256=sha(dest/'valid.png'), source_candidate_sha256=old_target_hash,
            source_agreement_sha256=old_valid_hash, valid_pixels=int(valid.sum()),
            valid_ink=int(((target==0)&valid).sum()), additionally_excluded=before-int(valid.sum()),
            source_review='selected regions only; dense labels remain provisional',
            exclusions=exclusions, formal_gt=False)
        index[patch['patch_id']] = info
        stats.append(info)
        assert sha(folder/'target_candidate.png') == old_target_hash
        assert sha(folder/'agreement_valid.png') == old_valid_hash
    for split in SPLITS:
        old = [json.loads(x) for x in (SOURCE/(split+'_manifest.jsonl')).read_text().splitlines()]
        new = []
        for row in old:
            assert row['review_status']=='pending' and row['training_eligible'] is False
            assert sha(ROOT/row['input']) == row['input_sha256']
            new.append(dict(id=row['id'], patch_id=row['patch_id'], page=row['page'],
                group=row['group'], split=split, input=row['input'], input_sha256=row['input_sha256'],
                clean=row['clean'], alpha=row['alpha'], damage_family=row['damage_family'],
                damage_strength=row['damage_strength'], labels=index[row['patch_id']],
                use='provisional_masked_training_or_consistency_only', formal_gt=False))
        (OUT/(split+'_manifest.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new))
    dump(OUT/'label_protocol.json', dict(seed=SEED, source=rel(SOURCE),
        purpose='Provisional masked-label engineering pilot, not a supervised accuracy benchmark',
        target_source='Undamaged original page; frozen before artificial damage',
        target='Sauvola candidate, never computed from artificially damaged input',
        validity='Three-method agreement excluding 1px disagreement neighborhood, minus source-only exclusions',
        revisions=EXCLUSIONS, expert_reviewed=False, original_artifacts_unchanged=True,
        dense_labels_pending=True, evaluation_is_formal_accuracy=False,
        patch_count=len(stats), additional_excluded=sum(p['additionally_excluded'] for p in stats)))
    dump(OUT/'labels_index.json', stats)
    sources=[Path(__file__), Path(__file__).with_name('ragan_pilot_support.py')]
    for source in sources:
        dest=OUT/'code_snapshot'/source.name; dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest)
    dump(OUT/'code_hashes.json',{rel(p):sha(p) for p in sources})
    print('Prepared masked provisional labels:', len(stats), 'patches', flush=True)


def audit():
    torch.set_num_threads(2)
    _, GGC, _ = reference_classes()
    torch.manual_seed(SEED)
    probes=[]
    for stride in (1,2):
        for bias in (False, True):
            old=GGC(3,4,kernel_size=3,stride=stride,padding=1,bias=bias)
            new=copy.deepcopy(old); stabilize_ggc(new)
            a=torch.randn(2,3,24,24,requires_grad=True)
            b=a.detach().clone().requires_grad_(True)
            expected=old(a); actual=new(b)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
            expected.square().mean().backward(); actual.square().mean().backward()
            torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-7)
            old_params=dict(old.named_parameters())
            for name,p in new.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p.grad,old_params[name].grad,rtol=1e-5,atol=1e-7)
            before={n:id(p) for n,p in new.named_parameters()}
            optimizer=torch.optim.Adam([p for p in new.parameters() if p.requires_grad], lr=3e-5)
            initial=new.conv.weight.detach().clone()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                (new(b).square().mean()+new(b+0.1).abs().mean()).backward()
                optimizer.step()
            assert before=={n:id(p) for n,p in new.named_parameters()}
            ids={id(p) for group in optimizer.param_groups for p in group['params']}
            assert all(id(p) in ids for p in new.parameters() if p.requires_grad)
            assert not torch.equal(initial,new.conv.weight)
            restored=GGC(3,4,kernel_size=3,stride=stride,padding=1,bias=bias)
            stabilize_ggc(restored); restored.load_state_dict(new.state_dict())
            torch.testing.assert_close(new(b),restored(b),rtol=0,atol=0)
            probes.append(dict(stride=stride,bias=bias,forward_matches=True,
                active_gradients_match=True,parameters_stable=True,multi_forward_backward=True,
                optimizer_updates=True,checkpoint_roundtrip=True))
    # Unknown pixels must influence neither supervised nor adversarial labels.
    p=torch.full((1,1,8,8),.6,requires_grad=True)
    target=torch.ones_like(p); valid=torch.ones_like(p); valid[:,:,2:5,2:5]=0
    alternate=target.clone(); alternate[:,:,2:5,2:5]=0
    torch.testing.assert_close(supervised_loss(p,target,valid),supervised_loss(p,alternate,valid),rtol=0,atol=0)
    loss=supervised_loss(p,target,valid)+critic_input(p,valid).sum()
    loss.backward(); assert (p.grad[valid==0]==0).all()
    # Dataset pairing, auxiliary channels and deterministic native cropping.
    dataset=PilotDataset('train', crop_size=256)
    one=dataset[0]; two=dataset[0]
    assert all(torch.equal(x,y) for x,y in zip(one,two))
    assert one[0].shape==(5,256,256) and one[1].shape==(1,256,256)
    dataset.epoch=1; three=dataset[0]
    assert not torch.equal(one[0],three[0])
    groups={split:{r['group'] for r in rows(split)} for split in SPLITS}
    for split in SPLITS:
        for other in SPLITS:
            if split!=other: assert not groups[split]&groups[other]
    for split in SPLITS:
        for row in rows(split):
            labels=row['labels']
            assert sha(ROOT/labels['target'])==labels['target_sha256']
            assert sha(ROOT/labels['valid'])==labels['valid_sha256']
    dump(OUT/'adapter_verification.json',dict(ggc=probes,unknown_target_invariance=True,
        unknown_gradient_zero=True,paired_native_crop=True,deterministic_crop=True,
        group_splits_disjoint=True,label_hashes_verified=True,
        auxiliary_source='Only damaged RGB after native crop; no target/clean input',
        frozen_unused_weight='diff_conv.weight; kept only for state_dict compatibility',
        retained_semantics='Differential kernel detached from normal convolution weight'))
    print('Adapter and partial-label checks passed',flush=True)


class PilotDataset(Dataset):
    def __init__(self, split, crop_size=256):
        self.records=rows(split); self.split=split; self.crop_size=crop_size; self.epoch=0
        self.labels={}
        for row in self.records:
            if row['patch_id'] not in self.labels:
                labels=row['labels']
                self.labels[row['patch_id']]=(gray(ROOT/labels['target']),gray(ROOT/labels['valid']))

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        row=self.records[idx]; raw=rgb(ROOT/row['input']); target,valid=self.labels[row['patch_id']]
        size=self.crop_size
        if self.split=='train':
            rng=np.random.default_rng(SEED+self.epoch*100000+idx)
            x,y=map(int,rng.integers(0,513-size,size=2))
        else:
            x=y=(512-size)//2
        raw=raw[y:y+size,x:x+size]
        image=auxiliaries(raw)
        target=target[y:y+size,x:x+size]; valid=valid[y:y+size,x:x+size]
        return (torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
                torch.from_numpy(target.copy()).unsqueeze(0).float()/255,
                torch.from_numpy(valid.copy()).unsqueeze(0).float()/255)


def validate(model, kind, loader, device):
    model.eval(); losses=[]; tp=fp=fn=0
    with torch.inference_mode():
        for image,target,valid in loader:
            image,target,valid=(a.to(device) for a in (image,target,valid))
            with torch.cuda.amp.autocast(enabled=AMP):
                p=predict(model,kind,image)
            losses.append(float(supervised_loss(p,target,valid)))
            ink=p<.5; truth=target<.5; mask=valid>.5
            tp+=int((ink&truth&mask).sum()); fp+=int((ink&~truth&mask).sum()); fn+=int((~ink&truth&mask).sum())
    return dict(loss=float(np.mean(losses)),ink_f1_to_provisional=2*tp/max(1,2*tp+fp+fn),
                tp=tp,fp=fp,fn=fn,formal_accuracy=False,view='fixed native 256 center crop')


def train(args):
    assert (OUT/'adapter_verification.json').exists()
    for path,digest in json.loads((OUT/'code_hashes.json').read_text()).items():
        assert sha(ROOT/path)==digest,'Freeze code before running experiment'
    destination=OUT/'models'/args.model
    assert not destination.exists(), 'Keep existing runs immutable; use a new experiment for a new run.'
    destination.mkdir(parents=True)
    torch.set_num_threads(2); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.30,device)
    model=make_model(args.model).to(device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    discriminator=None
    if args.model=='ragan':
        _,_,D=reference_classes(); discriminator=D(4).to(device)
        d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=1e-6)
    trainset=PilotDataset('train'); valset=PilotDataset('val')
    loader=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2,
        generator=torch.Generator().manual_seed(SEED),pin_memory=device.type=='cuda')
    val_loader=DataLoader(valset,batch_size=args.batch_size,num_workers=2)
    config=dict(model=args.model,seed=SEED,steps=args.steps,batch_size=args.batch_size,
        lr=args.lr,d_lr=1e-6 if discriminator else None,crop='native random 256, no resize or flip',
        input='Damaged RGB + Otsu + Sobel for all neural models',
        target='Frozen before artificial damage, masked provisional label',
        loss='10 masked L1 + masked BCE + masked ink Dice + masked ink Jaccard',
        adversarial='Author sigmoid-critic scalar-mean hinge D loss; G BCE weight .1; unknown=0.5 in real and fake' if discriminator else None,
        official_reproduction=False,formal_accuracy=False,precision='float32',
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        label_protocol_sha256=sha(OUT/'label_protocol.json'),code_hashes=json.loads((OUT/'code_hashes.json').read_text()))
    dump(destination/'config.json',config)
    initial=validate(model,args.model,val_loader,device)
    history=[dict(step=0,validation=initial)]; dump(destination/'history.json',history)
    scaler=torch.cuda.amp.GradScaler(enabled=AMP)
    d_scaler=torch.cuda.amp.GradScaler(enabled=AMP)
    start=time.monotonic(); step=epoch=0; running=[]
    print(args.model,'initial',initial,flush=True)
    while step<args.steps:
        trainset.epoch=epoch
        for image,target,valid in loader:
            model.train(); image,target,valid=(a.to(device) for a in (image,target,valid))
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=AMP):
                p=predict(model,args.model,image)
            p=p.float(); adversarial=p.new_zeros(()); d_value=None
            if discriminator:
                discriminator.train()
                for parameter in discriminator.parameters(): parameter.requires_grad_(True)
                d_optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=AMP):
                    real=discriminator(critic_input(target,valid),image[:,:3]).float().mean()
                    fake=discriminator(critic_input(p.detach(),valid),image[:,:3]).float().mean()
                d_loss=torch.relu(1-real)+torch.relu(1+fake)
                d_scaler.scale(d_loss).backward(); d_scaler.step(d_optimizer); d_scaler.update()
                discriminator.eval()
                for parameter in discriminator.parameters(): parameter.requires_grad_(False)
                with torch.cuda.amp.autocast(enabled=AMP):
                    judgment=discriminator(critic_input(p,valid),image[:,:3]).float().clamp(1e-6,1-1e-6)
                adversarial=torch.nn.functional.binary_cross_entropy(judgment,torch.ones_like(judgment))
                d_value=float(d_loss.detach())
            loss=supervised_loss(p,target,valid)+.1*adversarial
            assert torch.isfinite(loss), (args.model,step,'nonfinite loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm), (args.model,step,'nonfinite gradient')
            scaler.step(optimizer); scaler.update(); step+=1; running.append(float(loss.detach()))
            if step==1 or step%25==0:
                print(args.model,step,'loss',round(float(loss.detach()),4),'seconds',round(time.monotonic()-start,1),flush=True)
            if step%args.validate_every==0 or step==args.steps:
                metrics=validate(model,args.model,val_loader,device)
                entry=dict(step=step,epoch=epoch,train_loss=float(np.mean(running)),validation=metrics,
                    d_loss=d_value,adversarial=float(adversarial.detach()),seconds=time.monotonic()-start)
                history.append(entry); running=[]; dump(destination/'history.json',history)
                torch.save(dict(kind=args.model,model=model.state_dict(),step=step,
                    optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),config=config,
                    discriminator=discriminator.state_dict() if discriminator else None,
                    d_optimizer=d_optimizer.state_dict() if discriminator else None,
                    d_scaler=d_scaler.state_dict() if discriminator else None),destination/'last.pt')
                print(args.model,'validation',step,metrics,flush=True)
            if step>=args.steps: break
        epoch+=1
    dump(destination/'completion.json',dict(steps=step,epochs_seen=epoch,initial=initial,final=history[-1]['validation'],
        seconds=time.monotonic()-start,training_completed=True,formal_accuracy=False,
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','audit','train'])
    parser.add_argument('--model',choices=['unet','ra_no_gan','ragan'],default='unet')
    parser.add_argument('--steps',type=int,default=432)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--lr',type=float,default=3e-5)
    parser.add_argument('--validate-every',type=int,default=108)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='audit': audit()
    else: train(args)
