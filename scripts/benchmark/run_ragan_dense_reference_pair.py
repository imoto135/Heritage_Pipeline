"""Paired denseGT author-code control. Explicit train action; never auto queue.

64-update smoke run verifies execution only;60epoch run is a small-corpus
comparison, not reproduction of the paper's full training data or reported score.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
from PIL import Image
from compare_v5_aligned_fade_control import dump,sha,save
from ragan_dense_loader_variants import loader_class
from ragan_author_dense_support import ROOT,REFERENCE,GPUUnavailable,gpu_window,author_objectives,make_models,state_digest,tensor_digest,gradient_norm

OUT=ROOT/'tmp_work/ragan_dense_reference_pair_20261023'
FOUNDATION=ROOT/'tmp_work/ragan_reproduction_foundation_20261023'


def read(p):return json.loads(Path(p).read_text())


def freeze():
    assert not (OUT/'protocol.json').exists()
    manifest=read(FOUNDATION/'patch_io.json')
    rows={s:sorted([r for r in manifest['rows'] if r['split']==s],key=lambda r:r['files']['img']['path']) for s in ['train','val']}
    assert len(rows['train'])==497 and len(rows['val'])==113
    rng=np.random.default_rng(1028);schedule=[]
    for epoch in range(60):
        order=rng.permutation(497).tolist()
        for j in range(0,len(order),4):
            batch=order[j:j+4]
            schedule.append(dict(epoch=epoch+1,indices=batch,augmentation_seeds=[1028+epoch*100000+index for index in batch]))
    assert len(schedule)==7500 and len(schedule[-1]['indices'])==1
    dump(OUT/'schedule.json',schedule);dump(OUT/'ordered_patches.json',rows)
    files={}
    for f in [Path(__file__),ROOT/'scripts/benchmark/ragan_author_dense_support.py',ROOT/'scripts/benchmark/ragan_dense_loader_variants.py',ROOT/'scripts/benchmark/verify_ragan_dense_reference_pair.py']:
        files[str(f.relative_to(ROOT))]=sha(f)
    source=read(REFERENCE.parent/'reference_audit.json')['code_sha256']
    for name,h in source.items():
        f=REFERENCE/name;assert sha(f)==h;files[str(f.relative_to(ROOT))]=h
    for f in [FOUNDATION/'patch_io.json',FOUNDATION/'dense_benchmark_manifest.json',OUT/'schedule.json',OUT/'ordered_patches.json']:
        files[str(f.relative_to(ROOT))]=sha(f)
    dump(OUT/'protocol.json',dict(source_files=files,source_commit='4dc8ca7711c36c796400101bcec1648be8ffb359',
        seed=1028,arms=['no_gan','gan'],loader_mode='official',batch_size=4,
        stages=dict(smoke=dict(updates=64,purpose='Execution/numerical signal check only; not GAN efficacy, convergence or failure evidence'),
                    full=dict(updates=7500,epochs=60,purpose='Fixed-final small denseGT corpus comparison; not full-paper numerical reproduction')),
        optimizer=dict(name='Adam',G_lr=3e-5,D_lr=1e-6,betas=[.9,.999],weight_decay=0),
        source_preserved=['Raw authorG/D classes, no GGC/GA/reflection adapters','Author fullGT Loss_Doc','Exact D/G objective assignments extracted from train.py',
                          'SigmoidD + scalar-mean hingeD + BCEG weight.1','D stays train during G update; no D freezing/eval','No probability clamp, gradient clip, mask or AMP'],
        controlled_deviations=['Small explicit16page train/4page val split;2018test never accessed',
            'Explicit native page-derived auxiliary preparation frozen in foundation','Frozen NumPy page-independent epoch permutation rather than global DataLoader RNG',
            'Per-occurrence Python/NumPy augmentation seed shared by botharms','Additional checkpoints/metrics/gradient diagnostics without optimizer or SN-buffer updates'],
        runtime='Float32; cudnn.deterministic=True and cudnn.benchmark=True as source; defaults forTF32. No bitwise CUDA repeat claim.',
        gpu_policy='Require no foreign compute PID on any GPU before CUDA setup, each update, and each validation patch. Unknown status blocks. On interference save own checkpoint and exit; never operate on foreign process.',
        adoption='No best checkpoint selection. Smoke never used for efficacy. Full requires both smoke arms completed with matched initialG and input hashes. No automatic launch of full stage.',
        no_reserved_test=True,not_full_paper_reproduction=True))
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    for f in files:
        if f.startswith('scripts/benchmark/'):
            shutil.copy2(ROOT/f,snap/Path(f).name)
    print('Frozen paired64update smoke and7500update full60epoch comparison; no training started',flush=True)


def check():
    p=read(OUT/'protocol.json')
    for f,h in p['source_files'].items():assert sha(ROOT/f)==h,f
    return p


def datasets():
    rows=read(OUT/'ordered_patches.json');cls=loader_class('official');result={}
    for split in ['train','val']:
        ds=cls(str(FOUNDATION/'patch_io'/split/'img'),(256,256),training=split=='train',only_one=True)
        ds.ImgFiles=[str(ROOT/r['files']['img']['path']) for r in rows[split]];ds.number=len(ds.ImgFiles)
        result[split]=ds
    return result,rows


def validate_data():
    rows=read(OUT/'ordered_patches.json');count=0
    for split in ['train','val']:
        for row in rows[split]:
            for item in row['files'].values():
                assert sha(ROOT/item['path'])==item['sha256'],item['path'];count+=1
    return count


def batch_data(ds,entry):
    samples=[]
    for index,seed in zip(entry['indices'],entry['augmentation_seeds']):
        # Isolate augmentation RNG from model initialization/evaluation/control arm.
        ps,ns=random.getstate(),np.random.get_state()
        try:
            random.seed(seed);np.random.seed(seed);item=ds[index];samples.append(item[:4])
        finally:random.setstate(ps);np.random.set_state(ns)
    return [torch.stack([s[k] for s in samples]) for k in range(4)]


def inspect():
    p=check()
    try:window=gpu_window();status='idle'
    except GPUUnavailable as e:window=dict(reason=str(e));status='blocked_foreign_or_unknown'
    result=dict(protocol_sha256=sha(OUT/'protocol.json'),gpu_status=status,window=window,
        smoke_updates=p['stages']['smoke']['updates'],full_updates=p['stages']['full']['updates'],training_started=False,
        torch_cuda_initialized=torch.cuda.is_initialized())
    dump(OUT/'latest_readiness.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


def pair_ready():
    records=[read(OUT/'smoke'/a/'completion.json') for a in ['no_gan','gan']]
    assert all(r['steps']==64 for r in records)
    assert records[0]['initial_G_digest']==records[1]['initial_G_digest']
    assert records[0]['input_hashes']==records[1]['input_hashes']
    verification=read(OUT/'verification.json');assert verification['no_optimizer_steps'] and verification['objectives_match_literal_source']


def evaluate(g,ds,rows,dest,device):
    pages={p['id']:p for p in read(FOUNDATION/'dense_benchmark_manifest.json')['pages'] if p['split']=='val'}
    maps={k:np.ones((p['height'],p['width']),np.float32) for k,p in pages.items()}
    g.eval()
    for index,row in enumerate(rows):
        gpu_window()
        item=ds[index];x,o,s=[item[k][None].to(device) for k in range(3)]
        with torch.inference_mode():pred=g(x,o,s)[0,0].cpu().numpy()
        x0,y0=row['xy'];w,h=row['valid_wh'];maps[row['page']][y0:y0+h,x0:x0+w]=pred[:h,:w]
    metrics=[]
    for key,a in maps.items():
        gt=np.asarray(Image.open(ROOT/pages[key]['gt']).convert('L'))==0;pred=a<.5
        tp=int((gt&pred).sum());fp=int((~gt&pred).sum());fn=int((gt&~pred).sum())
        f=dest/'validation'/f'{key}.png';save(f,pred)
        metrics.append(dict(page=key,tp=tp,fp=fp,fn=fn,ink_f1=2*tp/max(1,2*tp+fp+fn),mask=str(f.relative_to(ROOT)),mask_sha256=sha(f)))
    dump(dest/'validation.json',dict(rows=metrics,macro_ink_f1=float(np.mean([r['ink_f1'] for r in metrics])),reserved_test_scored=False))


def train(arm,stage,device_name):
    p=check()
    assert (OUT/'verification.json').exists(),'Run no-update CPU verification first'
    assert read(OUT/'verification.json')['protocol_sha256']==sha(OUT/'protocol.json')
    # This precedes model/optimizer construction and any CUDA initialization.
    gpu_window()
    if stage=='full':pair_ready()
    validate_data();gpu_window()
    dest=OUT/stage/arm;dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'completion.json').exists(),'Completed runs are immutable'
    device=torch.device(device_name);assert device.type=='cuda'
    torch.set_num_threads(2);torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=True
    g,d,loss_doc=make_models(p['seed']);initial=state_digest(g)
    g.to(device);d.to(device);loss_doc.to(device)
    # Construct optimizer before first forward as author does, preserving the
    # public GGC's Parameter replacement behavior (do not silently stabilize it).
    opt=torch.optim.Adam(g.parameters(),lr=3e-5);dopt=torch.optim.Adam(d.parameters(),lr=1e-6)
    params=list(g.parameters());objective=author_objectives();data,rows=datasets()
    schedule=read(OUT/'schedule.json')[:p['stages'][stage]['updates']]
    next_step=0;history=[];hashes=[];start=time.monotonic()
    interrupted=dest/'resume.pt'
    if interrupted.exists():
        q=torch.load(interrupted,map_location=device,weights_only=False);assert q['protocol_sha256']==sha(OUT/'protocol.json') and q['initial_G_digest']==initial
        g.load_state_dict(q['g']);d.load_state_dict(q['d']);opt.load_state_dict(q['opt']);dopt.load_state_dict(q['dopt'])
        next_step=q['next_step'];history=q['history'];hashes=q['input_hashes']
        torch.set_rng_state(q['torch_rng'].cpu());torch.cuda.set_rng_state(q['cuda_rng'].cpu(),device);del q
    def checkpoint(status):
        target=dest/'resume.pt' if status!='complete' else dest/'final.pt'
        torch.save(dict(g=g.state_dict(),d=d.state_dict(),opt=opt.state_dict(),dopt=dopt.state_dict(),
            next_step=next_step,history=history,input_hashes=hashes,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),
            protocol_sha256=sha(OUT/'protocol.json'),initial_G_digest=initial,arm=arm,stage=stage,status=status),target)
    try:
        for step in range(next_step,len(schedule)):
            gpu_window();t0=time.monotonic();inputs=batch_data(data['train'],schedule[step]);ih=tensor_digest(inputs)
            x,o,s,t=[a.to(device) for a in inputs];g.train();d.train();dopt.zero_grad()
            pred=g(x,o,s)
            if arm=='gan':
                dl,rs,fs=objective['d'](d,t,x,pred);assert torch.isfinite(dl);dl.backward();dopt.step()
            else:dl=pred.new_zeros(())
            g.zero_grad()
            if arm=='gan':
                total,l1,bce,dice,jaccard,adv=objective['g'](d,loss_doc,x,pred,t)
                supervised=10*l1+bce+dice+jaccard
            else:
                l1,bce,dice,jaccard=loss_doc(x,pred,t);supervised=10*l1+bce+dice+jaccard;total=supervised;adv=pred.new_zeros(())
            assert torch.isfinite(total)
            signal={}
            if step==0:
                signal['supervised_gradient_l2']=gradient_norm(supervised,params)
                if arm=='gan':signal['weighted_adversarial_gradient_l2']=gradient_norm(.1*adv,params)
            total.backward();opt.step()
            hashes.append(ih);history.append(dict(step=step+1,epoch=schedule[step]['epoch'],input_sha256=ih,
                supervised=float(supervised.detach()),adversarial=float(adv.detach()),D=float(dl.detach()),total=float(total.detach()),
                seconds=time.monotonic()-t0,**signal));next_step=step+1
            if next_step%16==0 or next_step==len(schedule):
                checkpoint('in_progress');dump(dest/'history.json',history)
                print(stage,arm,next_step,'loss',history[-1]['total'],'seconds',round(time.monotonic()-start,1),flush=True)
        checkpoint('complete')
        evaluate(g,data['val'],rows['val'],dest,device)
        dump(dest/'completion.json',dict(steps=next_step,arm=arm,stage=stage,initial_G_digest=initial,input_hashes=hashes,
            final_checkpoint_sha256=sha(dest/'final.pt'),protocol_sha256=sha(OUT/'protocol.json'),reserved_test_scored=False,
            seconds_this_session=time.monotonic()-start,not_paper_reproduction=True))
    except GPUUnavailable as e:
        checkpoint('interrupted_foreign_or_unknown_GPU')
        dump(dest/'interruption.json',dict(next_step=next_step,reason=str(e),foreign_process_untouched=True))
        print('Stopped own job:',e,flush=True);raise


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','inspect','train']);a.add_argument('--arm',choices=['gan','no_gan'],default='no_gan')
    a.add_argument('--stage',choices=['smoke','full'],default='smoke');a.add_argument('--device',default='cuda:0');args=a.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='inspect':inspect()
    else:train(args.arm,args.stage,args.device)
