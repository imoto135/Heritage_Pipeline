"""Fixed-generator critic diagnostics: native fakes, added stains, and stroke removal."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from ragan_pilot_support import make_model,predict,auxiliaries,critic_input,reference_classes
from compare_v5_aligned_fade_control import sha,dump,rgb,save
from evaluate_ridge_gate_independent_val import proxies
from evaluate_contact_stain_counterfactual import make_field
from run_contact_stain_training_control import OUT as PARENT,stain,make_training_field
import apply_ragan_real_regions as real

OUT=ROOT/'tmp_work/ragan_discriminator_diagnosis_20261022'
CHECKPOINT=ROOT/'tmp_work/ragan_convergence_audit_20261022/ragan_contact/models/ragan/step_02160.pt'
SEED=20261022
CONDITIONS={
    'reference_lr1e6':dict(lr=1e-6,objective='sigmoid_scalar_hinge'),
    'reference_lr1e4':dict(lr=1e-4,objective='sigmoid_scalar_hinge'),
    'bce_lr1e4':dict(lr=1e-4,objective='sigmoid_bce'),
    'logit_hinge_lr1e4':dict(lr=1e-4,objective='logit_pointwise_hinge'),
    'bce_balanced_corruptions_lr1e4':dict(lr=1e-4,objective='sigmoid_bce',negative_source='balanced_corruptions'),
}
VARIANTS=['real','generator_soft','generator_binary','added_stain','erased_overlap_ink','same_area_relocation','softened_real']


def auc(real_scores,fake_scores):
    real_scores=np.asarray(real_scores);fake_scores=np.asarray(fake_scores)
    n=len(real_scores);m=len(fake_scores)
    ranks=rankdata(np.r_[real_scores,fake_scores])
    return float((ranks[:n].sum()-n*(n+1)/2)/(n*m))


def get_discriminator(payload,objective,device):
    _,_,D=reference_classes();model=D(4)
    model.load_state_dict(payload['discriminator'])
    if objective=='logit_pointwise_hinge':model.shrink[1]=torch.nn.Identity()
    return model.to(device)


def prepare():
    assert not OUT.exists();OUT.mkdir()
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    generator=make_model('ragan').to(device);generator.load_state_dict(payload['model']);generator.eval()
    records={}
    for split in ('train','val'):
        rows=[json.loads(s) for s in (PARENT/'ragan_contact'/(split+'_manifest.jsonl')).read_text().splitlines()]
        selected={}
        for row in sorted(rows,key=lambda r:r['patch_id']):selected.setdefault(row['group'],row)
        records[split]=list(selected.values())
    assert len(records['train'])==24 and len(records['val'])==8
    dump(OUT/'protocol.json',dict(seed=SEED,generator_checkpoint_sha256=sha(CHECKPOINT),
        train_groups=[r['group'] for r in records['train']],val_groups=[r['group'] for r in records['val']],
        selection='lexicographically first patch per existing page group, before critic diagnostics',
        fixed_generator=True,fixed_inputs=True,fixed_native_generator_fakes=True,
        train_examples=24,val_examples=8,input_size=256,
        train_fakes='generator soft output only; hand-crafted corruptions are probes, never training negatives',
        all_conditions_same_initial_D_checkpoint=True,all_conditions_new_Adam=True,
        conditions={k:v for k,v in CONDITIONS.items() if not v.get('negative_source')},steps=800,batch_size=4,eval_steps=[0,200,400,800],
        variants=VARIANTS,source_teacher_provisional=True,known_development_sources=True,
        purpose='stationary discriminator capability/optimization audit, not GAN restoration or model reproduction',
        joint_G_training=False,formal_accuracy=False))
    index=[]
    for split,rows in records.items():
        for row in rows:
            patch=row['patch_id'];source=rgb(ROOT/row['clean'])
            _,core,remote,_=proxies(source)
            field,anchor=make_training_field(core) if split=='train' else make_field(core)
            damaged=stain(source,field,.65)
            cx,cy=anchor;x,y=int(np.clip(cx-128,0,256)),int(np.clip(cy-128,0,256))
            crop=damaged[y:y+256,x:x+256];clean=source[y:y+256,x:x+256]
            target=np.asarray(Image.open(ROOT/row['labels']['target']).convert('L'))[y:y+256,x:x+256].astype(np.float32)/255
            valid=np.asarray(Image.open(ROOT/row['labels']['valid']).convert('L'))[y:y+256,x:x+256]>0
            active=field[y:y+256,x:x+256]>=.5
            paper=valid&(target>.5);ink=valid&(target<.5)
            add_region=paper&active;erase_region=ink&active
            assert add_region.sum()>200 and erase_region.sum()>20,(patch,add_region.sum(),erase_region.sum())
            inp=torch.from_numpy(auxiliaries(crop)).permute(2,0,1).unsqueeze(0).float().to(device)/255
            with torch.inference_mode():p=predict(generator,'ragan',inp)[0,0].cpu().numpy()
            corrupt={}
            corrupt['real']=target.copy()
            corrupt['generator_soft']=p
            corrupt['generator_binary']=(p>=.5).astype(np.float32)
            corrupt['added_stain']=np.where(add_region,0,target)
            corrupt['erased_overlap_ink']=np.where(erase_region,1,target)
            # Move an equal number of known ink pixels to a compact paper blob.
            n=int(min(ink.sum()//4,add_region.sum()//2))
            yy,xx=np.mgrid[:256,:256];distance=(xx-(cx-x))**2+(yy-(cy-y))**2
            add_indices=np.flatnonzero(add_region);add_indices=add_indices[np.argsort(distance.flat[add_indices],kind='stable')[:n]]
            remove_indices=np.flatnonzero(ink);remove_indices=remove_indices[np.argsort(-distance.flat[remove_indices],kind='stable')[:n]]
            relocated=target.copy();relocated.flat[add_indices]=0;relocated.flat[remove_indices]=1
            assert int((relocated[valid]<.5).sum())==int(ink.sum())
            corrupt['same_area_relocation']=relocated
            corrupt['softened_real']=cv2.GaussianBlur(target,(0,0),.8)
            folder=OUT/'cases'/split/patch;folder.mkdir(parents=True)
            Image.fromarray(clean).save(folder/'source.png');Image.fromarray(crop).save(folder/'input.png')
            Image.fromarray(np.rint(target*255).astype(np.uint8)).save(folder/'paper_target.png')
            Image.fromarray(valid.astype(np.uint8)*255).save(folder/'valid.png')
            np.savez_compressed(folder/'arrays.npz',image=crop,source=clean,target=target,valid=valid,
                add_region=add_region,erase_region=erase_region,**{f'p_{k}':v for k,v in corrupt.items()})
            items=[('原本',clean),('固定入力',crop)]
            for key in VARIANTS:
                a=corrupt[key];items.append((key,np.rint(a*255).astype(np.uint8)))
            if split=='val':real.panel(folder/'原本と識別器診断用画像.png',patch+' / 改変は識別器の診断だけに使用',items)
            index.append(dict(split=split,patch=patch,group=row['group'],crop_xy=[x,y],
                source_sha256=sha(ROOT/row['clean']),cache=str((folder/'arrays.npz').relative_to(OUT)),
                cache_sha256=sha(folder/'arrays.npz'),known_ink=int(ink.sum()),known_paper=int(paper.sum()),
                added_stain_pixels=int(add_region.sum()),erased_ink_pixels=int(erase_region.sum()),relocated_pixels=n))
    dump(OUT/'index.json',dict(rows=index));print('Prepared',len(index),'fixed discriminator cases',flush=True)


def dataset(split,device):
    rows=[r for r in json.loads((OUT/'index.json').read_text())['rows'] if r['split']==split]
    arrays=[]
    for row in rows:
        path=OUT/row['cache'];assert sha(path)==row['cache_sha256'];arrays.append(dict(np.load(path)))
    result=dict(rows=rows)
    result['rgb']=torch.from_numpy(np.stack([a['image'] for a in arrays])).permute(0,3,1,2).float().to(device)/255
    result['valid']=torch.from_numpy(np.stack([a['valid'] for a in arrays])).unsqueeze(1).float().to(device)
    for key in VARIANTS:
        result[key]=torch.from_numpy(np.stack([a['p_'+key] for a in arrays])).unsqueeze(1).float().to(device)
    for key in ('add_region','erase_region'):
        result[key]=torch.from_numpy(np.stack([a[key] for a in arrays])).unsqueeze(1).to(device)
    return result


def probe(model,data,objective):
    model.eval();scores={key:[] for key in VARIANTS};directions=[]
    for start in range(0,len(data['rows']),4):
        stop=start+4;rgb_=data['rgb'][start:stop];valid=data['valid'][start:stop]
        for key in VARIANTS:
            p=critic_input(data[key][start:stop],valid)
            with torch.inference_mode():s=model(p,rgb_).mean(1)
            scores[key].extend(s.cpu().tolist())
        for key,region_key in [('added_stain','add_region'),('erased_overlap_ink','erase_region')]:
            p=data[key][start:stop].detach().clone().requires_grad_(True)
            score=model(critic_input(p,valid),rgb_)
            loss=-score.mean() if objective=='logit_pointwise_hinge' else F.binary_cross_entropy(score,torch.ones_like(score))
            grad=torch.autograd.grad(loss,p)[0]
            for i in range(p.shape[0]):
                region=data[region_key][start+i];values=grad[i][region]
                desired=(values<0) if key=='added_stain' else (values>0)
                directions.append(dict(patch=data['rows'][start+i]['patch'],variant=key,
                    n=int(values.numel()),desired_direction_fraction=float(desired.float().mean()),
                    gradient_abs_mean=float(values.abs().mean()),gradient_signed_mean=float(values.mean())))
    pairs=[]
    for key in VARIANTS[1:]:
        delta=np.asarray(scores['real'])-np.asarray(scores[key])
        pairs.append(dict(variant=key,auc_real_vs_variant=auc(scores['real'],scores[key]),
            paired_real_higher=float((delta>0).mean()),mean_real_minus_variant=float(delta.mean())))
    return dict(scores=scores,pairs=pairs,correction_direction=directions)


def run(condition,device_name):
    conf=CONDITIONS[condition];dest=OUT/condition;assert not dest.exists();dest.mkdir()
    if conf.get('negative_source'):assert (OUT/'balanced_negative_protocol.json').exists()
    else:assert condition in json.loads((OUT/'protocol.json').read_text())['conditions']
    torch.set_num_threads(2);torch.manual_seed(SEED);device=torch.device(device_name);torch.cuda.set_device(device)
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    model=get_discriminator(payload,conf['objective'],device)
    optimizer=torch.optim.Adam(model.parameters(),lr=conf['lr'])
    train=dataset('train',device);val=dataset('val',device)
    rng=np.random.default_rng(SEED);history=[];start=time.monotonic()
    for step in range(801):
        if step in (0,200,400,800):
            measures={s:probe(model,d,conf['objective']) for s,d in [('train',train),('val',val)]}
            entry=dict(step=step,seconds=time.monotonic()-start,**measures);history.append(entry)
            dump(dest/'history.json',history)
            brief=[(r['variant'],round(r['auc_real_vs_variant'],3),round(r['paired_real_higher'],3)) for r in entry['val']['pairs']]
            print(condition,step,brief,flush=True)
            if step==800:break
        indices=rng.choice(len(train['rows']),4,replace=False)
        model.train();optimizer.zero_grad(set_to_none=True)
        valid=train['valid'][indices];rgb_=train['rgb'][indices]
        rs=model(critic_input(train['real'][indices],valid),rgb_)
        if conf.get('negative_source')=='balanced_corruptions':
            keys=['generator_soft','added_stain','erased_overlap_ink','same_area_relocation']
            fake=torch.stack([train[keys[(step+i)%4]][j] for i,j in enumerate(indices)])
        else:fake=train['generator_soft'][indices]
        fs=model(critic_input(fake,valid),rgb_)
        if conf['objective']=='sigmoid_scalar_hinge':loss=F.relu(1-rs.mean())+F.relu(1+fs.mean())
        elif conf['objective']=='sigmoid_bce':loss=F.binary_cross_entropy(rs,torch.ones_like(rs))+F.binary_cross_entropy(fs,torch.zeros_like(fs))
        else:loss=F.relu(1-rs).mean()+F.relu(1+fs).mean()
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    torch.save(dict(discriminator=model.state_dict(),optimizer=optimizer.state_dict(),condition=condition,steps=800),dest/'last.pt')
    dump(dest/'completion.json',dict(condition=condition,config=conf,steps=800,seconds=time.monotonic()-start,
        generator_updated=False,checkpoint_sha256=sha(CHECKPOINT),final=history[-1],formal_accuracy=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train'])
    parser.add_argument('--condition',choices=list(CONDITIONS));parser.add_argument('--device',default='cuda:0');a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:assert a.condition;run(a.condition,a.device)
