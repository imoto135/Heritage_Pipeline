"""Four from-scratch, input-matched UNet/public-G supervised runs."""
import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_occurrence_jitter_control import bounds,OUT as PARENT
from ragan_local_region_support import region_masks
from matched_generator_support import hooked,logit_of,objective,checks
from ragan_pilot_support import make_model,predict,auxiliaries
from run_ragan_joint_continuation import arrays
from compare_v5_aligned_fade_control import dump,sha
from audit_ragan_inference_context import OUT as CONTEXT

OUT=ROOT/'tmp_work/matched_supervised_generators_20261022'
CONDITIONS={f'{kind}_seed{s}':dict(kind=kind,seed=20261021+s,seed_index=s) for s in (0,1) for kind in ['unet','ragan']}
STEPS=3024


def freeze():
    assert not OUT.exists();OUT.mkdir()
    records=full.read_rows('train');meta=full.metadata();masks=region_masks(records)
    val=full.read_rows('val')
    assert {r['group'] for r in records}.isdisjoint({r['group'] for r in val})
    targets=[i for i,r in enumerate(records) if r['patch_id'] in masks and r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8]
    assert len(records)==864 and len(targets)==7
    legal={p:bounds(m['core']|m['paper']) for p,m in masks.items()}
    audits=[]
    for s in (0,1):
        seed=20261021+s;plan=[];batches=[];counts=Counter();usage={p:[] for p in masks}
        rngs={p:np.random.default_rng(7102026+next(i for i,r in enumerate(records) if r['patch_id']==p)*7919) for p in masks};last={}
        for epoch in range(14):
            indices=torch.randperm(864,generator=torch.Generator().manual_seed(seed+epoch*100000)).numpy()
            rng=np.random.default_rng(seed+epoch*100000+709)
            replace=rng.choice(np.flatnonzero(~np.isin(indices,targets)),166,replace=False)
            additions=np.concatenate([rng.permutation(targets) for _ in range(24)])[:166]
            indices[replace]=additions
            assert int(np.isin(indices,targets).sum())==173
            for k in range(0,864,4):
                batches.append(dict(epoch=epoch,indices=indices[k:k+4].tolist()))
            for index in indices:
                index=int(index);row=records[index];patch=row['patch_id'];active=index in targets
                xy=tuple(full.coordinates(index,epoch,row,meta))
                if active:
                    lo,hi=legal[patch]
                    while True:
                        xy=tuple(int(rngs[patch].integers(a,b+1)) for a,b in zip(lo,hi))
                        if xy!=last.get(patch):break
                    last[patch]=xy;x,y=xy;m=masks[patch]
                    assert m['core'][y:y+256,x:x+256].sum()==m['core'].sum() and m['paper'][y:y+256,x:x+256].sum()==m['paper'].sum()
                    counts['active_examples']+=1;counts['core_pixels']+=int(m['core'].sum());counts['paper_pixels']+=int(m['paper'].sum())
                    usage[patch].append(list(xy))
                plan.append(dict(index=index,epoch=epoch,xy=list(xy),active=active))
        assert len(batches)==STEPS and len(plan)==STEPS*4 and counts['active_examples']==2422
        dump(OUT/'schedules'/f'seed{s}.json',batches);dump(OUT/'crops'/f'seed{s}.json',plan)
        audits.append(dict(seed_index=s,exposures=dict(counts),patches=[dict(patch=p,draws=len(xs),distinct_positions=len({tuple(x) for x in xs}),legal_bounds=legal[p]) for p,xs in usage.items()]))
    dump(OUT/'crop_exposure_audit.json',audits)
    shutil.copy2(PARENT/'source_hashes.json',OUT/'source_hashes.json')
    shutil.copytree(PARENT/'region_masks',OUT/'region_masks')
    for p,m in masks.items():
        for k in ['core','paper','roi','known_paper']:
            assert np.array_equal(full.gray(OUT/'region_masks'/(p+'_'+k+'.png'))>0,m[k])
    dump(OUT/'loss_checks.json',checks())
    counts={}
    for kind in ['unet','ragan']:
        torch.manual_seed(20261021);model=make_model(kind)
        counts[kind]=dict(parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    dump(OUT/'protocol.json',dict(conditions=CONDITIONS,architecture_parameters=counts,
        comparison='generator architecture only within each paired seed; architecture includes parameter count, normalization, sampling and output head',
        training='fresh random model and fresh Adam in all4 runs; no warm start; no discriminator at any stage',
        epochs=14,steps=STEPS,batch_size=4,lr=3e-5,gradient_clip=1.,optimizer='Adam default betas(.9,.999)',
        common_loss='previous corrected BCE +10L1+inkDice+inkJaccard +per-example core mean BCE+paper mean BCE; coefficients unchanged',
        output_logits=dict(unet='single pre-sigmoid output',ragan='exact product-head logit a+b-logsumexp(0,a,b)'),
        common_input='RGB+after-crop Otsu+Sobel; native256; all864 RGB and label files unchanged',
        common_sampling='173/864 reviewed strong-fade draws per epoch from epoch0; fixed per-presentation crops; same file per paired seed',
        schedules_sha256={str(s):sha(OUT/'schedules'/f'seed{s}.json') for s in (0,1)},
        crop_plans_sha256={str(s):sha(OUT/'crops'/f'seed{s}.json') for s in (0,1)},
        train_manifest_sha256=sha(full.PARENT/'train_manifest.jsonl'),val_manifest_sha256=sha(full.PARENT/'val_manifest.jsonl'),
        train_patches=96,train_page_groups=24,train_records=864,val_groups=8,independent_initialization_seeds=True,
        same_seed_does_not_mean_equal_initial_weights_across_architectures=True,parameter_counts_not_matched=True,
        primary_step=3024,secondary_step=2160,saved_steps=[432,2160,3024],eval_steps=[0,432,864,1296,1728,2160,2592,3024],threshold=.5,
        official_reproduction=False,public_GA_cancellation_remains=True,labels_provisional=True,known_development=True,formal_accuracy=False,
        prior_continuations_not_equal_training_controls=True,no_damage_changes=True,no_label_changes=True))
    dump(OUT/'evaluation_plan.json',dict(primary_step=3024,secondary_step=2160,all4_conditions_reported=True,no_best_checkpoint_selection=True,
        core_cases='same7 training traces455; original epoch13 canonical windows; hard44 x origins0,4,...32 from full auxiliary image',
        transfer='same6 synthetic cases, known49, D01 D03 D04 T07 C01, same partial labels and fixed Howe',
        native_val8='supplementary provisional labels only',tiling_primary_only=['native256_stride128_grid0','native256_stride128_grid16'],
        inherited_tiling_protocol=str((CONTEXT/'tiling_protocol.json').relative_to(ROOT)),inherited_tiling_sha256=sha(CONTEXT/'tiling_protocol.json'),
        exploratory_interpretation='UNet-only success suggests generator dependence; both failures require shared teacher/objective audit; both success suggests earlier optimization/history contributed',
        known_development=True,formal_accuracy=False))
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for name in [Path(__file__).name,'matched_generator_support.py','ragan_pilot_support.py','run_ragan_stable_bce_control.py','ragan_local_region_support.py','run_ragan_full_continuation.py']:
        shutil.copy2(ROOT/'scripts/benchmark'/name,snapshot/name)
    print('Frozen4 fresh supervised runs;3024 updates; paired identical schedules/crops/objectives',flush=True)


class Data(full.Crops):
    def __init__(self,rows):super().__init__(rows);self.regions=region_masks(rows)
    def sample(self,entry):
        row=self.rows[entry['index']];x,y=entry['xy'];raw=self.inputs[row['input']][y:y+256,x:x+256]
        t,v=self.labels[row['patch_id']]
        if entry['active']:
            m=self.regions[row['patch_id']];positive=m['core'][y:y+256,x:x+256];negative=m['paper'][y:y+256,x:x+256]
        else:positive=negative=np.zeros((256,256),bool)
        return (torch.from_numpy(auxiliaries(raw).copy()).permute(2,0,1).float()/255,
            torch.from_numpy(t[y:y+256,x:x+256].copy()).unsqueeze(0).float()/255,
            torch.from_numpy(v[y:y+256,x:x+256].copy()).unsqueeze(0).float()/255,
            torch.from_numpy(positive.copy()).unsqueeze(0),torch.from_numpy(negative.copy()).unsqueeze(0))


def native(model,kind,device,validation):
    model.eval();tp=fp=fn=0
    with torch.inference_mode():
        for a,x in validation:
            pred=predict(model,kind,x)[0,0].cpu().numpy()<.5;t=a['target']<.5;v=a['valid']
            tp+=int((pred&t&v).sum());fp+=int((pred&~t&v).sum());fn+=int((~pred&t&v).sum())
    return dict(tp=tp,fp=fp,fn=fn,ink_f1_to_provisional=2*tp/max(1,2*tp+fp+fn))


def train(condition,device_name):
    p=json.loads((OUT/'protocol.json').read_text());conf=CONDITIONS[condition];kind=conf['kind'];s=conf['seed_index']
    dest=OUT/condition;assert not dest.exists();dest.mkdir()
    assert all(sha(ROOT/path)==h for path,h in json.loads((OUT/'source_hashes.json').read_text()).items())
    assert sha(OUT/'schedules'/f'seed{s}.json')==p['schedules_sha256'][str(s)] and sha(OUT/'crops'/f'seed{s}.json')==p['crop_plans_sha256'][str(s)]
    torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device);torch.manual_seed(conf['seed'])
    model,state=hooked(kind);model=model.to(device)
    opt=torch.optim.Adam([q for q in model.parameters() if q.requires_grad],lr=p['lr'])
    assert not opt.state
    torch.save(dict(model=model.state_dict(),kind=kind,seed=conf['seed'],step=0,protocol_sha256=sha(OUT/'protocol.json')),dest/'initial.pt')
    data=Data(full.read_rows('train'));plan=json.loads((OUT/'crops'/f'seed{s}.json').read_text());schedule=json.loads((OUT/'schedules'/f'seed{s}.json').read_text())
    _,aa=arrays('val');validation=[(a,torch.from_numpy(auxiliaries(a['image']).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255) for a in aa]
    history=[];losses=[];start=time.monotonic()
    for step in range(STEPS+1):
        if step in p['eval_steps']:
            record=dict(step=step,validation=native(model,kind,device,validation),seconds=time.monotonic()-start)
            if losses:record['mean_training_objective']=float(np.mean([r['total'] for r in losses[-432:]]))
            history.append(record);dump(dest/'history.json',history)
            print(condition,step,record['validation'],'seconds',round(record['seconds'],1),flush=True)
            if step in p['saved_steps']:
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,kind=kind,seed=conf['seed'],condition=condition,
                    protocol_sha256=sha(OUT/'protocol.json'),discriminator=None),dest/f'step_{step:05d}.pt')
            if step==STEPS:break
        entries=plan[step*4:step*4+4];assert [e['index'] for e in entries]==schedule[step]['indices']
        samples=[data.sample(e) for e in entries]
        x,t,v,positive,negative=[torch.stack([a[j] for a in samples]).to(device) for j in range(5)]
        assert not (positive&((v==0)|(t>.5))).any() and not (negative&((v==0)|(t<.5))).any()
        model.train();opt.zero_grad(set_to_none=True);prob=predict(model,kind,x)
        loss,parts=objective(prob,t,v,logit_of(kind,state['raw']),positive,negative)
        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']);assert torch.isfinite(norm);opt.step()
        losses.append(dict(total=float(loss.detach()),**{k:float(q.detach()) for k,q in parts.items()},
            active_examples=sum(e['active'] for e in entries),positive_pixels=int(positive.sum()),paper_pixels=int(negative.sum())))
    dump(dest/'loss_components.json',losses)
    dump(dest/'completion.json',dict(condition=condition,kind=kind,seed=conf['seed'],fresh_model=True,fresh_Adam=True,
        initial_optimizer_steps=[],final_optimizer_steps=sorted({int(st['step']) for st in opt.state.values() if 'step' in st}),
        input_draws=len(plan),active_draws=sum(e['active'] for e in plan),schedule_sha256=sha(OUT/'schedules'/f'seed{s}.json'),
        crop_plan_sha256=sha(OUT/'crops'/f'seed{s}.json'),seconds=time.monotonic()-start,
        numeric_configuration=dict(torch=torch.__version__,cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32),formal_accuracy=False))
    print('Training completed',condition,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','train']);parser.add_argument('--condition',choices=list(CONDITIONS));parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    freeze() if args.action=='freeze' else train(args.condition,args.device)
