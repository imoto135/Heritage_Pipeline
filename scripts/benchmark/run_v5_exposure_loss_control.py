"""One-factor RA-GAN/U-Net controls for fade exposure and source-core loss."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
OUT=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
RUNNER=ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot/run_makura_supervised_pilot.py'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_v5_aligned_fade_control import crop_xy,trace_centers,OLD,SEED,sha,dump

def load_rows(p):return [json.loads(line) for line in p.read_text().splitlines()]
def core_masks():
    paths={}
    for folder in (ROOT/'tmp_work/makura_additional_thin_cores_20261021/masks',
                   ROOT/'tmp_work/makura_thin_width_candidate_20261021/masks'):
        for p in folder.glob('*_core.png'):
            patch=p.name.removesuffix('_core.png')
            if patch=='200017197_00172_0_p00':continue
            paths[patch]=p
    assert len(paths)==7
    return paths

def prepare():
    assert not OUT.exists(),OUT
    OUT.mkdir(parents=True)
    cores=core_masks();probes=[]
    for stem,kind in (('unet_v3','unet'),('ragan_public_v3','ragan')):
        parent=BASE/(stem+'_v5');rows=load_rows(parent/'train_manifest.jsonl')
        targets=[i for i,r in enumerate(rows) if r['patch_id'] in cores and
                 r['damage_family']=='targeted_fade' and float(r['damage_strength'])==.8]
        assert len(rows)==864 and len(targets)==7
        for factor in ('exposure','core_loss'):
            dest=OUT/(stem+'_'+factor);dest.mkdir()
            for name in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
                (dest/name).write_bytes((parent/name).read_bytes())
            dump(dest/'condition.json',dict(model=kind,factor=factor,parent=str(parent.relative_to(ROOT)),
                steps=432,batch_size=4,seed=SEED,all_RGB_and_labels_fixed=True,
                exposure_target_probability=.20 if factor=='exposure' else None,
                core_pixel_weight=8 if factor=='core_loss' else 1,
                validation_unweighted=True,formal_accuracy=False))
            probes.append(dest.name)
    dump(OUT/'protocol.json',dict(conditions=probes,baseline=str(BASE.relative_to(ROOT)),
        target_rows='seven traced train patches, targeted_fade 0.80',n_target_rows=7,
        exposure_probability=.20,exposure_num_samples_per_epoch=864,epochs=2,
        core_pixel_weight=8,other_valid_pixel_weight=1,
        same_input_RGB=True,same_training_labels=True,same_validation=True,
        same_steps=432,same_seed=SEED,formal_accuracy=False,dkds_reproduction=False))
    print('prepared',probes,flush=True)

def weighted_loss(paper_probability,paper_target,valid):
    p=paper_probability.float().clamp(1e-6,1-1e-6)
    t=paper_target.float();w=valid.float();n=w.sum().clamp_min(1)
    l1=((p-t).abs()*w).sum()/n
    bce=(F.binary_cross_entropy(p,t,reduction='none')*w).sum()/n
    ink=1-p;truth=1-t;intersection=(ink*truth*w).sum()
    ink_sum=(ink*w).sum();truth_sum=(truth*w).sum()
    dice=1-(2*intersection+1e-6)/(ink_sum+truth_sum+1e-6)
    jaccard=1-(intersection+1e-6)/(ink_sum+truth_sum-intersection+1e-6)
    return 10*l1+bce+dice+jaccard

def run(condition,device):
    assert (OUT/condition/'condition.json').exists()
    sys.path.insert(0,str(RUNNER.parent))
    import ragan_pilot_support as support
    support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
    spec=importlib.util.spec_from_file_location('fixed_pilot_v5_ablation',RUNNER)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner.ROOT=ROOT;runner.OUT=OUT/condition
    conf=json.loads((runner.OUT/'condition.json').read_text());factor=conf['factor']
    centers=trace_centers()
    generic={r['patch']:r['selected_generic_xy'] for r in json.loads((OLD/'input_audit.json').read_text())['rows'] if r['strength']==.8}
    cores={patch:np.asarray(Image.open(path).convert('L'))>0 for patch,path in core_masks().items()}
    class CropDataset(runner.PilotDataset):
        def __getitem__(self,idx):
            row=self.records[idx]
            if self.split!='train':return super().__getitem__(idx)
            if row['damage_family']=='targeted_fade':x,y=crop_xy(idx,self.epoch,row,centers,generic)
            else:
                rng=np.random.default_rng(SEED+self.epoch*100000+idx)
                x,y=map(int,rng.integers(0,257,size=2))
            raw=runner.rgb(ROOT/row['input'])[y:y+256,x:x+256]
            target,valid=self.labels[row['patch_id']]
            target=target[y:y+256,x:x+256];valid=(valid[y:y+256,x:x+256]>0).astype(np.float32)
            if factor=='core_loss' and row['patch_id'] in cores:
                core=cores[row['patch_id']][y:y+256,x:x+256]
                valid[core&(target==0)&(valid>0)]=8
            image=runner.auxiliaries(raw)
            return (torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
                torch.from_numpy(target.copy()).unsqueeze(0).float()/255,
                torch.from_numpy(valid.copy()).unsqueeze(0).float())
    runner.PilotDataset=CropDataset
    if factor=='core_loss':
        # Decode weights only for the critic; the supervised terms use weights.
        runner.supervised_loss=weighted_loss
        runner.critic_input=lambda p,v:support.critic_input(p,(v>0).float())
        # Validation uses binary masks, making weighted_loss identical to the baseline formula.
        probe=torch.tensor([[[[.2,.8]]]])
        truth=torch.tensor([[[[0.,1.]]]]);valid=torch.ones_like(truth)
        torch.testing.assert_close(weighted_loss(probe,truth,valid),support.supervised_loss(probe,truth,valid))
    else:
        original_loader=runner.DataLoader
        def sampled_loader(dataset,*args,**kwargs):
            if dataset.split!='train':return original_loader(dataset,*args,**kwargs)
            rows=dataset.records
            target=np.array([r['patch_id'] in cores and r['damage_family']=='targeted_fade' and
                float(r['damage_strength'])==.8 for r in rows])
            assert target.sum()==7
            weights=np.where(target,.20/7,.80/(len(rows)-7))
            generator=torch.Generator().manual_seed(SEED)
            sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),
                num_samples=864,replacement=True,generator=generator)
            # Store the exact sampled index stream without advancing the training sampler.
            audit_sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),
                num_samples=864,replacement=True,generator=torch.Generator().manual_seed(SEED))
            indices=list(audit_sampler)+list(audit_sampler)
            dump(OUT/condition/'sample_audit.json',dict(draws=len(indices),target_draws=int(target[indices].sum()),
                per_patch={p:int(sum(rows[i]['patch_id']==p and target[i] for i in indices)) for p in cores},
                target_probability_configured=.20))
            kwargs.pop('shuffle',None);kwargs.pop('generator',None)
            return original_loader(dataset,*args,sampler=sampler,shuffle=False,**kwargs)
        runner.DataLoader=sampled_loader
    args=argparse.Namespace(model=conf['model'],steps=432,batch_size=4,lr=3e-5,validate_every=108,device=device)
    runner.train(args)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--condition',choices=['unet_v3_exposure','unet_v3_core_loss','ragan_public_v3_exposure','ragan_public_v3_core_loss'])
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.condition,args.device)
