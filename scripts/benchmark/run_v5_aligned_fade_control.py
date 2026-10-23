"""Matched v4/v5 pilot on the same trace-aligned training fade inputs/crops."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
V4=ROOT/'tmp_work/makura_teacher_v4_width_20261021'
V5=ROOT/'tmp_work/makura_teacher_v5_diverse_20261021'
CORES=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
OLD=ROOT/'tmp_work/targeted_fade_training_20261021'
OUT=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
RUNNER=ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot/run_makura_supervised_pilot.py'
SEED=20261021

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def gray(a):
    import cv2
    return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)

def trace_centers():
    centers={}
    for patch in ('200017197_00103_1_p00','200017197_00214_1_p00'):
        path=ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png')
        p=np.asarray(Image.open(path).convert('L'))>0;y,x=np.where(p);centers[patch]=[int(np.rint((x.min()+x.max())/2)),int(np.rint((y.min()+y.max())/2))]
    for row in json.loads((CORES/'audit.json').read_text())['rows']:
        patch=row['patch'];p=np.asarray(Image.open(CORES/'masks'/(patch+'_trace.png')).convert('L'))>0
        y,x=np.where(p);centers[patch]=[int(np.rint((x.min()+x.max())/2)),int(np.rint((y.min()+y.max())/2))]
    assert len(centers)==7
    return centers

def crop_xy(idx,epoch,row,centers,generic):
    patch=row['patch_id']
    if patch in centers:x,y=centers[patch]
    else:
        options=generic[patch];x,y=options[(idx+epoch)%len(options)]
    rng=np.random.default_rng(SEED+epoch*100000+idx)
    dx,dy=map(int,rng.integers(-16,17,size=2))
    return int(np.clip(x-128+dx,0,256)),int(np.clip(y-128+dy,0,256))

def prepare():
    assert not OUT.exists(),f'Preserve existing experiment: {OUT}'
    OUT.mkdir(parents=True)
    from run_targeted_fade_training import build_input
    base=[json.loads(x) for x in (V5/'unet_v3_v5teacher/train_manifest.jsonl').read_text().splitlines()]
    extra={r['patch'] for r in json.loads((CORES/'audit.json').read_text())['rows']}
    generic={r['patch']:r['selected_generic_xy'] for r in json.loads((OLD/'input_audit.json').read_text())['rows'] if r['strength']==.8}
    assert len(extra)==5 and len(generic)==96
    new_inputs={};input_audit=[]
    for patch in sorted(extra):
        rec=next(r for r in base if r['patch_id']==patch)
        source=rgb(ROOT/rec['clean']);trace=np.asarray(Image.open(CORES/'masks'/(patch+'_trace.png')).convert('L'))>0
        center=trace.copy()
        for x,y in generic[patch]:center[y,x]=True
        for strength in (.3,.55,.8):
            out,field,paper=build_input(source,None,center,strength)
            path=OUT/'inputs'/patch/(f'fade_{strength:.2f}.png');path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(out).save(path)
            new_inputs[(patch,strength)]=str(path.relative_to(ROOT))
            y,x=np.where(trace);delta=gray(out).astype(np.int16)[y,x]-gray(source).astype(np.int16)[y,x]
            input_audit.append(dict(patch=patch,strength=strength,n=len(x),changed_ge10=int((np.abs(delta)>=10).sum()),
                median_delta=float(np.median(delta)),input_sha256=sha(path),source_sha256=sha(ROOT/rec['clean'])))
    assert len(new_inputs)==15
    all_conditions=[]
    for model,stem in [('unet','unet_v3'),('ragan','ragan_public_v3')]:
        for teacher,parent in [('v4',V4/(stem+'_v4teacher')),('v5',V5/(stem+'_v5teacher'))]:
            dest=OUT/(stem+'_'+teacher);dest.mkdir()
            revised=[]
            for line in (parent/'train_manifest.jsonl').read_text().splitlines():
                r=json.loads(line)
                key=(r['patch_id'],float(r['damage_strength']))
                if r['damage_family']=='targeted_fade' and key in new_inputs:
                    r['input']=new_inputs[key];r['input_sha256']=sha(ROOT/r['input'])
                revised.append(r)
            (dest/'train_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in revised))
            for name in ('val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
                (dest/name).write_bytes((parent/name).read_bytes())
            dump(dest/'condition.json',dict(model=model,teacher=teacher,parent=str(parent.relative_to(ROOT)),
                same_training_inputs_across_teachers=True,same_crop_schedule=True,seed=SEED,steps=432,batch_size=4,
                formal_accuracy=False))
            all_conditions.append(dest)
    for stem in ('unet_v3','ragan_public_v3'):
        a=[json.loads(x) for x in (OUT/(stem+'_v4')/'train_manifest.jsonl').read_text().splitlines()]
        b=[json.loads(x) for x in (OUT/(stem+'_v5')/'train_manifest.jsonl').read_text().splitlines()]
        assert len(a)==len(b)==864
        assert all(x['input_sha256']==y['input_sha256'] for x,y in zip(a,b))
        assert sum(x['labels']['target_sha256']!=y['labels']['target_sha256'] for x,y in zip(a,b))==45
        assert sha(OUT/(stem+'_v4')/'val_manifest.jsonl')==sha(OUT/(stem+'_v5')/'val_manifest.jsonl')
    centers=trace_centers();schedule=[]
    for i,r in enumerate(a):
        if r['damage_family']!='targeted_fade':continue
        for epoch in (0,1):
            x,y=crop_xy(i,epoch,r,centers,generic)
            entry=dict(index=i,patch=r['patch_id'],strength=r['damage_strength'],epoch=epoch,xy=[x,y])
            if r['patch_id'] in centers:
                path=(CORES/'masks'/(r['patch_id']+'_trace.png')) if r['patch_id'] in extra else (ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(r['patch_id']+'_high_centers.png'))
                mask=np.asarray(Image.open(path).convert('L'))>0;yy,xx=np.where(mask)
                entry['trace_n']=len(xx);entry['trace_in_crop']=int(((xx>=x)&(xx<x+256)&(yy>=y)&(yy<y+256)).sum())
                assert entry['trace_in_crop']==entry['trace_n'],entry
            schedule.append(entry)
    dump(OUT/'input_audit.json',dict(rows=input_audit))
    dump(OUT/'crop_schedule.json',dict(rows=schedule))
    dump(OUT/'protocol.json',dict(conditions=[x.name for x in all_conditions],
        changed_input_patches=sorted(extra),changed_input_rows=15,
        same_training_RGB_within_each_v4_v5_pair=True,same_crops=True,
        same_v2_validation=True,source_traces_used_for_training_only=True,
        training_pages_with_traces=7,steps=432,batch_size=4,seed=SEED,
        labels='v4 versus v5 provisional source-visible core labels',
        no_validation_trace_in_training=True,formal_accuracy=False,dkds_reproduction=False))
    print('Prepared 4 matched runs,',len(input_audit),'new aligned fades and',len(schedule),'scheduled fade crops',flush=True)
    for r in input_audit:
        if r['strength']==.8:print(r['patch'],'strong fade centers >=10:',r['changed_ge10'],'/',r['n'],flush=True)

def run(condition,device):
    sys.path.insert(0,str(RUNNER.parent))
    import ragan_pilot_support as support
    support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
    spec=importlib.util.spec_from_file_location('fixed_pilot_v5',RUNNER)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner.ROOT=ROOT;runner.OUT=OUT/condition
    centers=trace_centers()
    generic={r['patch']:r['selected_generic_xy'] for r in json.loads((OLD/'input_audit.json').read_text())['rows'] if r['strength']==.8}
    base=runner.PilotDataset
    class CropDataset(base):
        def __getitem__(self,idx):
            row=self.records[idx]
            if self.split!='train' or row['damage_family']!='targeted_fade':return super().__getitem__(idx)
            x,y=crop_xy(idx,self.epoch,row,centers,generic)
            raw=runner.rgb(ROOT/row['input'])[y:y+256,x:x+256]
            target,valid=self.labels[row['patch_id']]
            image=runner.auxiliaries(raw)
            target=target[y:y+256,x:x+256];valid=valid[y:y+256,x:x+256]
            return (torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
                torch.from_numpy(target.copy()).unsqueeze(0).float()/255,
                torch.from_numpy(valid.copy()).unsqueeze(0).float()/255)
    runner.PilotDataset=CropDataset
    model=json.loads((runner.OUT/'condition.json').read_text())['model']
    args=argparse.Namespace(model=model,steps=432,batch_size=4,lr=3e-5,validate_every=108,device=device)
    runner.train(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--condition',choices=['unet_v3_v4','unet_v3_v5','ragan_public_v3_v4','ragan_public_v3_v5'])
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        assert a.condition
        run(a.condition,a.device)
