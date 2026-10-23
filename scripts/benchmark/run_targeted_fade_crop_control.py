"""Matched old/new fade input training with identical trace/anchor-aware crops.

Only the synthetic fade RGB differs between paired arms. Crop selection uses
training source traces or automatically selected training skeleton anchors.
No validation source trace enters training or crop selection.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
PREV=ROOT/'tmp_work/targeted_fade_training_20261021'
OLD=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021'
OUT=ROOT/'tmp_work/targeted_fade_crop_control_20261021'
RUNNER=ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot/run_makura_supervised_pilot.py'
SEED=20261021

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def crop_xy(idx,epoch,row,centers):
    patch=row['patch_id']
    positions=centers[patch]
    if patch in ('200017197_00103_1_p00','200017197_00214_1_p00'):
        # Midpoints of training-only source traces, fixed before this paired run.
        x,y={'200017197_00103_1_p00':(126,390),'200017197_00214_1_p00':(142,60)}[patch]
    else:
        x,y=positions[(idx+epoch)%len(positions)]
    rng=np.random.default_rng(SEED+epoch*100000+idx)
    dx,dy=map(int,rng.integers(-24,25,size=2))
    return int(np.clip(x-128+dx,0,256)),int(np.clip(y-128+dy,0,256))

def prepare():
    assert not OUT.exists(),f'Preserve existing experiment: {OUT}'
    audit=json.loads((PREV/'input_audit.json').read_text())['rows']
    centers={r['patch']:r['selected_generic_xy'] for r in audit if r['strength']==.8}
    assert len(centers)==96 and all(centers.values())
    OUT.mkdir(parents=True)
    for model,cond in [('unet','unet_v3'),('ragan','ragan_public_v3')]:
        for arm in ('original_fade','targeted_fade'):
            src=(OLD if arm=='original_fade' else PREV)/cond;dst=OUT/(cond+'_'+arm)
            dst.mkdir()
            for name in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
                shutil.copy2(src/name,dst/name)
            dump(dst/'condition.json',dict(model=model,arm=arm,source_condition=cond,
                crop_policy='same training anchor/trace-aware crop in both arms; fixed center val crop',
                steps=432,batch_size=4,seed=SEED,formal_accuracy=False))
    baseline=[json.loads(x) for x in (OUT/'unet_v3_original_fade/train_manifest.jsonl').read_text().splitlines()]
    targeted=[json.loads(x) for x in (OUT/'unet_v3_targeted_fade/train_manifest.jsonl').read_text().splitlines()]
    assert len(baseline)==len(targeted)==864
    assert all(a['id']==b['id'] and a['labels']==b['labels'] for a,b in zip(baseline,targeted))
    assert sum(a['input_sha256']!=b['input_sha256'] for a,b in zip(baseline,targeted))==288
    target_audit=[]
    for idx,(a,b) in enumerate(zip(baseline,targeted)):
        if b['damage_family']!='targeted_fade':continue
        for epoch in (0,1):
            x,y=crop_xy(idx,epoch,b,centers)
            target_audit.append(dict(index=idx,epoch=epoch,patch=b['patch_id'],strength=b['damage_strength'],crop_xy=[x,y]))
    dump(OUT/'crop_schedule.json',dict(rows=target_audit))
    dump(OUT/'protocol.json',dict(arms=('original_fade','targeted_fade'),
        models=('unet_v3','ragan_public_v3'),changed_train_rgb_rows=288,
        target_teacher='same v3 provisional labels',validation='same fixed v2 provisional labels',
        crop='Identical source thin-anchor crops in paired arms for 288 fade rows; other rows retain frozen random crop',
        training_source_traces=('200017197_00103_1_p00','200017197_00214_1_p00'),
        validation_source_trace_used_in_training=False,seed=SEED,steps=432,batch_size=4,
        formal_accuracy=False,dkds_reproduction=False))
    print('Prepared four paired conditions,',len(target_audit),'scheduled fade crops',flush=True)

def run(condition,device):
    sys.path.insert(0,str(RUNNER.parent))
    import ragan_pilot_support as support
    support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
    spec=importlib.util.spec_from_file_location('fixed_pilot_crop',RUNNER)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner.ROOT=ROOT;runner.OUT=OUT/condition
    base=runner.PilotDataset
    audit=json.loads((PREV/'input_audit.json').read_text())['rows']
    centers={r['patch']:r['selected_generic_xy'] for r in audit if r['strength']==.8}
    class CropDataset(base):
        def __getitem__(self,idx):
            row=self.records[idx]
            if self.split!='train' or row['damage_family'] not in ('fade','targeted_fade'):
                return super().__getitem__(idx)
            x,y=crop_xy(idx,self.epoch,row,centers)
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
    p.add_argument('--condition',choices=['unet_v3_original_fade','unet_v3_targeted_fade','ragan_public_v3_original_fade','ragan_public_v3_targeted_fade'])
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        assert a.condition
        run(a.condition,a.device)
