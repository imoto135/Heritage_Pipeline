"""Matched v5 RA-GAN pilot using the active residual GA engineering candidate."""
from __future__ import annotations
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
OUT=ROOT/'tmp_work/makura_v5_residual_ga_20261021'
RUNNER=ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot/run_makura_supervised_pilot.py'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_v5_aligned_fade_control import crop_xy,trace_centers,OLD,SEED,sha,dump

def prepare():
    assert not OUT.exists(),OUT
    parent=BASE/'ragan_public_v3_v5';dest=OUT/'ragan_residual_ga_v5'
    dest.mkdir(parents=True)
    for name in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
        (dest/name).write_bytes((parent/name).read_bytes())
    dump(dest/'condition.json',dict(model='ragan',ga='paper Eq.9-12 candidate with residual I*(1+W) output',
        parent=str(parent.relative_to(ROOT)),same_train_RGB=True,same_labels=True,same_crops=True,
        same_validation=True,same_seed=True,steps=432,batch_size=4,formal_accuracy=False,official_reproduction=False))
    dump(OUT/'protocol.json',dict(condition='ragan_residual_ga_v5',parent=str(parent.relative_to(ROOT)),
        difference='GA module and its residual feature output; multiple internal formula differences remain',
        train_manifest_sha256=sha(dest/'train_manifest.jsonl'),val_manifest_sha256=sha(dest/'val_manifest.jsonl'),
        steps=432,batch_size=4,seed=SEED,formal_accuracy=False,official_reproduction=False))
    print('prepared residual GA v5 control',flush=True)

def prepare_nogan():
    parent=BASE/'ragan_public_v3_v5';dest=OUT/'ragan_residual_ga_nogan_v5'
    assert OUT.exists() and not dest.exists()
    dest.mkdir()
    for name in ('train_manifest.jsonl','val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
        (dest/name).write_bytes((parent/name).read_bytes())
    dump(dest/'condition.json',dict(model='ra_no_gan',ga='paper Eq.9-12 candidate with residual I*(1+W) output',
        parent=str(parent.relative_to(ROOT)),same_train_RGB=True,same_labels=True,same_crops=True,
        same_validation=True,same_seed=True,steps=432,batch_size=4,formal_accuracy=False,official_reproduction=False,
        purpose='generator-only control for adversarial training'))
    dump(OUT/'protocol_nogan.json',dict(condition=dest.name,parent=str(parent.relative_to(ROOT)),
        difference='Residual GA generator without discriminator',train_manifest_sha256=sha(dest/'train_manifest.jsonl'),
        val_manifest_sha256=sha(dest/'val_manifest.jsonl'),steps=432,batch_size=4,seed=SEED,
        formal_accuracy=False,official_reproduction=False))
    print('prepared residual GA generator-only v5 control',flush=True)

def run(device,condition):
    sys.path.insert(0,str(RUNNER.parent))
    import ragan_pilot_support as support
    support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
    spec=importlib.util.spec_from_file_location('fixed_pilot_v5_residual_ga',RUNNER)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner.ROOT=ROOT;runner.OUT=OUT/condition
    from ragan_paper_ga import candidate_generator
    runner.make_model=lambda kind:candidate_generator('residual_gate') if kind in ('ragan','ra_no_gan') else support.make_model(kind)
    centers=trace_centers()
    generic={r['patch']:r['selected_generic_xy'] for r in json.loads((OLD/'input_audit.json').read_text())['rows'] if r['strength']==.8}
    class CropDataset(runner.PilotDataset):
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
    model_kind=json.loads((runner.OUT/'condition.json').read_text())['model']
    args=argparse.Namespace(model=model_kind,steps=432,batch_size=4,lr=3e-5,validate_every=108,device=device)
    runner.train(args)
    torch.manual_seed(SEED)
    initial=candidate_generator('residual_gate')
    ckpt=runner.OUT/'models'/model_kind/'last.pt'
    final=torch.load(ckpt,map_location='cpu',weights_only=False)['model']
    ga=[name for name,p in initial.named_parameters() if name.startswith(('ga1.','ga2.','ga3.','ga4.'))]
    changed=[name for name,p in initial.named_parameters() if name in ga and not torch.equal(p,final[name])]
    dump(OUT/('attention_update_audit_nogan.json' if model_kind=='ra_no_gan' else 'attention_update_audit.json'),dict(attention_tensors=len(ga),changed_tensors=len(changed),
        all_changed=len(ga)==len(changed),checkpoint_sha256=sha(ckpt),
        note='Compared with generator initialized under the same seed; does not establish learned attention quality'))
    print('GA parameters changed',len(changed),'/',len(ga),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare_nogan','run']);p.add_argument('--device',default='cuda:0')
    p.add_argument('--condition',choices=['ragan_residual_ga_v5','ragan_residual_ga_nogan_v5'],default='ragan_residual_ga_v5')
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='prepare_nogan':prepare_nogan()
    else:run(a.device,a.condition)
