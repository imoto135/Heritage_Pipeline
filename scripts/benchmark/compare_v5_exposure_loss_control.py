"""Evaluate same-input exposure and core-loss controls against v5 baselines."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
OUT=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
CORES=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,save,coverage
from score_makura_heldout_thin_transfer import neural_mask
from ragan_pilot_support import make_model
from audit_targeted_thin_fade import montage
import apply_ragan_real_regions as real

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    models={}
    for stem,kind in (('unet_v3','unet'),('ragan_public_v3','ragan')):
        for factor in ('baseline','exposure','core_loss'):
            folder=(BASE/(stem+'_v5')) if factor=='baseline' else (OUT/(stem+'_'+factor))
            path=folder/'models'/kind/'last.pt';model=make_model(kind)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            models[(stem,factor)]=(model.to(device).eval(),kind,sha(path))
    rows=[];trainrows=[];realrows=[];figures=[]
    patch='200017197_00172_0_p00'
    source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/patch/'clean.png')
    trace=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png')).convert('L'))>0
    for alpha in (0.,.45,.70,.85):
        key=f'alpha_{alpha:.2f}';path=DIAG/'cases'/key/'input.png';inp=rgb(path)
        items=[('損傷前原本',source),('診断入力',inp),('Howe固定',np.where(mask(DIAG/'cases'/key/'howe_fixed.png'),0,255).astype(np.uint8))]
        for (stem,factor),(model,kind,cksha) in models.items():
            pred=neural_mask(model,kind,inp,device);dest=OUT/'evaluation'/'targeted'/key/(stem+'_'+factor+'.png')
            save(dest,pred)
            rows.append(dict(input=key,model=stem,factor=factor,checkpoint_sha256=cksha,
                input_sha256=sha(path),prediction_sha256=sha(dest),**coverage(pred,trace)))
            items.append((stem+' '+factor,np.where(pred,0,255).astype(np.uint8)))
        if alpha:
            for detail in (False,True):
                dest=OUT/'evaluation'/'figures'/(key+('_details' if detail else '_full')+'.png')
                montage(dest,'提示頻度・墨芯重み / 同一入力 '+key,items,detail)
                figures.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    for record in json.loads((CORES/'audit.json').read_text())['rows']:
        patch=record['patch'];source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/train'/patch/'clean.png')
        path=BASE/'inputs'/patch/'fade_0.80.png';inp=rgb(path)
        trace=np.asarray(Image.open(CORES/'masks'/(patch+'_trace.png')).convert('L'))>0
        core=np.asarray(Image.open(CORES/'masks'/(patch+'_core.png')).convert('L'))>0
        items=[('原本',source),('訓練入力 .80',inp)]
        for (stem,factor),(model,kind,cksha) in models.items():
            pred=neural_mask(model,kind,inp,device)
            trainrows.append(dict(patch=patch,model=stem,factor=factor,trace=coverage(pred,trace),
                core=coverage(pred,core),checkpoint_sha256=cksha,input_sha256=sha(path)))
            items.append((stem+' '+factor,np.where(pred,0,255).astype(np.uint8)))
        dest=OUT/'evaluation'/'figures'/(patch+'_train_0.80.png')
        montage(dest,'学習ページの強い退色 / '+patch,items,False)
        figures.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    cases=json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    cases=[c for c in cases if c['id'] in ('D01','D04')]
    spec=importlib.util.spec_from_file_location('source_scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scoring;spec.loader.exec_module(scoring)
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text());ann={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    for case in cases:
        page=rgb(ROOT/case['source']);assert sha(ROOT/case['source'])==case['source_sha256']
        x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        items=[('原画像',raw),('Howe固定',np.where(mask(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/case['id']/'howe_fixed.png'),0,255).astype(np.uint8))]
        for (stem,factor),(model,kind,cksha) in models.items():
            pred=real.infer_roi(model,kind,page,case['roi'],device)<.5
            dest=OUT/'evaluation'/'real'/case['id']/(stem+'_'+factor+'.png');save(dest,pred)
            items.append((stem+' '+factor,np.where(pred,0,255).astype(np.uint8)))
            for region in regions:
                if region['case']!=case['id']:continue
                labels,paths=scoring.raster(region,ann[region['id']]);rx,ry,rw,rh=region['roi']
                q=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
                realrows.append(dict(case=case['id'],region=region['id'],model=stem,factor=factor,
                    checkpoint_sha256=cksha,prediction_sha256=sha(dest),**scoring.measure(q,labels,paths)))
        dest=OUT/'evaluation'/'figures'/(case['id']+'_real.png')
        real.panel(dest,case['id']+' 実損傷 / 提示頻度・墨芯重み',items)
        figures.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    dump(OUT/'evaluation'/'results.json',dict(targeted=rows,train=trainrows,real=realrows,figures=figures,
        formal_accuracy=False,source_trace_used_to_construct_diagnostic=True,known_development_cases=True))
    for row in rows:
        if row['input'] in ('alpha_0.45','alpha_0.70'):
            print(row['input'],row['model'],row['factor'],row['r0'],'/',row['n'],'r2',row['r2'],flush=True)
    for row in realrows:print('REAL',row['region'],row['model'],row['factor'],row['ink_r2'],'/',row['ink_n'],flush=True)
    for row in trainrows:print('TRAIN',row['patch'],row['model'],row['factor'],row['trace']['r0'],'/',row['trace']['n'],flush=True)
if __name__=='__main__':main()
