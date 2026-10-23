"""Compare old and new matched pilots on targeted fade and known real ROIs."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/targeted_fade_training_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
OLD=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from score_makura_heldout_thin_transfer import get_model,neural_mask
import apply_ragan_real_regions as real
from ragan_pilot_support import make_model

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def mask(p):return np.asarray(Image.open(p).convert('L'))==0
def save(p,a):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.where(a,0,255).astype(np.uint8)).save(p)
def coverage(pred,points):
    y,x=np.where(points);dist=distance_transform_edt(~pred)
    return dict(n=len(x),r0=int(pred[y,x].sum()),r2=int((dist[y,x]<=2).sum()))

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.25,device)
    kinds={'unet_v3':'unet','ragan_public_v3':'ragan'}
    models={}
    for name,kind in kinds.items():
        old=get_model(name,device)
        new=make_model(kind);ck=OUT/name/'models'/kind/'last.pt'
        new.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);new.to(device).eval()
        models[name]={'old':old,'new':(new,kind,sha(ck))}
    source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val/200017197_00172_0_p00/clean.png')
    points=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review/200017197_00172_0_p00_high_centers.png'))>0
    rows=[];figures=[]
    for level in (0.,.45,.70,.85):
        key=f'alpha_{level:.2f}';inp=rgb(DIAG/'cases'/key/'input.png')
        outputs={}
        for name in kinds:
            for version in ('old','new'):
                model,kind,cksha=models[name][version]
                pred=neural_mask(model,kind,inp,device)
                outputs[(name,version)]=pred
                path=OUT/'evaluation'/'targeted'/key/(name+'_'+version+'.png');save(path,pred)
                rows.append(dict(input=key,model=name,version=version,checkpoint_sha256=cksha,
                    input_sha256=sha(DIAG/'cases'/key/'input.png'),mask_sha256=sha(path),**coverage(pred,points)))
        if level:
            items=[('損傷前原本',source),('診断入力',inp),
                   ('Howe固定',np.where(mask(DIAG/'cases'/key/'howe_fixed.png'),0,255).astype(np.uint8))]
            for name in kinds:
                for version in ('old','new'):
                    items.append((name+' '+version,np.where(outputs[(name,version)],0,255).astype(np.uint8)))
            path=OUT/'evaluation'/'figures'/(key+'_full.png')
            real.panel(path,'旧訓練と細線狙い訓練 / '+key,items)
            figures.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
            detail=OUT/'evaluation'/'figures'/(key+'_details.png')
            # Reuse the preselected crop boxes and geometry-preserving display.
            from audit_targeted_thin_fade import montage
            montage(detail,'旧訓練と細線狙い訓練 / '+key,items,True)
            figures.append(dict(path=str(detail.relative_to(OUT)),sha256=sha(detail)))
    oldreal=json.loads((OLD/'real_application/real_inference_verification.json').read_text())
    cases=json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    cases=[c for c in cases if c['id'] in ('D01','D04')]
    assert len(cases)==2
    spec=importlib.util.spec_from_file_location('scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scoring;spec.loader.exec_module(scoring)
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text())
    annotations={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    realrows=[]
    for case in cases:
        page=rgb(ROOT/case['source']);assert sha(ROOT/case['source'])==case['source_sha256']
        x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        items=[('原画像',raw),('Howe固定',np.where(mask(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/case['id']/'howe_fixed.png'),0,255).astype(np.uint8))]
        for name in kinds:
            for version in ('old','new'):
                if version=='old':
                    pred=mask(OLD/'real_application/predictions'/case['id']/(name+'.png'))
                    cksha=models[name]['old'][2]
                else:
                    model,kind,cksha=models[name]['new'];prob=real.infer_roi(model,kind,page,case['roi'],device);pred=prob<.5
                    dest=OUT/'evaluation'/'real'/case['id']/(name+'_new.png');save(dest,pred)
                items.append((name+' '+version,np.where(pred,0,255).astype(np.uint8)))
                for region in regions:
                    if region['case']!=case['id']:continue
                    labels,paths=scoring.raster(region,annotations[region['id']]);rx,ry,rw,rh=region['roi']
                    q=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
                    realrows.append(dict(case=case['id'],region=region['id'],model=name,version=version,
                        checkpoint_sha256=cksha,**scoring.measure(q,labels,paths)))
        path=OUT/'evaluation'/'figures'/(case['id']+'_real.png')
        real.panel(path,case['id']+' 実損傷（既知開発例）',items)
        figures.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    dump(OUT/'evaluation'/'results.json',dict(targeted=rows,real=realrows,figures=figures,
        baseline_real_rows_sha256=sha(OLD/'real_application/real_inference_verification.json'),
        formal_accuracy=False,source_trace_used_to_position_diagnostic=True,
        real_cases_development_known=True))
    for row in rows:
        if row['input'] in ('alpha_0.45','alpha_0.70'):
            print(row['input'],row['model'],row['version'],f"r0 {row['r0']}/49 r2 {row['r2']}/49",flush=True)
    for row in realrows:
        print('REAL',row['case'],row['region'],row['model'],row['version'],
              f"ink r2 {row['ink_r2']}/{row['ink_n']}",f"nonink FP {row['nonink_false']}/{row['nonink_n']}",flush=True)

if __name__=='__main__':main()
