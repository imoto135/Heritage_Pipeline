"""Evaluate paired source-targeted fade training with identical crop schedules."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/targeted_fade_crop_control_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from score_makura_heldout_thin_transfer import neural_mask
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real
from audit_targeted_thin_fade import montage

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def mask(p):return np.asarray(Image.open(p).convert('L'))==0
def save(p,a):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.where(a,0,255).astype(np.uint8)).save(p)

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.25,device)
    model_kinds={'unet_v3':'unet','ragan_public_v3':'ragan'}
    models={}
    for name,kind in model_kinds.items():
        for arm in ('original_fade','targeted_fade'):
            ck=OUT/(name+'_'+arm)/'models'/kind/'last.pt'
            model=make_model(kind);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model'])
            model.to(device).eval();models[(name,arm)]=(model,kind,sha(ck))
    source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val/200017197_00172_0_p00/clean.png')
    points=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review/200017197_00172_0_p00_high_centers.png'))>0
    yy,xx=np.where(points);rows=[];figures=[]
    for alpha in (0.,.45,.70,.85):
        key=f'alpha_{alpha:.2f}';inp=rgb(DIAG/'cases'/key/'input.png')
        outputs={}
        for (name,arm),(model,kind,cksha) in models.items():
            pred=neural_mask(model,kind,inp,device);outputs[(name,arm)]=pred
            path=OUT/'evaluation'/'targeted'/key/(name+'_'+arm+'.png');save(path,pred)
            dist=distance_transform_edt(~pred)
            rows.append(dict(input=key,model=name,arm=arm,checkpoint_sha256=cksha,
                input_sha256=sha(DIAG/'cases'/key/'input.png'),prediction_sha256=sha(path),n=len(xx),
                r0=int(pred[yy,xx].sum()),r2=int((dist[yy,xx]<=2).sum())))
        if alpha:
            items=[('損傷前原本',source),('診断入力',inp),
                ('Howe固定',np.where(mask(DIAG/'cases'/key/'howe_fixed.png'),0,255).astype(np.uint8))]
            for name in model_kinds:
                for arm in ('original_fade','targeted_fade'):
                    items.append((name+' '+arm,np.where(outputs[(name,arm)],0,255).astype(np.uint8)))
            for detail in (False,True):
                path=OUT/'evaluation'/'figures'/(key+('_details' if detail else '_full')+'.png')
                montage(path,'切出し統一: 従来退色と薄線狙い退色 / '+key,items,detail)
                figures.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    trainrows=[]
    for patch in ('200017197_00103_1_p00','200017197_00214_1_p00'):
        raw=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/train'/patch/'clean.png')
        inp_path=ROOT/'tmp_work/targeted_fade_training_20261021/inputs'/patch/'fade_0.80.png'
        inp=rgb(inp_path)
        traced=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png')))>0
        py,px=np.where(traced)
        items=[('原本',raw),('訓練入力: 強度0.80',inp)]
        for (name,arm),(model,kind,cksha) in models.items():
            pred=neural_mask(model,kind,inp,device);dist=distance_transform_edt(~pred)
            trainrows.append(dict(patch=patch,model=name,arm=arm,n=len(px),r0=int(pred[py,px].sum()),
                r2=int((dist[py,px]<=2).sum()),input_sha256=sha(inp_path),checkpoint_sha256=cksha))
            items.append((name+' '+arm,np.where(pred,0,255).astype(np.uint8)))
        path=OUT/'evaluation'/'figures'/(patch+'_train_0.80.png')
        montage(path,'学習済み画像の線中心 / '+patch,items,False)
        figures.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    case_defs=json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    case_defs=[c for c in case_defs if c['id'] in ('D01','D04')];assert len(case_defs)==2
    spec=importlib.util.spec_from_file_location('scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scoring;spec.loader.exec_module(scoring)
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text())
    annotations={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    realrows=[]
    for case in case_defs:
        page=rgb(ROOT/case['source']);assert sha(ROOT/case['source'])==case['source_sha256']
        x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        items=[('原画像',raw),('Howe固定',np.where(mask(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/case['id']/'howe_fixed.png'),0,255).astype(np.uint8))]
        for (name,arm),(model,kind,cksha) in models.items():
            pred=real.infer_roi(model,kind,page,case['roi'],device)<.5
            p=OUT/'evaluation'/'real'/case['id']/(name+'_'+arm+'.png');save(p,pred)
            items.append((name+' '+arm,np.where(pred,0,255).astype(np.uint8)))
            for region in regions:
                if region['case']!=case['id']:continue
                labels,paths=scoring.raster(region,annotations[region['id']]);rx,ry,rw,rh=region['roi']
                q=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
                realrows.append(dict(case=case['id'],region=region['id'],model=name,arm=arm,
                    checkpoint_sha256=cksha,prediction_sha256=sha(p),**scoring.measure(q,labels,paths)))
        path=OUT/'evaluation'/'figures'/(case['id']+'_real.png');real.panel(path,case['id']+' 実損傷（開発例）',items)
        figures.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    dump(OUT/'evaluation'/'results.json',dict(targeted=rows,train=trainrows,real=realrows,figures=figures,
        formal_accuracy=False,known_development_cases=True,source_trace_used_to_construct_diagnostic=True))
    for row in rows:
        if row['input'] in ('alpha_0.45','alpha_0.70'):
            print(row['input'],row['model'],row['arm'],f"r0 {row['r0']}/{row['n']} r2 {row['r2']}/{row['n']}",flush=True)
    for row in realrows:
        print('REAL',row['region'],row['model'],row['arm'],f"ink r2 {row['ink_r2']}/{row['ink_n']}",flush=True)
    for row in trainrows:
        print('TRAIN',row['patch'],row['model'],row['arm'],f"r0 {row['r0']}/{row['n']} r2 {row['r2']}/{row['n']}",flush=True)

if __name__=='__main__':main()
