"""Evaluate v3 center-only versus v4 source-core provisional teachers."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/makura_teacher_v4_width_20261021'
BASE=ROOT/'tmp_work/targeted_fade_crop_control_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from score_makura_heldout_thin_transfer import neural_mask
from ragan_pilot_support import make_model
from audit_targeted_thin_fade import montage
import apply_ragan_real_regions as real

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def mask(p):return np.asarray(Image.open(p).convert('L'))==0
def save(p,pred):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.where(pred,0,255).astype(np.uint8)).save(p)
def centers(pred,trace):
    y,x=np.where(trace);dist=distance_transform_edt(~pred)
    return dict(n=len(x),r0=int(pred[y,x].sum()),r2=int((dist[y,x]<=2).sum()))

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction(.25,device)
    models={}
    kinds={'unet_v3':'unet','ragan_public_v3':'ragan'}
    for name,kind in kinds.items():
        for teacher in ('v3','v4'):
            base=(BASE/(name+'_targeted_fade')) if teacher=='v3' else (OUT/(name+'_v4teacher'))
            ck=base/'models'/kind/'last.pt'
            model=make_model(kind);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']);model.to(device).eval()
            models[(name,teacher)]=(model,kind,sha(ck))
    valpatch='200017197_00172_0_p00'
    source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/valpatch/'clean.png')
    trace=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(valpatch+'_high_centers.png')))>0
    rows=[];trainrows=[];realrows=[];figures=[]
    for alpha in (0.,.45,.70,.85):
        key=f'alpha_{alpha:.2f}';inp=rgb(DIAG/'cases'/key/'input.png');preds={}
        for (name,teacher),(model,kind,cksha) in models.items():
            pred=neural_mask(model,kind,inp,device);preds[(name,teacher)]=pred
            p=OUT/'evaluation'/'targeted'/key/(name+'_'+teacher+'.png');save(p,pred)
            rows.append(dict(input=key,model=name,teacher=teacher,checkpoint_sha256=cksha,
                input_sha256=sha(DIAG/'cases'/key/'input.png'),prediction_sha256=sha(p),**centers(pred,trace)))
        if alpha:
            items=[('損傷前原本',source),('診断入力',inp),
                ('Howe固定',np.where(mask(DIAG/'cases'/key/'howe_fixed.png'),0,255).astype(np.uint8))]
            for name in kinds:
                for teacher in ('v3','v4'):
                    items.append((name+' 教師'+teacher,np.where(preds[(name,teacher)],0,255).astype(np.uint8)))
            for detail in (False,True):
                p=OUT/'evaluation'/'figures'/(key+('_details' if detail else '_full')+'.png')
                montage(p,'中心教師v3 と 墨芯教師v4 / '+key,items,detail)
                figures.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p)))
    for patch in ('200017197_00103_1_p00','200017197_00214_1_p00'):
        source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/train'/patch/'clean.png')
        inp_path=ROOT/'tmp_work/targeted_fade_training_20261021/inputs'/patch/'fade_0.80.png';inp=rgb(inp_path)
        trace=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(patch+'_high_centers.png')))>0
        core=np.asarray(Image.open(ROOT/'tmp_work/makura_thin_width_candidate_20261021/masks'/(patch+'_core.png')))>0
        items=[('原本',source),('学習入力 .80',inp)]
        for (name,teacher),(model,kind,cksha) in models.items():
            pred=neural_mask(model,kind,inp,device)
            trainrows.append(dict(patch=patch,model=name,teacher=teacher,checkpoint_sha256=cksha,
                input_sha256=sha(inp_path),trace=centers(pred,trace),core=centers(pred,core)))
            items.append((name+' 教師'+teacher,np.where(pred,0,255).astype(np.uint8)))
        p=OUT/'evaluation'/'figures'/(patch+'_train_0.80.png');montage(p,'学習済み薄線の強い退色 / '+patch,items,False)
        figures.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p)))
    case_defs=json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    case_defs=[c for c in case_defs if c['id'] in ('D01','D04')]
    spec=importlib.util.spec_from_file_location('source_scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scoring;spec.loader.exec_module(scoring)
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text())
    ann={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    for case in case_defs:
        page=rgb(ROOT/case['source']);assert sha(ROOT/case['source'])==case['source_sha256']
        x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        items=[('原画像',raw),('Howe固定',np.where(mask(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/case['id']/'howe_fixed.png'),0,255).astype(np.uint8))]
        for (name,teacher),(model,kind,cksha) in models.items():
            pred=real.infer_roi(model,kind,page,case['roi'],device)<.5
            p=OUT/'evaluation'/'real'/case['id']/(name+'_'+teacher+'.png');save(p,pred)
            items.append((name+' 教師'+teacher,np.where(pred,0,255).astype(np.uint8)))
            for region in regions:
                if region['case']!=case['id']:continue
                labels,paths=scoring.raster(region,ann[region['id']]);rx,ry,rw,rh=region['roi']
                q=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
                realrows.append(dict(case=case['id'],region=region['id'],model=name,teacher=teacher,
                    checkpoint_sha256=cksha,prediction_sha256=sha(p),**scoring.measure(q,labels,paths)))
        p=OUT/'evaluation'/'figures'/(case['id']+'_real.png');real.panel(p,case['id']+' 実損傷 / 教師v3とv4',items)
        figures.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p)))
    dump(OUT/'evaluation'/'results.json',dict(targeted=rows,train=trainrows,real=realrows,figures=figures,
        formal_accuracy=False,source_trace_used_to_construct_diagnostic=True,known_development_cases=True))
    for row in rows:
        if row['input'] in ('alpha_0.45','alpha_0.70'):
            print(row['input'],row['model'],row['teacher'],row['r0'],'/',row['n'],'r2',row['r2'],flush=True)
    for row in realrows:
        print('REAL',row['region'],row['model'],row['teacher'],row['ink_r2'],'/',row['ink_n'],flush=True)
    for row in trainrows:
        print('TRAIN',row['patch'],row['model'],row['teacher'],row['trace']['r0'],'/',row['trace']['n'],flush=True)

if __name__=='__main__':main()
