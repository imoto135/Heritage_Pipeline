"""Test whether thresholding explains missing traced ink, with paper-error controls."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
EXP=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT=EXP/'probability_audit'
CORES=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
THRESHOLDS=(.50,.40,.30,.20,.10,.05,.02)
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,coverage
from ragan_pilot_support import make_model,predict,auxiliaries
from audit_targeted_thin_fade import montage
import apply_ragan_real_regions as real

def gray(p):return np.asarray(Image.open(p).convert('L'))
def ink_probability(model,kind,inp,device):
    a=auxiliaries(inp)
    x=torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    with torch.inference_mode():return 1-predict(model,kind,x)[0,0].cpu().numpy()
def stats(q,trace):
    vals=q[trace]
    return dict(n=int(len(vals)),min=float(vals.min()),median=float(np.median(vals)),
        q10=float(np.quantile(vals,.10)),q90=float(np.quantile(vals,.90)),max=float(vals.max()))
def threshold_rows(q,trace,paper):
    return [dict(threshold=th,trace=coverage(q>=th,trace),
        paper_n=int(paper.sum()),paper_false=int(((q>=th)&paper).sum())) for th in THRESHOLDS]

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    models={}
    for stem,kind in (('unet_v3','unet'),('ragan_public_v3','ragan')):
        for factor in ('baseline','exposure','core_loss'):
            folder=BASE/(stem+'_v5') if factor=='baseline' else EXP/(stem+'_'+factor)
            path=folder/'models'/kind/'last.pt';model=make_model(kind)
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
            models[(stem,factor)]=(model.to(device).eval(),kind,sha(path))
    valpatch='200017197_00172_0_p00'
    valrec=next(json.loads(line) for line in (BASE/'unet_v3_v5/val_manifest.jsonl').read_text().splitlines() if valpatch in line)
    valtrace=gray(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(valpatch+'_high_centers.png'))>0
    vt=gray(ROOT/valrec['labels']['target']);vv=gray(ROOT/valrec['labels']['valid'])>0
    valpaper=vv&(vt==255)
    synthetic=[];figure=[]
    for alpha in (0.,.45,.70,.85):
        key=f'alpha_{alpha:.2f}';path=DIAG/'cases'/key/'input.png';inp=rgb(path)
        probabilities={}
        for (stem,factor),(model,kind,cksha) in models.items():
            q=ink_probability(model,kind,inp,device);probabilities[(stem,factor)]=q
            synthetic.append(dict(case=key,model=stem,factor=factor,checkpoint_sha256=cksha,
                input_sha256=sha(path),trace_probability=stats(q,valtrace),
                thresholds=threshold_rows(q,valtrace,valpaper)))
        if alpha in (.45,.70):
            items=[('損傷前原本',rgb(ROOT/valrec['clean'])),('診断入力',inp),
                   ('Howe固定',np.where(mask(DIAG/'cases'/key/'howe_fixed.png'),0,255).astype(np.uint8))]
            for stem,factor in (('unet_v3','exposure'),('ragan_public_v3','core_loss')):
                q=probabilities[(stem,factor)]
                for th in (.5,.2,.05):items.append((stem+' '+factor+' th='+str(th),np.where(q>=th,0,255).astype(np.uint8)))
            for detail in (False,True):
                dest=OUT/'figures'/(key+('_details' if detail else '_full')+'.png')
                montage(dest,'出力確率の閾値 / '+key,items,detail)
                figure.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    train=[]
    for record in json.loads((CORES/'audit.json').read_text())['rows']:
        patch=record['patch'];path=BASE/'inputs'/patch/'fade_0.80.png';inp=rgb(path)
        trace=gray(CORES/'masks'/(patch+'_trace.png'))>0
        rec=next(json.loads(line) for line in (BASE/'unet_v3_v5/train_manifest.jsonl').read_text().splitlines() if '"patch_id": "'+patch+'"' in line)
        target=gray(ROOT/rec['labels']['target']);valid=gray(ROOT/rec['labels']['valid'])>0
        paper=valid&(target==255)
        for (stem,factor),(model,kind,cksha) in models.items():
            q=ink_probability(model,kind,inp,device)
            train.append(dict(patch=patch,model=stem,factor=factor,checkpoint_sha256=cksha,
                input_sha256=sha(path),trace_probability=stats(q,trace),
                thresholds=threshold_rows(q,trace,paper)))
    spec=importlib.util.spec_from_file_location('source_scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scoring;spec.loader.exec_module(scoring)
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text());ann={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    cases=json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    realrows=[]
    for case in [c for c in cases if c['id'] in ('D01','D04')]:
        page=rgb(ROOT/case['source']);x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        items=[('原画像',raw),('Howe固定',np.where(mask(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/case['id']/'howe_fixed.png'),0,255).astype(np.uint8))]
        for (stem,factor),(model,kind,cksha) in models.items():
            q=1-real.infer_roi(model,kind,page,case['roi'],device)
            for region in regions:
                if region['case']!=case['id']:continue
                labels,paths=scoring.raster(region,ann[region['id']]);rx,ry,rw,rh=region['roi']
                qq=q[ry-y:ry-y+rh,rx-x:rx-x+rw]
                for th in THRESHOLDS:
                    realrows.append(dict(case=case['id'],region=region['id'],model=stem,factor=factor,
                        threshold=th,checkpoint_sha256=cksha,**scoring.measure(qq>=th,labels,paths)))
            if (stem,factor) in (('unet_v3','exposure'),('ragan_public_v3','core_loss')):
                for th in (.5,.2,.05):items.append((stem+' '+factor+' th='+str(th),np.where(q>=th,0,255).astype(np.uint8)))
        dest=OUT/'figures'/(case['id']+'_thresholds.png')
        real.panel(dest,case['id']+' 閾値と背景残存',items)
        figure.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    dump(OUT/'results.json',dict(synthetic=synthetic,train=train,real=realrows,figures=figure,
        thresholds=THRESHOLDS,formal_accuracy=False,validation_paper_is_provisional=True,
        D01_source_only_partial_labels=True,D04_visual_only=True))
    for row in synthetic:
        if row['case']=='alpha_0.70':
            print(row['case'],row['model'],row['factor'],'median ink probability',round(row['trace_probability']['median'],4),
                [(r['threshold'],r['trace']['r0'],r['paper_false']) for r in row['thresholds']],flush=True)
    for row in realrows:
        if row['region']=='thin' and row['threshold'] in (.5,.2,.05):
            print('D01',row['model'],row['factor'],row['threshold'],'ink r2',row['ink_r2'],'/',row['ink_n'],
                'paper false',row['nonink_false'],'/',row['nonink_n'],flush=True)
if __name__=='__main__':main()
