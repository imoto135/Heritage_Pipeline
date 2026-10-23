"""Fixed-rule Howe/U-Net fusion diagnostic on known thin-stroke and spot cases."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT=EXP/'howe_unet_fusion'
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,coverage,save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

def variants(howe,q):
    candidate=q>=.2
    dist=ndi.distance_transform_edt(~howe)
    label,n=ndi.label(candidate,np.ones((3,3),bool))
    hits=np.unique(label[candidate&(dist<=2)])
    hits=hits[hits>0]
    supported=np.isin(label,hits)&candidate
    return {'howe':howe,'unet_020':candidate,
        'howe_plus_r3':howe|(candidate&(dist<=3)),
        'howe_plus_r8':howe|(candidate&(dist<=8)),
        'howe_plus_connected':howe|supported}

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    path=EXP/'unet_v3_exposure/models/unet/last.pt';model=make_model('unet')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    model.to(device).eval();rows=[];figures=[]
    valpatch='200017197_00172_0_p00'
    trace=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(valpatch+'_high_centers.png')).convert('L'))>0
    target=np.asarray(Image.open(ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/valpatch/'paper_target.png').convert('L'))
    valid=np.asarray(Image.open(ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/valpatch/'valid.png').convert('L'))>0
    paper=valid&(target==255)
    for alpha in (.45,.70,.85):
        key=f'alpha_{alpha:.2f}';inp=rgb(DIAG/'cases'/key/'input.png')
        howe=mask(DIAG/'cases'/key/'howe_fixed.png');q=ink_probability(model,'unet',inp,device)
        results=variants(howe,q)
        items=[('損傷前原本',rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/valpatch/'clean.png')),
            ('診断入力',inp)]
        for name,pred in results.items():
            dest=OUT/'synthetic'/key/(name+'.png');save(dest,pred)
            rows.append(dict(case=key,method=name,trace=coverage(pred,trace),paper_n=int(paper.sum()),
                paper_false=int((pred&paper).sum()),ink_pixels=int(pred.sum()),prediction_sha256=sha(dest)))
            items.append((name,np.where(pred,0,255).astype(np.uint8)))
        for detail in (False,True):
            dest=OUT/'figures'/(key+('_details' if detail else '_full')+'.png')
            montage(dest,'固定規則のHowe + U-Net / '+key,items,detail)
            figures.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    spec=importlib.util.spec_from_file_location('source_scoring',ROOT/'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring=importlib.util.module_from_spec(spec);sys.modules[spec.name]=scoring;spec.loader.exec_module(scoring)
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text());ann={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    cases=json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    for case in [c for c in cases if c['id'] in ('D01','D04')]:
        page=rgb(ROOT/case['source']);x,y,w,h=case['roi'];raw=page[y:y+h,x:x+w]
        howe=mask(ROOT/'tmp_work/classical_limit_audit_20261021/crops'/case['id']/'howe_fixed.png')
        q=1-real.infer_roi(model,'unet',page,case['roi'],device)
        results=variants(howe,q);items=[('原画像',raw)]
        for name,pred in results.items():
            dest=OUT/'real'/case['id']/(name+'.png');save(dest,pred)
            items.append((name,np.where(pred,0,255).astype(np.uint8)))
            for region in regions:
                if region['case']!=case['id']:continue
                labels,paths=scoring.raster(region,ann[region['id']]);rx,ry,rw,rh=region['roi']
                mm=scoring.measure(pred[ry-y:ry-y+rh,rx-x:rx-x+rw],labels,paths)
                rows.append(dict(case=case['id'],method=name,region=region['id'],prediction_sha256=sha(dest),**mm))
        dest=OUT/'figures'/(case['id']+'_real.png')
        real.panel(dest,case['id']+' 実画像 / Howe＋U-Net固定規則',items)
        figures.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    dump(OUT/'results.json',dict(rows=rows,figures=figures,unet_checkpoint_sha256=sha(path),
        candidate_threshold=.2,support_rule='candidate pixels within 3/8 Euclidean pixels of Howe or whole 8-connected components that touch within 2 pixels',
        formal_accuracy=False,known_development_cases=True,parameters_selected_for_diagnostic_not_blind=True))
    for row in rows:
        if row['case']=='alpha_0.70':print(row['case'],row['method'],row['trace']['r0'],'/',row['trace']['n'],'paper false',row['paper_false'],flush=True)
        if row['case']=='D01':print('D01',row['method'],'r2',row['ink_r2'],'/',row['ink_n'],'nonink false',row['nonink_false'],flush=True)
if __name__=='__main__':main()
