"""Gate U-Net additions to Howe using only damaged-image local contrast."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT=EXP/'local_contrast_gate'
BASE=ROOT/'tmp_work/makura_v5_aligned_fade_control_20261021'
DIAG=ROOT/'tmp_work/targeted_thin_fade_20261021'
CORES=ROOT/'tmp_work/makura_additional_thin_cores_20261021'
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
sys.path.insert(0,str(ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,coverage,save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

def contrast(inp):
    paper=cv2.morphologyEx(inp,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    return cv2.cvtColor(paper,cv2.COLOR_RGB2GRAY).astype(np.int16)-cv2.cvtColor(inp,cv2.COLOR_RGB2GRAY).astype(np.int16)
def variants(howe,q,local):
    candidate=q>=.2;dist=ndi.distance_transform_edt(~howe)
    return {'howe':howe,'unet_020':candidate,'howe_plus_r3':howe|(candidate&(dist<=3)),
        'contrast16_r3':howe|(candidate&(dist<=3)&(local>=16)),
        'contrast20_r3':howe|(candidate&(dist<=3)&(local>=20)),
        'contrast16_r8':howe|(candidate&(dist<=8)&(local>=16))}

def main():
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25,device)
    path=EXP/'unet_v3_exposure/models/unet/last.pt';model=make_model('unet')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    model.to(device).eval();rows=[];figures=[]
    train=[]
    for record in json.loads((CORES/'audit.json').read_text())['rows']:
        patch=record['patch'];inp=rgb(BASE/'inputs'/patch/'fade_0.80.png')
        tr=np.asarray(Image.open(CORES/'masks'/(patch+'_trace.png')).convert('L'))>0
        loc=contrast(inp);q=ink_probability(model,'unet',inp,device)
        train.append(dict(patch=patch,n=int(tr.sum()),input_sha256=sha(BASE/'inputs'/patch/'fade_0.80.png'),
            visible_at_16=int(((loc>=16)&tr).sum()),visible_at_20=int(((loc>=20)&tr).sum()),
            unet_q020_at_16=int(((loc>=16)&(q>=.2)&tr).sum()),
            unet_q020_at_20=int(((loc>=20)&(q>=.2)&tr).sum())))
    valpatch='200017197_00172_0_p00'
    tr=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(valpatch+'_high_centers.png')).convert('L'))>0
    target=np.asarray(Image.open(ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/valpatch/'paper_target.png').convert('L'))
    valid=np.asarray(Image.open(ROOT/'tmp_work/makura_supervised_smoke_20261021/labels_v2'/valpatch/'valid.png').convert('L'))>0
    paper=valid&(target==255)
    for alpha in (.45,.70,.85):
        key=f'alpha_{alpha:.2f}';inp=rgb(DIAG/'cases'/key/'input.png')
        howe=mask(DIAG/'cases'/key/'howe_fixed.png');q=ink_probability(model,'unet',inp,device)
        outputs=variants(howe,q,contrast(inp))
        items=[('損傷前原本',rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/valpatch/'clean.png')),
            ('診断入力',inp)]
        for name,pred in outputs.items():
            dest=OUT/'synthetic'/key/(name+'.png');save(dest,pred)
            rows.append(dict(case=key,method=name,trace=coverage(pred,tr),paper_n=int(paper.sum()),
                paper_false=int((pred&paper).sum()),prediction_sha256=sha(dest)))
            items.append((name,np.where(pred,0,255).astype(np.uint8)))
        for detail in (False,True):
            dest=OUT/'figures'/(key+('_details' if detail else '_full')+'.png')
            montage(dest,'損傷入力の局所コントラスト / '+key,items,detail)
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
        outputs=variants(howe,q,contrast(page)[y:y+h,x:x+w]);items=[('原画像',raw)]
        for name,pred in outputs.items():
            dest=OUT/'real'/case['id']/(name+'.png');save(dest,pred)
            items.append((name,np.where(pred,0,255).astype(np.uint8)))
            for region in regions:
                if region['case']!=case['id']:continue
                labels,paths=scoring.raster(region,ann[region['id']]);rx,ry,rw,rh=region['roi']
                m=scoring.measure(pred[ry-y:ry-y+rh,rx-x:rx-x+rw],labels,paths)
                rows.append(dict(case=case['id'],method=name,region=region['id'],prediction_sha256=sha(dest),**m))
        dest=OUT/'figures'/(case['id']+'_real.png')
        real.panel(dest,case['id']+' 実画像 / 局所コントラスト制限',items)
        figures.append(dict(path=str(dest.relative_to(OUT)),sha256=sha(dest)))
    dump(OUT/'results.json',dict(rows=rows,train_trace=train,figures=figures,
        unet_checkpoint_sha256=sha(path),candidate_threshold=.2,
        local_feature='cv2 morphology close 41x41 on damaged RGB, grayscale contrast',
        contrast_thresholds=[16,20],howe_distance=[3,8],formal_accuracy=False,
        same_development_cases_inspected_before_threshold_choice=True))
    for row in rows:
        if row['case']=='alpha_0.70':print(row['case'],row['method'],row['trace']['r0'],'/',row['trace']['n'],'paper false',row['paper_false'],flush=True)
        if row['case']=='D01':print('D01',row['method'],'r2',row['ink_r2'],'/',row['ink_n'],flush=True)
    print('TRAIN',[(r['patch'],r['visible_at_16'],r['n']) for r in train],flush=True)
if __name__=='__main__':main()
