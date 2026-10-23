"""Fixed readout across 2/4/6/8/10 epochs; same-run 432 checkpoint is reference."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_convergence_audit import OUT
from run_contact_stain_training_control import OUT as PARENT
from evaluate_ridge_gate_independent_val import proxies
from evaluate_contact_stain_counterfactual import make_field
from compare_v5_aligned_fade_control import sha,dump,rgb,mask,save,coverage
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from ragan_pilot_support import make_model
import evaluate_classical_partial_labels as scoring
import apply_ragan_real_regions as real

STEPS=[432,864,1296,1728,2160]


def freeze():
    path=OUT/'evaluation_protocol.json'
    assert not path.exists()
    dump(path,dict(checkpoints=STEPS,conditions=['ragan_original','ragan_contact'],threshold=.5,
        primary_comparison='2160 versus 432 checkpoint within each new run; not prior short-run weights',
        synthetic_patches=json.loads((PARENT/'protocol.json').read_text())['heldout_patches'],
        synthetic_families=['contact_stain','clean','uniform_fade_070'],
        real_cases=['D01','D03','D04','T07','C01'],
        known_traced_fade_patch='200017197_00172_0_p00',known_traced_fade_alpha=.70,
        fixed_last_checkpoint_primary=True,all_checkpoints_reported=True,
        prior_exposure=True,provisional_masks=True,formal_accuracy=False))


def main(nogan=False,unet=False):
    assert not (nogan and unet)
    prefix='unet_' if unet else ('nogan_' if nogan else '')
    protocol_path=OUT/(prefix+'evaluation_protocol.json')
    protocol=json.loads(protocol_path.read_text())
    dest=OUT/(prefix+'evaluation');assert not (dest/'results.json').exists()
    kind=protocol.get('prediction_kind','ragan')
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3,device)
    cases=[]
    for patch in protocol['synthetic_patches']:
        source=rgb(ROOT/'tmp_work/makura_supervised_pilot_20261021/patches/val'/patch/'clean.png')
        paper,core,remote,_=proxies(source);field,_=make_field(core)
        image=rgb(PARENT/'evaluation/inputs'/f'{patch}_stain.png')
        fade=np.rint(source.astype(np.float32)*.3+paper.astype(np.float32)*.7).astype(np.uint8)
        for family,inp in [('contact_stain',image),('clean',source),('uniform_fade_070',fade)]:
            affected=field>=.5 if family=='contact_stain' else np.ones(core.shape,bool)
            cases.append(dict(id=patch+'_'+family,patch=patch,family=family,input=inp,source=source,
                core=core&affected,paper=remote&affected))
    diag=ROOT/'tmp_work/targeted_thin_fade_20261021'
    tpatch=protocol['known_traced_fade_patch']
    from PIL import Image
    trace=np.asarray(Image.open(ROOT/'tmp_work/makura_damage_challenge_20261021/source_thin_review'/(tpatch+'_high_centers.png')).convert('L'))>0
    thin_input=rgb(diag/'cases/alpha_0.70/input.png')
    classic={c['id']:c for c in json.loads((ROOT/'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']}
    real_cases=[dict(id=k,source=classic[k]['source'],source_sha256=classic[k]['source_sha256'],roi=classic[k]['roi'])
                for k in ('D01','D03','D04')]
    real_cases.extend(json.loads((PARENT/'other_book_transfer/protocol.json').read_text())['cases'])
    part=ROOT/'tmp_work/classical_partial_labels_20261021'
    regions=json.loads((part/'source_regions.json').read_text())
    ann={r['id']:r for r in json.loads((part/'annotations_source_only.json').read_text())['regions']}
    real_sources={c['id']:rgb(ROOT/c['source']) for c in real_cases}
    for c in real_cases:assert sha(ROOT/c['source'])==c['source_sha256']
    rows=[];realrows=[];thinrows=[];figures=[]
    for condition in protocol['conditions']:
        for step in protocol['checkpoints']:
            checkpoint=OUT/condition/'models'/protocol.get('checkpoint_model_folder','ragan')/f'step_{step:05d}.pt'
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert payload['step']==step
            model=make_model(kind);model.load_state_dict(payload['model']);model.to(device).eval()
            cksha=sha(checkpoint)
            for case in cases:
                q=ink_probability(model,kind,case['input'],device);pred=q>=.5
                r2=ndi.distance_transform_edt(~pred)<=2
                path=dest/'synthetic_masks'/condition/f'{step:05d}'/(case['id']+'.png');save(path,pred)
                rows.append(dict(condition=condition,step=step,patch=case['patch'],family=case['family'],
                    core_n=int(case['core'].sum()),core_exact=int((pred&case['core']).sum()),core_r2=int((r2&case['core']).sum()),
                    paper_n=int(case['paper'].sum()),paper_ink=int((pred&case['paper']).sum()),
                    checkpoint_sha256=cksha,prediction_sha256=sha(path)))
            pred=ink_probability(model,kind,thin_input,device)>=.5
            thinrows.append(dict(condition=condition,step=step,**coverage(pred,trace)))
            save(dest/'traced_fade'/condition/f'{step:05d}.png',pred)
            for case in real_cases:
                key=case['id'];box=case.get('roi',case.get('crop_xywh'));x,y,w,h=box
                page=real_sources[key];pred=real.infer_roi(model,kind,page,box,device)<.5
                path=dest/'real_masks'/condition/f'{step:05d}'/(key+'.png');save(path,pred)
                if key in ('D01','D03'):
                    for reg in [r for r in regions if r['case']==key]:
                        rx,ry,rw,rh=reg['roi'];lab,paths=scoring.raster(reg,ann[reg['id']])
                        m=pred[ry-y:ry-y+rh,rx-x:rx-x+rw]
                        realrows.append(dict(condition=condition,step=step,case=key,region=reg['id'],**scoring.measure(m,lab,paths)))
                else:
                    if key=='D04':rbox=json.loads((PARENT/'d04_spot_control/protocol.json').read_text())['spot_box_xywh']
                    else:rbox=json.loads((PARENT/'other_book_transfer/protocol.json').read_text())['T07_stain_paper_box_xywh' if key=='T07' else 'C01_spot_box_xywh']
                    rx,ry,rw,rh=rbox
                    realrows.append(dict(condition=condition,step=step,case=key,region='selected_spot_or_stain',
                        nonink_n=rw*rh,nonink_false=int(pred[ry:ry+rh,rx:rx+rw].sum())))
            print(condition,step,'evaluated',flush=True)
            del model,payload;torch.cuda.empty_cache()
    for case in cases:
        if case['family']!='contact_stain':continue
        items=[('原本',case['source']),('評価入力',case['input'])]
        for condition in protocol['conditions']:
            for step in (432,2160):
                p=mask(dest/'synthetic_masks'/condition/f'{step:05d}'/(case['id']+'.png'))
                items.append((condition+f' {step}',np.where(p,0,255).astype(np.uint8)))
        fig=dest/'figures'/(case['id']+'_原本比較.png');montage(fig,'学習2周と10周 / '+case['id'],items)
        figures.append(dict(path=str(fig.relative_to(dest)),sha256=sha(fig)))
    for case in real_cases:
        key=case['id'];box=case.get('roi',case.get('crop_xywh'));x,y,w,h=box
        items=[('原本',real_sources[key][y:y+h,x:x+w])]
        for condition in protocol['conditions']:
            for step in (432,2160):
                p=mask(dest/'real_masks'/condition/f'{step:05d}'/(key+'.png'))
                items.append((condition+f' {step}',np.where(p,0,255).astype(np.uint8)))
        fig=dest/'figures'/(key+'_原本比較.png');real.panel(fig,key+' / 学習2周と10周',items)
        figures.append(dict(path=str(fig.relative_to(dest)),sha256=sha(fig)))
    dump(dest/'results.json',dict(protocol_sha256=sha(protocol_path),synthetic=rows,
        real=realrows,traced_fade=thinrows,figures=figures,formal_accuracy=False))
    for condition in protocol['conditions']:
        for step in protocol['checkpoints']:
            a=[r for r in rows if r['condition']==condition and r['step']==step and r['family']=='contact_stain']
            print(condition,step,'contact core',sum(r['core_r2'] for r in a),'/',sum(r['core_n'] for r in a),
                'paper',sum(r['paper_ink'] for r in a),'/',sum(r['paper_n'] for r in a),flush=True)
    for r in realrows:
        if r['step'] in (432,2160):print('REAL',r['condition'],r['step'],r['case'],r['region'],r.get('ink_r2'),r['nonink_false'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--nogan',action='store_true');p.add_argument('--unet',action='store_true');a=p.parse_args()
    if a.action=='freeze':freeze()
    else:main(nogan=a.nogan,unet=a.unet)
