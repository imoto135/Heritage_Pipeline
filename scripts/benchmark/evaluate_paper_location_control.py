"""Same26 frozen readouts plus8 training-fit diagnostics; no checkpoint selection."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import run_paper_location_control as tr
import evaluate_background_diversity_control as old
from ragan_pilot_support import make_model,auxiliaries
from evaluate_matched_supervised_generators import ThreeInput,probability
from evaluate_ragan_native_tiling import tiled_probability
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import dump,sha,rgb,save
import apply_ragan_real_regions as real
ROOT=tr.ROOT;OUT=tr.OUT;GRIDS=old.GRIDS
def freeze():
    assert not (OUT/'evaluation_plan.json').exists();tr.check();ep=old.check();cases=old.cases()
    files={**ep['source_files']};extra=[Path(__file__),ROOT/'scripts/benchmark/evaluate_background_diversity_control.py',tr.base.OUT/'evaluation_plan.json']
    for f in extra:files[str(f.relative_to(ROOT))]=sha(f)
    dump(OUT/'evaluation_plan.json',dict(utc=datetime.now(timezone.utc).isoformat(),train_protocol_sha256=sha(OUT/'protocol.json'),
      primary_additional_step=864,secondary432_not_selected=True,threshold=.5,grids=GRIDS,
      cases=ep['cases'],source_files=files,training_fit='8 new teacher windows, ink1159/paper14728; training fit only, not transfer',
      comparisons='8 rerun paired near/far masks; matched counts/contrast/luminance strata, coefficient1 both. Old inputs and labels unchanged.',
      known_development=True,formal_accuracy=False))
    print('Frozen624 new26case readouts and64 training-window diagnostics',flush=True)
def check():
    p=json.loads((OUT/'evaluation_plan.json').read_text());tr.check();assert p['train_protocol_sha256']==sha(OUT/'protocol.json')
    for n,h in p['source_files'].items():assert sha(ROOT/n)==h,n
    return p
def run(seed,device_name):
    p=check();cases=old.cases();cs=json.loads((tr.OLD/'selected_sources.json').read_text())['cases'];torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device)
    for kind in ['unet','ragan']:
      for arm in tr.ARMS:
        name=f'{kind}_seed{seed}_{arm}';dest=OUT/'evaluation'/name;assert not (dest/'results.json').exists();checkpoint=OUT/'runs'/name/'step_00864.pt'
        q=torch.load(checkpoint,map_location='cpu',weights_only=False);assert q['protocol_sha256']==sha(OUT/'protocol.json') and q['additional_steps']==864 and q['step']==3888
        model=make_model(kind).to(device);model.load_state_dict(q['model']);del q;model.eval();before=digest_state(model);adapter=ThreeInput(model,kind).eval();rows=[];fits=[]
        for c in cases:
            for grid,offset in GRIDS.items():
                if grid=='full512':prob=real.infer_roi(model,kind,c['page'],c['roi'],device) if c['group'] in ['real','reviewed_real'] else probability(model,kind,auxiliaries(c['page']),device)
                else:prob,*_=tiled_probability(adapter,c['page'],c['roi'],device,offset)
                assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1;pred=prob<.5;target=dest/'masks'/grid/f"{c['id']}.png";save(target,pred)
                rows.append(dict(condition=name,kind=kind,seed=seed,arm=arm,case=c['id'],grid=grid,group=c['group'],family=c['family'],secondary=c.get('secondary',False),mask_sha256=sha(target),**old.metrics(pred,c)))
            if c['id'] in ['R03','R05','R06','R08','D04']:print(name,c['id'],[(r['grid'],r['core_r2'],r['paper_ink']) for r in rows[-3:]],flush=True)
        for c in cs:
            lab=np.asarray(Image.open(tr.OLD/'labels'/f"{c['id']}.png"));m=dict(core=lab==1,paper=lab==2,group='new_teacher_fit');prob=probability(model,kind,auxiliaries(rgb(ROOT/c['crop'])),device);pred=prob<.5
            target=dest/'new_teacher_fit'/f"{c['id']}.png";save(target,pred);extra={}
            for location in tr.ARMS:
                paper=np.asarray(Image.open(OUT/'paper_masks'/location/f"{c['id']}.png"))>0;extra[location+'_paper_n']=int(paper.sum());extra[location+'_paper_ink']=int((pred&paper).sum())
            fits.append(dict(condition=name,case=c['id'],mask_sha256=sha(target),**extra,**old.metrics(pred,m)))
        assert before==digest_state(model);dump(dest/'results.json',dict(rows=rows,teacher_fit=fits,checkpoint_sha256=sha(checkpoint),unchanged_state_sha256=before,evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),formal_accuracy=False))
        print('Evaluated',name,len(rows),'readouts;',sum(r['core_r2'] for r in fits),'/1159 teacher ink;',sum(r['paper_ink'] for r in fits),'paper ink',flush=True);del model,adapter;torch.cuda.empty_cache()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--device',default='cuda:0');args=a.parse_args();freeze() if args.action=='freeze' else run(args.seed,args.device)
