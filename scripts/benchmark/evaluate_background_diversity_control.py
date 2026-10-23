"""Frozen readout of12 continuations on19 inherited and7 reviewed cases."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from PIL import Image
import run_background_diversity_control as train
from evaluate_ragan_native_tiling import load_cases,tiled_probability
from evaluate_matched_supervised_generators import ThreeInput,probability
from ragan_pilot_support import make_model,auxiliaries
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import dump,sha,rgb,save
import apply_ragan_real_regions as real
import evaluate_classical_partial_labels as partial
from evaluate_frozen_real_native256 import sources as reviewed_sources,label as reviewed_label
ROOT=train.ROOT;OUT=train.OUT;GRIDS={'full512':None,'tiled0':0,'tiled16':16}
def cases():
    cc=load_cases()
    for c in reviewed_sources():
        labels,paths=reviewed_label(c)
        cc.append(dict(id=c['id'],family=c['family'],group='reviewed_real',page=rgb(ROOT/c['source']),roi=c['roi'],
          core=labels==1,paper=labels==2,labels=labels,paths=paths,secondary=c['id']=='R02',book=c['book'],files=[c['source']]))
    assert len(cc)==26 and len({c['id'] for c in cc})==26
    return cc
def freeze():
    assert not (OUT/'evaluation_plan.json').exists();train.check();cc=cases();files={}
    for c in cc:
        for name in c['files']:files[name]=sha(ROOT/name)
    for p in [ROOT/'tmp_work/frozen_real_transfer_20261022/protocol.json',ROOT/'tmp_work/frozen_real_transfer_20261022/additional_control/protocol.json',ROOT/'tmp_work/frozen_real_transfer_20261022/annotation_review.json',Path(__file__),ROOT/'scripts/benchmark/evaluate_frozen_real_native256.py',ROOT/'scripts/benchmark/evaluate_ragan_native_tiling.py']:
        files[str(p.relative_to(ROOT))]=sha(p)
    dump(OUT/'evaluation_plan.json',dict(utc=datetime.now(timezone.utc).isoformat(),train_protocol_sha256=sha(OUT/'protocol.json'),
      source_files=files,primary_additional_steps=864,secondary_checkpoint_saved_but_not_selected=432,threshold=.5,grids=GRIDS,
      cases=[dict(id=c['id'],group=c['group'],family=c['family'],roi=c['roi'],core_n=int(c['core'].sum()),paper_n=int(c['paper'].sum()),secondary=c.get('secondary',False)) for c in cc],
      comparisons='Original14epoch plus matched18epoch continue/samebook/multibook; all4 kind/seed pairs, allgrids',
      interpretation='Known development and source-only provisional labels; no formal accuracy, no best-seed or grid selection',no_R07=True))
    print('Frozen26cases x12conditions x3grids=936readouts',flush=True)
def check():
    p=json.loads((OUT/'evaluation_plan.json').read_text());assert p['train_protocol_sha256']==sha(OUT/'protocol.json');train.check()
    for n,h in p['source_files'].items():assert sha(ROOT/n)==h,n
    return p
def metrics(pred,c):
    distance=ndi.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape,np.inf)
    out=dict(core_n=int(c['core'].sum()),core_exact=int((pred&c['core']).sum()),core_r2=int(((distance<=2)&c['core']).sum()),paper_n=int(c['paper'].sum()),paper_ink=int((pred&c['paper']).sum()),total_ink=int(pred.sum()))
    if c['group']=='reviewed_real':out['paths']=partial.measure(pred,c['labels'],c['paths'])['paths']
    return out
def run(seed,device_name):
    p=check();cc=cases();torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device)
    for kind in ['unet','ragan']:
      for arm in train.ARMS:
        name=f'{kind}_seed{seed}_{arm}';dest=OUT/'evaluation'/name;assert not (dest/'results.json').exists()
        checkpoint=OUT/'runs'/name/'step_00864.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['protocol_sha256']==sha(OUT/'protocol.json') and payload['additional_steps']==864 and payload['step']==3888
        model=make_model(kind).to(device);model.load_state_dict(payload['model']);del payload;model.eval();before=digest_state(model);adapter=ThreeInput(model,kind).eval();rows=[]
        for c in cc:
            for grid,offset in GRIDS.items():
                if grid=='full512':
                    prob=real.infer_roi(model,kind,c['page'],c['roi'],device) if c['group'] in ['real','reviewed_real'] else probability(model,kind,auxiliaries(c['page']),device)
                else:prob,*_=tiled_probability(adapter,c['page'],c['roi'],device,offset)
                assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1;pred=prob<.5
                target=dest/'masks'/grid/f"{c['id']}.png";save(target,pred)
                rows.append(dict(condition=name,kind=kind,seed=seed,arm=arm,case=c['id'],grid=grid,group=c['group'],family=c['family'],secondary=c.get('secondary',False),mask_sha256=sha(target),**metrics(pred,c)))
            if c['id'] in ['R03','R05','R08','D04']:print(name,c['id'],[(r['grid'],r['core_r2'],r['paper_ink']) for r in rows[-3:]],flush=True)
        assert before==digest_state(model)
        dump(dest/'results.json',dict(rows=rows,checkpoint_sha256=sha(checkpoint),unchanged_state_sha256=before,evaluation_plan_sha256=sha(OUT/'evaluation_plan.json'),formal_accuracy=False))
        del adapter,model;torch.cuda.empty_cache();print('Evaluated',name,len(rows),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--device',default='cuda:0');args=a.parse_args();freeze() if args.action=='freeze' else run(args.seed,args.device)
