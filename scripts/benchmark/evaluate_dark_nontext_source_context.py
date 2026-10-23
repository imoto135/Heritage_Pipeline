"""Fixed same-source context diagnostic, not held-out performance."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import run_dark_nontext_teacher_control as tr
import evaluate_dark_nontext_teacher_control as ev
from ragan_pilot_support import make_model
from evaluate_matched_supervised_generators import ThreeInput
from evaluate_ragan_native_tiling import tiled_probability
from apply_ragan_real_regions import infer_roi
from audit_ragan_inference_context import digest_state
from compare_v5_aligned_fade_control import rgb,sha,dump,save
ROOT=tr.ROOT;OUT=tr.OUT

def freeze():
    ev.check();assert not (OUT/'source_context_plan.json').exists()
    dump(OUT/'source_context_plan.json',dict(utc=datetime.now(timezone.utc).isoformat(),status='Training-source fit under page context; never a held-out test. Frozen before new predictions.',protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(__file__),cases=[c['id'] for c in json.loads((OUT/'selected_sources.json').read_text())['cases']],grids=ev.GRIDS,readouts=96,threshold=.5,interpretation='Compare source native256 fit to same ROI on original full page via512 or fixed256 grids0/16. Selected stain and clear boxes unchanged. Pixels outside boxes unknown, including visible text in S03. No checkpoint/threshold selection.'))
    print('Frozen96 source-context readouts, distinct from transfer evaluation',flush=True)
def check():
    ev.check();p=json.loads((OUT/'source_context_plan.json').read_text());assert p['protocol_sha256']==sha(OUT/'protocol.json') and p['script_sha256']==sha(__file__);return p

def run(seed,device_name):
    check();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device(device_name);torch.cuda.set_device(device)
    cases=json.loads((OUT/'selected_sources.json').read_text())['cases'];pages={c['source']:rgb(ROOT/c['source']) for c in cases}
    for arm in tr.ARMS:
        name=f'ragan_seed{seed}_{arm}';dest=OUT/'source_context'/name;assert not (dest/'results.json').exists();ck=OUT/'runs'/name/'step_00864.pt';q=torch.load(ck,map_location='cpu',weights_only=False);assert q['protocol_sha256']==sha(OUT/'protocol.json');model=make_model('ragan').to(device);model.load_state_dict(q['model']);del q;model.eval();adapter=ThreeInput(model,'ragan').eval();before=digest_state(model);rows=[]
        for c in cases:
            masks={a:np.asarray(Image.open(OUT/'paper_masks'/a/f"{c['id']}.png"))>0 for a in tr.ARMS}
            for grid,offset in ev.GRIDS.items():
                if grid=='full512':prob=infer_roi(model,'ragan',pages[c['source']],c['roi'],device)
                else:prob,*_=tiled_probability(adapter,pages[c['source']],c['roi'],device,offset)
                assert np.isfinite(prob).all();pred=prob<.5;target=dest/'masks'/grid/f"{c['id']}.png";save(target,pred);vals={a+'_paper_ink':int((pred&m).sum()) for a,m in masks.items()};rows.append(dict(condition=name,seed=seed,arm=arm,case=c['id'],grid=grid,paper_n_each=576,total_ink=int(pred.sum()),mask_sha256=sha(target),**vals))
        assert before==digest_state(model);dump(dest/'results.json',dict(rows=rows,source_context_plan_sha256=sha(OUT/'source_context_plan.json'),checkpoint_sha256=sha(ck),unchanged_state_sha256=before,training_fit_only=True));print('Context evaluated',name,len(rows),'readouts',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['freeze','run']);a.add_argument('--seed',type=int,choices=[0,1]);a.add_argument('--device',default='cuda:0');v=a.parse_args();freeze() if v.action=='freeze' else run(v.seed,v.device)
