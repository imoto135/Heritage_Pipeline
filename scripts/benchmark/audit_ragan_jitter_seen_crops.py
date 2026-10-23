"""Post-readout diagnostic of the actual last seven jitter training batches."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_occurrence_jitter_control import OUT,CONDITIONS
from ragan_pilot_support import make_model,predict,auxiliaries
from audit_ragan_inference_context import cases,digest_state
from compare_v5_aligned_fade_control import dump,sha,rgb,save,coverage
import run_ragan_full_continuation as full


def run(order,device_name):
    condition=f'jitter_order{order}'
    target=OUT/f'jitter_seen_order{order}.json'
    assert not target.exists()
    torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device)
    checkpoint=OUT/condition/'step_00864.pt'
    model=make_model('ragan').to(device)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
    before=digest_state(model)
    plan=json.loads((OUT/'per_presentation_crops'/(condition+'.json')).read_text())
    records=full.read_rows('train');rows=[]
    for case in cases():
        patch=case['patch']
        last=[r for r in plan if r['active'] and records[r['index']]['patch_id']==patch][-1]
        step,position=last['step'],last['position']
        entries=plan[step*4:step*4+4]
        inputs=[]
        for e in entries:
            x,y=e['xy'];a=rgb(ROOT/records[e['index']]['input'])[y:y+256,x:x+256]
            inputs.append(torch.from_numpy(auxiliaries(a).copy()).permute(2,0,1).float()/255)
        batch=torch.stack(inputs).to(device)
        with torch.inference_mode():
            output={'actual_batch4':predict(model,'ragan',batch)[position,0].cpu().numpy()<.5,
                'single_crop1':predict(model,'ragan',batch[position:position+1].contiguous())[0,0].cpu().numpy()<.5}
        x,y=last['xy'];trace=case['trace'][y:y+256,x:x+256]
        assert trace.sum()==case['trace'].sum()
        for view,pred in output.items():
            save(OUT/'jitter_seen_masks'/condition/(patch+'_'+view+'.png'),pred)
            rows.append(dict(condition=condition,patch=patch,view=view,step=step,position=position,xy=[x,y],
                batch_indices=[e['index'] for e in entries],**coverage(pred,trace)))
    assert before==digest_state(model)
    dump(target,dict(rows=rows,exploratory_after_primary_failure=True,does_not_change_progression_criterion=True,
        step864_only=True,checkpoint_sha256=sha(checkpoint),unchanged_state_sha256=before,
        crop_plan_sha256=sha(OUT/'per_presentation_crops'/(condition+'.json')),formal_accuracy=False))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    print(condition,[(r['view'],sum(q['r2'] for q in rows if q['view']==r['view'])) for r in rows[:2]],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--order',type=int,choices=[0,1],required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();run(args.order,args.device)
