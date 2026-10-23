"""Fixed512, original crop, nine offsets and fixed-grid transfer readouts."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_occurrence_jitter_control import OUT,PARENT,CONDITIONS
from audit_ragan_inference_context import HARD,Reader,cases,digest_state,OUT as CONTEXT
from evaluate_ragan_native_tiling import load_cases,tiled_probability,measure,METHODS
from compare_v5_aligned_fade_control import dump,sha,rgb,mask,save,coverage
from ragan_pilot_support import make_model,auxiliaries
import evaluate_ragan_joint_continuation as common
import evaluate_classical_partial_labels as partial
import run_ragan_full_continuation as full


def freeze():
    assert not (OUT/'evaluation_plan.json').exists()
    dump(OUT/'evaluation_plan.json',dict(training_protocol_sha256=sha(OUT/'protocol.json'),
        common_readout='unchanged evaluate_ragan_joint_continuation.py; fixed val8/synthetic6/known49/real5',
        common_code_sha256=sha(ROOT/'scripts/benchmark/evaluate_ragan_joint_continuation.py'),
        primary_step=864,secondary_step=432,all_conditions=True,no_checkpoint_selection=True,
        canonical_crop='original epoch13 positions, independent of new draws; explicitly not last seen training crops',
        origin_probes=dict(patch=HARD,x_origins=list(range(0,33,4)),y_origin=0,auxiliaries='fixed full512 auxiliary image'),
        tiling=dict(steps=[864],cases='same19',methods=METHODS,
            inherited_protocol_path=str((CONTEXT/'tiling_protocol.json').relative_to(ROOT)),inherited_sha256=sha(CONTEXT/'tiling_protocol.json')),
        labels_provisional=True,known_development=True,formal_accuracy=False))
    for filename in [Path(__file__).name,'evaluate_ragan_joint_continuation.py','audit_ragan_inference_context.py','evaluate_ragan_native_tiling.py']:
        shutil.copy2(ROOT/'scripts/benchmark'/filename,OUT/'source_snapshot'/filename)
    print('Frozen primary864/secondary432 readouts; same9 offset probes and19 tiling cases',flush=True)


def run(order,device_name):
    torch.set_num_threads(2)
    device=torch.device(device_name)
    torch.cuda.set_device(device)
    condition=f'jitter_order{order}'
    p=json.loads((OUT/'evaluation_plan.json').read_text())
    assert p['training_protocol_sha256']==sha(OUT/'protocol.json')
    assert sha(CONTEXT/'tiling_protocol.json')==p['tiling']['inherited_sha256']
    allcases=load_cases()
    train_records=full.read_rows('train')
    inherited=json.loads((CONTEXT/'tiling_protocol.json').read_text())
    assert all(sha(ROOT/name)==h for name,h in inherited['source_hashes'].items())
    for step in (864,432):
        path=OUT/condition/f'step_{step:05d}.pt'
        destination=f'evaluation_{step:05d}_order{order}'
        dest=OUT/destination
        common.OUT=OUT
        common.LABELS={condition:f'提示毎位置変更 順序{order+1}'}
        common.main(checkpoints_override={condition:path},destination_name=destination,primary_step_override=step,device_name=device_name)
        model=make_model('ragan').to(device)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['step']==2160+step and payload['discriminator'] is None
        model.load_state_dict(payload['model']);del payload
        model.eval();before=digest_state(model);read=Reader(model,device)
        trainrows,originrows=[],[]
        for c in cases():
            patch=c['patch'];x,y=c['xy'];record=c['record']
            identity=next(r for r in train_records if r['patch_id']==patch and r['damage_family']=='identity')
            for family,row in [('identity',identity),('targeted_fade',record)]:
                image=rgb(ROOT/row['input']);channels=auxiliaries(image)
                probability,_,_=read(channels,contiguous=False)
                pred=(1-probability)>=.5
                save(dest/'training_masks'/family/(patch+'_512.png'),pred)
                trainrows.append(dict(condition=condition,patch=patch,family=family,view='full512',input_sha256=sha(ROOT/row['input']),**coverage(pred,c['trace'])))
                if family!='targeted_fade':continue
                for view,a in [('canonical256_local',auxiliaries(image[y:y+256,x:x+256])),('canonical256_fullaux',channels[y:y+256,x:x+256])]:
                    prob,_,_=read(a)
                    pred=prob<.5
                    save(dest/'training_masks'/family/(patch+'_'+view+'.png'),pred)
                    trainrows.append(dict(condition=condition,patch=patch,family=family,view=view,xy=[x,y],
                        meaning='fixed original epoch13 crop, not last jitter presentation',**coverage(pred,c['trace'][y:y+256,x:x+256])))
                if patch==HARD:
                    for origin in range(0,33,4):
                        prob,_,_=read(channels[0:256,origin:origin+256])
                        pred=prob<.5
                        save(dest/'origin_masks'/f'x{origin:02d}.png',pred)
                        originrows.append(dict(condition=condition,patch=patch,xy=[origin,0],**coverage(pred,c['trace'][0:256,origin:origin+256])))
            print(condition,step,patch,'evaluated',flush=True)
        for handle in read.handles:handle.remove()
        if step==864:
            tr,rr=[],[]
            for c in allcases:
                key=c['id']
                for method,offset in [('full512_cached',None),*METHODS.items()]:
                    if method=='full512_cached':
                        if c['group']=='train':fp=dest/'training_masks/targeted_fade'/(key+'_512.png')
                        elif c['group']=='traced_fade':fp=dest/'traced_fade'/(condition+'.png')
                        else:fp=dest/c['group']/condition/(key+'.png')
                        pred=mask(fp);stats={}
                    else:
                        pmap,std,seams,split,stats=tiled_probability(model,c['page'],c['roi'],device,offset)
                        pred=pmap<.5
                        paper,core=c['paper'],c['core']
                        stats.update(seam_paper_n=int((seams&paper).sum()),seam_paper_ink=int((seams&paper&pred).sum()),
                            other_paper_n=int((~seams&paper).sum()),other_paper_ink=int((~seams&paper&pred).sum()),
                            core_disagree=int((split&core).sum()),paper_disagree=int((split&paper).sum()))
                        save(dest/'tiling/seams'/method/(key+'.png'),seams)
                    save(dest/'tiling/masks'/method/(key+'.png'),pred)
                    tr.append(dict(condition=condition,case=key,group=c['group'],family=c['family'],method=method,**measure(pred,c),**stats))
                    if c['group']=='real':
                        if c['subregions']:
                            for reg in c['subregions']:
                                rx,ry,rw,rh=reg['xywh']
                                rr.append(dict(condition=condition,case=key,method=method,region=reg['region'],**partial.measure(pred[ry:ry+rh,rx:rx+rw],reg['labels'],reg['paths'])))
                        else:
                            rr.append(dict(condition=condition,case=key,method=method,region='selected_spot_or_stain',nonink_n=int(c['paper'].sum()),nonink_false=int((pred&c['paper']).sum())))
                print(condition,'tiling',key,flush=True)
            dump(dest/'tiling_results.json',dict(rows=tr,real_regions=rr,source_grid_protocol_sha256=p['tiling']['inherited_sha256'],formal_accuracy=False))
        assert before==digest_state(model)
        dump(dest/'training_and_offsets.json',dict(rows=trainrows,origin_rows=originrows,checkpoint_sha256=sha(path),
            model_state_unchanged_sha256=before,canonical_crops_are_not_last_jitter_draws=True,formal_accuracy=False))
        del model,read;torch.cuda.empty_cache()
    print('Readout completed',condition,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--order',type=int,choices=[0,1]);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();freeze() if args.action=='freeze' else run(args.order,args.device)
