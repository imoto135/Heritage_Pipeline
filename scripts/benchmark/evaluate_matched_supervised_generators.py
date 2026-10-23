"""Same frozen readout for both generator families, without changing forwards."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_matched_supervised_generators import OUT,CONDITIONS,native
from ragan_pilot_support import make_model,predict,auxiliaries
from audit_ragan_inference_context import cases,HARD,digest_state,OUT as CONTEXT
from evaluate_ragan_native_tiling import load_cases,measure,tiled_probability,METHODS
from compare_v5_aligned_fade_control import dump,sha,rgb,save,coverage
from run_ragan_joint_continuation import arrays
import run_ragan_full_continuation as full
import apply_ragan_real_regions as real
import evaluate_classical_partial_labels as partial


def probability(model,kind,channels,device):
    x=torch.from_numpy(channels.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    with torch.inference_mode():return predict(model,kind,x)[0,0].cpu().numpy()


class ThreeInput(torch.nn.Module):
    """Only adapt the tiling helper's call signature; preserve model probabilities."""
    def __init__(self,model,kind):super().__init__();self.model=model;self.kind=kind
    def forward(self,ori,otsu,sobel):
        return predict(self.model,self.kind,torch.cat([ori,otsu,sobel],dim=1))


def freeze():
    assert not (OUT/'evaluation_sources.json').exists()
    tp=json.loads((CONTEXT/'tiling_protocol.json').read_text())
    dump(OUT/'evaluation_sources.json',dict(protocol_sha256=sha(OUT/'protocol.json'),plan_sha256=sha(OUT/'evaluation_plan.json'),
        sources=tp['source_hashes'],case_ids=[c['id'] for c in load_cases()],
        original_canonical_crops=[dict(patch=c['patch'],xy=c['xy']) for c in cases()],
        adapter='ThreeInput only combines5channels for existing tiled_probability; model forward and output unchanged'))
    for name in [Path(__file__).name,'evaluate_ragan_native_tiling.py','apply_ragan_real_regions.py','evaluate_classical_partial_labels.py']:
        shutil.copy2(ROOT/'scripts/benchmark'/name,OUT/'source_snapshot'/name)
    print('Frozen generic readout for4 new models, unchanged19 cases and original geometry',flush=True)


def run(condition,device_name):
    conf=CONDITIONS[condition];kind=conf['kind'];seed_index=conf['seed_index']
    torch.set_num_threads(2);device=torch.device(device_name);torch.cuda.set_device(device)
    ep=json.loads((OUT/'evaluation_sources.json').read_text())
    assert ep['protocol_sha256']==sha(OUT/'protocol.json') and ep['plan_sha256']==sha(OUT/'evaluation_plan.json')
    assert all(sha(ROOT/name)==h for name,h in ep['sources'].items())
    allcases=load_cases();traincases=cases();records=full.read_rows('train')
    plan=json.loads((OUT/'crops'/f'seed{seed_index}.json').read_text())
    for step in (3024,2160):
        checkpoint=OUT/condition/f'step_{step:05d}.pt';dest=OUT/f'evaluation_{step:05d}'/condition
        assert not (dest/'results.json').exists()
        model=make_model(kind).to(device);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['kind']==kind and payload['step']==step
        model.load_state_dict(payload['model']);del payload;model.eval();before=digest_state(model)
        rows=[];realrows=[];training=[];origins=[];seen=[];tile_rows=[];tile_real=[]
        def score_real(pred,c,method,target):
            if c['subregions']:
                for r in c['subregions']:
                    x,y,w,h=r['xywh'];target.append(dict(condition=condition,case=c['id'],method=method,region=r['region'],**partial.measure(pred[y:y+h,x:x+w],r['labels'],r['paths'])))
            else:target.append(dict(condition=condition,case=c['id'],method=method,region='selected_spot_or_stain',nonink_n=int(c['paper'].sum()),nonink_false=int((pred&c['paper']).sum())))
        for c in allcases:
            if c['group']=='real':paper=real.infer_roi(model,kind,c['page'],c['roi'],device)
            else:paper=probability(model,kind,auxiliaries(c['page']),device)
            pred=paper<.5
            save(dest/'masks'/(c['id']+'.png'),pred)
            rows.append(dict(condition=condition,case=c['id'],group=c['group'],family=c['family'],**measure(pred,c)))
            if c['group']=='real':score_real(pred,c,'full512',realrows)
        for c in traincases:
            patch=c['patch'];row=c['record'];x,y=c['xy'];trace=c['trace'];image=rgb(ROOT/row['input']);big=auxiliaries(image)
            identity=next(r for r in records if r['patch_id']==patch and r['damage_family']=='identity')
            pred=probability(model,kind,auxiliaries(rgb(ROOT/identity['input'])),device)<.5
            save(dest/'training'/patch/'identity512.png',pred)
            training.append(dict(condition=condition,patch=patch,family='identity',view='full512',**coverage(pred,trace)))
            for view,a in [('canonical_local',auxiliaries(image[y:y+256,x:x+256])),('canonical_fullaux',big[y:y+256,x:x+256])]:
                pred=probability(model,kind,a,device)<.5
                save(dest/'training'/patch/(view+'.png'),pred)
                training.append(dict(condition=condition,patch=patch,family='targeted_fade',view=view,xy=[x,y],**coverage(pred,trace[y:y+256,x:x+256])))
            if patch==HARD:
                for origin in range(0,33,4):
                    pred=probability(model,kind,big[0:256,origin:origin+256],device)<.5
                    save(dest/'origin_masks'/f'x{origin:02d}.png',pred)
                    origins.append(dict(condition=condition,xy=[origin,0],**coverage(pred,trace[0:256,origin:origin+256])))
            # Match each checkpoint to its own last seen strong-fade training batch.
            candidates=[(k,e) for k,e in enumerate(plan[:step*4]) if e['active'] and records[e['index']]['patch_id']==patch]
            k,last=candidates[-1];batch_entries=plan[(k//4)*4:(k//4)*4+4];position=k%4;inputs=[]
            for e in batch_entries:
                sx,sy=e['xy'];im=rgb(ROOT/records[e['index']]['input'])[sy:sy+256,sx:sx+256]
                inputs.append(torch.from_numpy(auxiliaries(im).copy()).permute(2,0,1).float()/255)
            batch=torch.stack(inputs).to(device)
            with torch.inference_mode():
                outputs={'actual_batch4':predict(model,kind,batch)[position,0].cpu().numpy()<.5,
                    'single_crop1':predict(model,kind,batch[position:position+1].contiguous())[0,0].cpu().numpy()<.5}
            sx,sy=last['xy']
            for view,pred in outputs.items():
                save(dest/'seen_masks'/patch/(view+'.png'),pred)
                seen.append(dict(condition=condition,patch=patch,view=view,xy=[sx,sy],batch_step=k//4,position=position,
                    **coverage(pred,trace[sy:sy+256,sx:sx+256])))
        if step==3024:
            adapter=ThreeInput(model,kind).eval()
            for c in allcases:
                for method,offset in METHODS.items():
                    pmap,std,seams,split,stats=tiled_probability(adapter,c['page'],c['roi'],device,offset)
                    pred=pmap<.5;paper,core=c['paper'],c['core']
                    save(dest/'tiling'/method/(c['id']+'.png'),pred)
                    save(dest/'seams'/method/(c['id']+'.png'),seams)
                    stats.update(seam_paper_n=int((seams&paper).sum()),seam_paper_ink=int((seams&paper&pred).sum()),
                        other_paper_n=int((~seams&paper).sum()),other_paper_ink=int((~seams&paper&pred).sum()),core_disagree=int((split&core).sum()))
                    tile_rows.append(dict(condition=condition,case=c['id'],group=c['group'],family=c['family'],method=method,**measure(pred,c),**stats))
                    if c['group']=='real':score_real(pred,c,method,tile_real)
                print(condition,step,'tiled',c['id'],flush=True)
        _,aa=arrays('val');validation=[(a,torch.from_numpy(auxiliaries(a['image']).copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255) for a in aa]
        val=native(model,kind,device,validation)
        assert before==digest_state(model)
        dump(dest/'results.json',dict(condition=condition,kind=kind,step=step,rows=rows,real_regions=realrows,training=training,origin_rows=origins,
            seen_rows=seen,tiling_rows=tile_rows,tiling_real_regions=tile_real,native_val8=val,checkpoint_sha256=sha(checkpoint),
            unchanged_model_state_sha256=before,source_protocol_sha256=sha(OUT/'protocol.json'),formal_accuracy=False))
        del model;torch.cuda.empty_cache()
        print('Readout completed',condition,step,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);parser.add_argument('--condition',choices=list(CONDITIONS));parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    freeze() if args.action=='freeze' else run(args.condition,args.device)
