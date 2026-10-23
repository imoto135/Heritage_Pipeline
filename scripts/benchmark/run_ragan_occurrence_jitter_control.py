"""Change only per-presentation crops of the seven reviewed strong-fade rows."""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/benchmark'))
import run_ragan_full_continuation as base
import run_ragan_local_region_control as local
from ragan_local_region_support import region_masks
from compare_v5_aligned_fade_control import dump, sha

PARENT = local.OUT
OUT = ROOT/'tmp_work/ragan_occurrence_jitter_control_20261022'
CONDITIONS = {f'jitter_order{o}': dict(gan=False,batch_order_seed=20261021+o,order=o,kind='inkpaper') for o in (0,1)}
ORIGINAL_COORDINATES = base.coordinates
ORIGINAL_CROPS = local.OriginalCrops


def bounds(mask):
    yy,xx=np.where(mask)
    lo=[max(0,int(xx.max())-255),max(0,int(yy.max())-255)]
    hi=[min(256,int(xx.min())),min(256,int(yy.min()))]
    assert all(a<=b for a,b in zip(lo,hi))
    return lo,hi


def freeze():
    assert not OUT.exists()
    OUT.mkdir()
    rows=base.read_rows('train')
    meta=base.metadata()
    regions=region_masks(rows)
    limits={p:bounds(m['core']|m['paper']) for p,m in regions.items()}
    audits=[]
    for condition,conf in CONDITIONS.items():
        source=PARENT/'batch_schedules'/f"inkpaper_order{conf['order']}.json"
        schedule=json.loads(source.read_text())
        dest=OUT/'batch_schedules'/(condition+'.json')
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest)
        rngs={p:np.random.default_rng(7102026+next(i for i,r in enumerate(rows) if r['patch_id']==p)*7919) for p in regions}
        last={};plan=[];usage={p:[] for p in regions};aux_counts=Counter();old_paper_missing=0
        for step,batch in enumerate(schedule):
            for position,index in enumerate(batch['indices']):
                row=rows[index];patch=row['patch_id'];epoch=batch['epoch']
                original=tuple(ORIGINAL_COORDINATES(index,epoch,row,meta))
                active=patch in regions and row['damage_family']=='targeted_fade' and float(row['damage_strength'])==.8
                xy=original
                if active:
                    lo,hi=limits[patch]
                    while True:
                        xy=tuple(int(rngs[patch].integers(a,b+1)) for a,b in zip(lo,hi))
                        if xy!=last.get(patch):break
                    last[patch]=xy
                    x,y=xy;m=regions[patch]
                    assert m['core'][y:y+256,x:x+256].sum()==m['core'].sum()
                    assert m['paper'][y:y+256,x:x+256].sum()==m['paper'].sum()
                    ox,oy=original
                    old_paper_missing+=int(m['paper'].sum()-m['paper'][oy:oy+256,ox:ox+256].sum())
                    aux_counts['examples']+=1
                    aux_counts['core_pixels']+=int(m['core'].sum())
                    aux_counts['paper_pixels']+=int(m['paper'].sum())
                    usage[patch].append(dict(epoch=epoch,xy=list(xy),original_xy=list(original)))
                plan.append(dict(step=step,position=position,index=index,epoch=epoch,active=active,xy=list(xy),original_xy=list(original)))
        assert len(plan)==3456 and aux_counts['examples']==692
        assert old_paper_missing==0, 'Keep auxiliary paper exposures identical to frozen comparator'
        dump(OUT/'per_presentation_crops'/(condition+'.json'),plan)
        audits.append(dict(condition=condition,auxiliary_exposures=dict(aux_counts),baseline_auxiliary_paper_missing=old_paper_missing,
            patches=[dict(patch=p,legal_bounds=limits[p],presentations=len(xs),distinct_xy=len({tuple(x['xy']) for x in xs}),
                x_phases32=sorted({x['xy'][0]%32 for x in xs}),y_phases32=sorted({x['xy'][1]%32 for x in xs}),
                per_epoch=[dict(epoch=e,n=sum(x['epoch']==e for x in xs),distinct_xy=len({tuple(x['xy']) for x in xs if x['epoch']==e})) for e in range(10,14)]) for p,xs in usage.items()]))
    dump(OUT/'crop_exposure_audit.json',audits)
    for name in ['source_hashes.json','evaluation_protocol.json','region_mask_audit.json']:
        shutil.copy2(PARENT/name,OUT/name)
    p=json.loads((PARENT/'protocol.json').read_text())
    p.update(conditions=CONDITIONS,parent_protocol_sha256=sha(PARENT/'protocol.json'),
        changed_factor='only per-presentation crop coordinates of existing7 targeted_fade.80 rows',
        baseline='cached inkpaper_order0/1 from ragan_local_region_control_20261022',
        crops='frozen per-draw legal positions preserving all existing auxiliary ink/paper pixels; other rows unchanged',
        per_draw_plan_sha256={c:sha(OUT/'per_presentation_crops'/(c+'.json')) for c in CONDITIONS},
        equal_auxiliary_pixel_exposures=True,no_input_file_edits=True,no_label_edits=True,no_loss_changes=True,
        primary_readout='step864, fixed full512 and all9 original x-offset probes; no best-checkpoint selection',
        progression_gate=dict(pilot_criterion_not_formal_accuracy=True,
            retained44_fixed512_at_least=36,retained44_worst9_x_origins_at_least=36,both_orders_required=True,
            adoption='also report paper/line width/real-case tradeoffs; passing core coverage alone does not authorize adoption'),
        secondary_step=432,known_development=True,formal_accuracy=False)
    dump(OUT/'protocol.json',p)
    snap=OUT/'source_snapshot';snap.mkdir()
    for name in [Path(__file__).name,'run_ragan_local_region_control.py','run_ragan_full_continuation.py',
                 'ragan_local_region_support.py','run_ragan_stable_bce_control.py','ragan_pilot_support.py']:
        shutil.copy2(ROOT/'scripts/benchmark'/name,snap/name)
    print('Frozen2 order-matched continuations; same692 auxiliary draws and exact auxiliary pixel exposures',flush=True)


def train(condition,device):
    plan=json.loads((OUT/'per_presentation_crops'/(condition+'.json')).read_text())
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'per_presentation_crops'/(condition+'.json'))==p['per_draw_plan_sha256'][condition]
    state={'current':None,'used':0}
    def coordinates(index,epoch,row,meta):
        entry=state['current']
        if entry is not None:
            assert entry['index']==index and entry['epoch']==epoch
            return tuple(entry['xy'])
        return ORIGINAL_COORDINATES(index,epoch,row,meta)
    class OccurrenceCrops(ORIGINAL_CROPS):
        def get(self,index,epoch):
            entry=plan[state['used']]
            assert entry['index']==index and entry['epoch']==epoch
            state['current']=entry
            sample=super().get(index,epoch)
            state['used']+=1
            return sample
    base.coordinates=coordinates
    local.OUT=OUT;local.CONDITIONS=CONDITIONS;local.OriginalCrops=OccurrenceCrops
    local.train(condition,device)
    assert state['used']==3456
    dump(OUT/condition/'crop_usage.json',dict(draws_used=state['used'],plan_sha256=sha(OUT/'per_presentation_crops'/(condition+'.json')),
        schedule_sha256=sha(OUT/'batch_schedules'/(condition+'.json')),all_gets_followed_frozen_plan=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','train'])
    parser.add_argument('--condition',choices=list(CONDITIONS))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    freeze() if args.action=='freeze' else train(args.condition,args.device)
