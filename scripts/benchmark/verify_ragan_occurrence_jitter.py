"""Verify the crop-only experiment and independently re-score its saved masks."""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/benchmark'))
from run_ragan_occurrence_jitter_control import OUT,PARENT,CONDITIONS,ORIGINAL_COORDINATES
from audit_ragan_inference_context import OUT as CONTEXT,HARD,cases
from evaluate_ragan_native_tiling import load_cases
from compare_v5_aligned_fade_control import dump,sha,mask
from ragan_local_region_support import region_masks
import run_ragan_full_continuation as base


def main():
    p=json.loads((OUT/'protocol.json').read_text())
    ep=json.loads((OUT/'evaluation_plan.json').read_text())
    assert p['parent_protocol_sha256']==sha(PARENT/'protocol.json')
    assert ep['training_protocol_sha256']==sha(OUT/'protocol.json')
    assert all(sha(ROOT/path)==h for path,h in json.loads((OUT/'source_hashes.json').read_text()).items())
    rows=base.read_rows('train');meta=base.metadata();regions=region_masks(rows)
    checkpoints=[];draws=0
    for c,conf in CONDITIONS.items():
        before=f"inkpaper_order{conf['order']}"
        assert sha(OUT/'batch_schedules'/(c+'.json'))==sha(PARENT/'batch_schedules'/(before+'.json'))
        plan=json.loads((OUT/'per_presentation_crops'/(c+'.json')).read_text())
        schedule=json.loads((OUT/'batch_schedules'/(c+'.json')).read_text())
        assert sha(OUT/'per_presentation_crops'/(c+'.json'))==p['per_draw_plan_sha256'][c]
        assert len(plan)==3456
        aux=[r for r in plan if r['active']];assert len(aux)==692
        core_count=paper_count=0
        for k,r in enumerate(plan):
            b=schedule[k//4]
            assert r['step']==k//4 and r['position']==k%4 and r['index']==b['indices'][k%4] and r['epoch']==b['epoch']
            record=rows[r['index']]
            original=list(ORIGINAL_COORDINATES(r['index'],r['epoch'],record,meta))
            assert original==r['original_xy']
            if not r['active']:
                assert r['xy']==original
                continue
            x,y=r['xy'];m=regions[record['patch_id']]
            for key in ['core','paper']:
                assert int(m[key][y:y+256,x:x+256].sum())==int(m[key].sum())
            core_count+=int(m['core'].sum());paper_count+=int(m['paper'].sum())
        oldloss=json.loads((PARENT/before/'loss_components.json').read_text())
        newloss=json.loads((OUT/c/'loss_components.json').read_text())
        assert len(newloss)==864 and sum(x['active_regions'] for x in newloss)==692
        assert sum(x['positive_pixels'] for x in newloss)==core_count==sum(x['positive_pixels'] for x in oldloss)
        assert sum(x['paper_pixels'] for x in newloss)==paper_count==sum(x['paper_pixels'] for x in oldloss)
        for x in newloss:
            expected=x['global_loss']+x['normalized_ink_BCE']+x['normalized_paper_BCE']
            assert abs(expected-x['total'])<1e-4*max(1,abs(expected))
        usage=json.loads((OUT/c/'crop_usage.json').read_text())
        assert usage['draws_used']==3456 and usage['plan_sha256']==p['per_draw_plan_sha256'][c]
        draws+=usage['draws_used']
        completion=json.loads((OUT/c/'completion.json').read_text())
        assert completion['initial_optimizer_steps']==[2160] and completion['final_optimizer_steps']==[3024]
        for step in (216,432,648,864):
            path=OUT/c/f'step_{step:05d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
            assert payload['step']==2160+step and payload['protocol_sha256']==sha(OUT/'protocol.json')
            assert payload['discriminator'] is None and payload['d_optimizer'] is None
            assert {int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}=={2160+step}
            checkpoints.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path)))
    for pth,m in regions.items():
        for key in ['core','paper','known_paper','roi']:
            file=OUT/'region_masks'/(pth+'_'+key+'.png')
            assert sha(file)==sha(PARENT/'region_masks'/file.name) and np.array_equal(base.gray(file)>0,m[key])
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4
    trains={c['patch']:c for c in cases()};scored=0
    for o in (0,1):
        for step in (864,432):
            dest=OUT/f'evaluation_{step:05d}_order{o}'
            data=json.loads((dest/'training_and_offsets.json').read_text())
            assert data['checkpoint_sha256']==sha(OUT/f'jitter_order{o}'/f'step_{step:05d}.pt')
            for r in data['rows']:
                trace=trains[r['patch']]['trace']
                if r['view']=='full512':name=r['patch']+'_512.png'
                else:
                    x,y=r['xy'];trace=trace[y:y+256,x:x+256];name=r['patch']+'_'+r['view']+'.png'
                pred=mask(dest/'training_masks'/r['family']/name)
                assert int(trace.sum())==r['n'] and int((trace&pred).sum())==r['r0']
                assert int((trace&ndi.binary_dilation(pred,structure=disk)).sum())==r['r2'];scored+=1
            for r in data['origin_rows']:
                x,y=r['xy'];trace=trains[HARD]['trace'][y:y+256,x:x+256]
                pred=mask(dest/'origin_masks'/f'x{x:02d}.png')
                assert int((trace&pred).sum())==r['r0'] and int((trace&ndi.binary_dilation(pred,structure=disk)).sum())==r['r2'];scored+=1
    data=json.loads((OUT/'combined_results.json').read_text())
    allcases={c['id']:c for c in load_cases()}
    for r in data['tiling']:
        c=r['condition'];method=r['method'];key=r['case'];case=allcases[key]
        path=(OUT/f'evaluation_00864_order{c[-1]}/tiling/masks'/method/(key+'.png')) if c.startswith('jitter') else (CONTEXT/'tiling/masks'/c/method/(key+'.png'))
        pred=mask(path)
        assert int((pred&case['core']).sum())==r['core_exact']
        assert int((ndi.binary_dilation(pred,structure=disk)&case['core']).sum())==r['core_r2']
        assert int((pred&case['paper']).sum())==r['paper_ink'];scored+=1
    for c in CONDITIONS:
        gate=data['progression_gate']['conditions'][c]
        actual=min(r['r2'] for r in data['origins_primary'] if r['condition']==c)
        assert actual==gate['worst9_origins']
        expected=gate['fixed512']>=p['progression_gate']['retained44_fixed512_at_least'] and actual>=p['progression_gate']['retained44_worst9_x_origins_at_least']
        assert expected==gate['passed']
        order=c[-1]
        seen=json.loads((OUT/f'jitter_seen_order{order}.json').read_text())
        primary=json.loads((OUT/f'evaluation_00864_order{order}/training_and_offsets.json').read_text())
        assert seen['unchanged_state_sha256']==primary['model_state_unchanged_sha256']
        assert seen['crop_plan_sha256']==p['per_draw_plan_sha256'][c]
        for r in seen['rows']:
            x,y=r['xy'];trace=trains[r['patch']]['trace'][y:y+256,x:x+256]
            pred=mask(OUT/'jitter_seen_masks'/c/(r['patch']+'_'+r['view']+'.png'))
            assert int((trace&pred).sum())==r['r0'] and int((trace&ndi.binary_dilation(pred,structure=disk)).sum())==r['r2'];scored+=1
    assert data['progression_gate']['passed']==all(v['passed'] for v in data['progression_gate']['conditions'].values())
    sourcefiles=['run_ragan_local_region_control.py','run_ragan_full_continuation.py','ragan_local_region_support.py','run_ragan_stable_bce_control.py','ragan_pilot_support.py']
    for name in sourcefiles:
        assert sha(OUT/'source_snapshot'/name)==sha(PARENT/'source_snapshot'/name)==sha(ROOT/'scripts/benchmark'/name)
    for name in ['run_ragan_occurrence_jitter_control.py','evaluate_ragan_occurrence_jitter.py','report_ragan_occurrence_jitter.py','audit_ragan_jitter_seen_crops.py']:
        assert sha(OUT/'source_snapshot'/name)==sha(ROOT/'scripts/benchmark'/name)
    links=0;report=OUT/'研究判断_提示毎位置変更の対照.md'
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',report.read_text()):
        if '://' not in target and not target.startswith('#'):
            assert (report.parent/target.split('#')[0]).exists(),target
            links+=1
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    dump(OUT/'integrity_verification.json',dict(checkpoints=checkpoints,initial_optimizer_step=2160,final_optimizer_step=3024,
        actual_draws_verified=draws,changed_target_draws=1384,unchanged_other_draws=5528,
        input_label_and_auxiliary_masks_unchanged=True,batch_schedules_identical_to_comparator=True,
        auxiliary_pixel_exposures_identical=True,loss_components_reconstruct_identical_objective=True,
        independent_saved_mask_scores_verified=scored,progression_gate_verified=data['progression_gate'],
        report_local_links_verified=links,report_sha256=sha(report),protocol_sha256=sha(OUT/'protocol.json'),
        figure_hashes={f:sha(OUT/f) for f in data['figures']},
        source_snapshot_hashes={f.name:sha(f) for f in sorted((OUT/'source_snapshot').glob('*.py'))},formal_accuracy=False))
    print('Verified',len(checkpoints),'checkpoints,',draws,'draws,',scored,'mask scores,',len(data['figures']),'PNGs',flush=True)


if __name__=='__main__':main()
