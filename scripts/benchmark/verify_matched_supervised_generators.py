"""Check fresh initialization, paired data/objectives and saved-mask readouts."""
import hashlib
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
from run_matched_supervised_generators import OUT,CONDITIONS
from audit_ragan_inference_context import cases,HARD
from evaluate_ragan_native_tiling import load_cases
from compare_v5_aligned_fade_control import dump,sha,mask
from ragan_local_region_support import region_masks
import run_ragan_full_continuation as full


def state_hash(state):
    digest=hashlib.sha256()
    for key,tensor in sorted(state.items()):digest.update(key.encode());digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=json.loads((OUT/'protocol.json').read_text());es=json.loads((OUT/'evaluation_sources.json').read_text())
    assert es['protocol_sha256']==sha(OUT/'protocol.json') and es['plan_sha256']==sha(OUT/'evaluation_plan.json')
    for hashes in [json.loads((OUT/'source_hashes.json').read_text()),es['sources']]:
        assert all(sha(ROOT/name)==h for name,h in hashes.items())
    assert sha(full.PARENT/'train_manifest.jsonl')==p['train_manifest_sha256'] and sha(full.PARENT/'val_manifest.jsonl')==p['val_manifest_sha256']
    records=full.read_rows('train');regions=region_masks(records);traincases={c['patch']:c for c in cases()};allcases={c['id']:c for c in load_cases()}
    assert {r['group'] for r in records}.isdisjoint({r['group'] for r in full.read_rows('val')})
    expected={}
    for s in (0,1):
        plan=json.loads((OUT/'crops'/f'seed{s}.json').read_text());schedule=json.loads((OUT/'schedules'/f'seed{s}.json').read_text())
        assert len(plan)==12096 and len(schedule)==3024
        assert sha(OUT/'crops'/f'seed{s}.json')==p['crop_plans_sha256'][str(s)] and sha(OUT/'schedules'/f'seed{s}.json')==p['schedules_sha256'][str(s)]
        active=positive=negative=0
        for k,e in enumerate(plan):
            assert e['index']==schedule[k//4]['indices'][k%4] and e['epoch']==schedule[k//4]['epoch']
            row=records[e['index']]
            wanted=row['patch_id'] in regions and row['damage_family']=='targeted_fade' and float(row['damage_strength'])==.8
            assert wanted==e['active']
            if not wanted:continue
            x,y=e['xy'];m=regions[row['patch_id']]
            assert int(m['core'][y:y+256,x:x+256].sum())==int(m['core'].sum()) and int(m['paper'][y:y+256,x:x+256].sum())==int(m['paper'].sum())
            active+=1;positive+=int(m['core'].sum());negative+=int(m['paper'].sum())
        assert active==2422 and len({e['index'] for e in plan})==864
        expected[s]=dict(active=active,positive=positive,negative=negative)
    initial={};checkpoints=[]
    for c,conf in CONDITIONS.items():
        s=conf['seed_index'];a=torch.load(OUT/c/'initial.pt',map_location='cpu',weights_only=False)
        assert a['step']==0 and a['kind']==conf['kind'] and a['seed']==conf['seed'] and a['protocol_sha256']==sha(OUT/'protocol.json')
        initial[c]=state_hash(a['model'])
        completion=json.loads((OUT/c/'completion.json').read_text())
        assert completion['fresh_model'] and completion['fresh_Adam'] and completion['initial_optimizer_steps']==[] and completion['final_optimizer_steps']==[3024]
        assert completion['schedule_sha256']==p['schedules_sha256'][str(s)] and completion['crop_plan_sha256']==p['crop_plans_sha256'][str(s)]
        losses=json.loads((OUT/c/'loss_components.json').read_text());assert len(losses)==3024
        assert sum(r['active_examples'] for r in losses)==expected[s]['active']
        assert sum(r['positive_pixels'] for r in losses)==expected[s]['positive'] and sum(r['paper_pixels'] for r in losses)==expected[s]['negative']
        for r in losses:
            reconstructed=r['global_loss']+r['normalized_ink_BCE']+r['normalized_paper_BCE']
            assert abs(reconstructed-r['total'])<1e-4*max(1,abs(reconstructed))
        for step in (432,2160,3024):
            path=OUT/c/f'step_{step:05d}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
            assert payload['step']==step and payload['kind']==conf['kind'] and payload['protocol_sha256']==sha(OUT/'protocol.json') and payload['discriminator'] is None
            assert {int(st['step']) for st in payload['optimizer']['state'].values() if 'step' in st}=={step}
            assert state_hash(payload['model'])!=initial[c]
            checkpoints.append(dict(condition=c,step=step,sha256=sha(path)))
    assert initial['unet_seed0']!=initial['unet_seed1'] and initial['ragan_seed0']!=initial['ragan_seed1']
    for patch,m in regions.items():
        for key in ['core','paper','known_paper','roi']:assert np.array_equal(full.gray(OUT/'region_masks'/(patch+'_'+key+'.png'))>0,m[key])
    yy,xx=np.ogrid[-2:3,-2:3];disk=xx*xx+yy*yy<=4;scored=0
    def core_score(pred,trace,r,exact='r0',near='r2'):
        nonlocal scored
        assert pred.shape==trace.shape
        assert int((pred&trace).sum())==r[exact] and int((ndi.binary_dilation(pred,structure=disk)&trace).sum())==r[near]
        scored+=1
    for step in (3024,2160):
        for c in CONDITIONS:
            dest=OUT/f'evaluation_{step:05d}'/c;a=json.loads((dest/'results.json').read_text())
            assert a['checkpoint_sha256']==sha(OUT/c/f'step_{step:05d}.pt') and a['source_protocol_sha256']==sha(OUT/'protocol.json')
            for r in a['rows']:
                case=allcases[r['case']];pred=mask(dest/'masks'/(r['case']+'.png'))
                core_score(pred,case['core'],r,'core_exact','core_r2');assert int((pred&case['paper']).sum())==r['paper_ink']
            for r in a['training']:
                trace=traincases[r['patch']]['trace']
                if r['view']=='full512':name='identity512.png'
                else:x,y=r['xy'];trace=trace[y:y+256,x:x+256];name=r['view']+'.png'
                core_score(mask(dest/'training'/r['patch']/name),trace,r)
            for r in a['seen_rows']:
                x,y=r['xy'];trace=traincases[r['patch']]['trace'][y:y+256,x:x+256]
                core_score(mask(dest/'seen_masks'/r['patch']/(r['view']+'.png')),trace,r)
            for r in a['origin_rows']:
                x,y=r['xy'];core_score(mask(dest/'origin_masks'/f'x{x:02d}.png'),traincases[HARD]['trace'][y:y+256,x:x+256],r)
            for r in a['tiling_rows']:
                case=allcases[r['case']];pred=mask(dest/'tiling'/r['method']/(r['case']+'.png'))
                core_score(pred,case['core'],r,'core_exact','core_r2');assert int((pred&case['paper']).sum())==r['paper_ink']
                assert r['seam_paper_n']+r['other_paper_n']==r['paper_n'] and r['seam_paper_ink']+r['other_paper_ink']==r['paper_ink']
            for r in [*a['real_regions'],*a['tiling_real_regions']]:
                case=allcases[r['case']]
                pred=mask(dest/'masks'/(r['case']+'.png')) if r['method']=='full512' else mask(dest/'tiling'/r['method']/(r['case']+'.png'))
                if case['subregions']:
                    region=next(q for q in case['subregions'] if q['region']==r['region']);x,y,w,h=region['xywh'];sub=pred[y:y+h,x:x+w];lab=region['labels']
                    core_score(sub,lab==1,r,'ink_exact','ink_r2');assert int((sub&(lab==2)).sum())==r['nonink_false']
                else:assert int((pred&case['paper']).sum())==r['nonink_false']
    combined=json.loads((OUT/'combined_results.json').read_text());classic=json.loads((OUT/'classical_reference.json').read_text())
    assert all(sha(ROOT/name)==h for name,h in classic['prediction_sha256'].items())
    links=0;report=OUT/'比較結果_教師あり生成器.md'
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',report.read_text()):
        if '://' not in target and not target.startswith('#'):assert (report.parent/target.split('#')[0]).exists(),target;links+=1
    for name in ['run_matched_supervised_generators.py','matched_generator_support.py','evaluate_matched_supervised_generators.py','report_matched_supervised_generators.py',
                 'ragan_pilot_support.py','run_ragan_stable_bce_control.py','ragan_local_region_support.py','run_ragan_full_continuation.py','evaluate_ragan_native_tiling.py']:
        assert sha(ROOT/'scripts/benchmark'/name)==sha(OUT/'source_snapshot'/name)
    assert all(r['exact_same_probabilities'] for r in json.loads((OUT/'adapter_checks.json').read_text()))
    shutil.copy2(Path(__file__),OUT/'source_snapshot'/Path(__file__).name)
    dump(OUT/'integrity_verification.json',dict(checkpoints=checkpoints,initial_state_sha256=initial,
        independently_initialized_twice_per_architecture=True,no_warm_start=True,no_GAN=True,paired_schedules_crops_objectives_and_exposures_verified=True,
        unchanged_input_teacher_and_mask_hashes_verified=True,all864_training_records_present=True,independent_saved_mask_scores_verified=scored,
        tiled_adapter_preserves_probabilities=True,report_local_links_verified=links,report_sha256=sha(report),protocol_sha256=sha(OUT/'protocol.json'),
        figure_hashes={f:sha(OUT/f) for f in combined['figures']},source_snapshot_hashes={f.name:sha(f) for f in sorted((OUT/'source_snapshot').glob('*.py'))},formal_accuracy=False))
    print('Verified12 checkpoints,4 fresh initializations, paired inputs/objectives,',scored,'mask scores,',len(combined['figures']),'PNGs',flush=True)


if __name__=='__main__':main()
