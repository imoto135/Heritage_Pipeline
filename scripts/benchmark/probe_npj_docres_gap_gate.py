"""Controlled DocRes gap-selection ablation preserving actual saved npj colours."""
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from build_npj_v6_comparison import ROOT, read, write, sha, dump
from package_saved_npj_inference_v6_review import check
from probe_saved_npj_role_separated_postprocess import infer

TRIAL = ROOT/'tmp_work/npj_literature_trial_20261002'
OUT = ROOT/'tmp_work/npj_docres_gap_gate_pilot_20261002'
SETTINGS = dict(
    scope='Entire historical saved input frame; no ROI coordinates or page id enter detection',
    baseline='Authentic saved npj, not rerun',
    candidate='Unchanged prior anchored short-gap detector, same role masks and thresholds',
    docres='Existing frozen binarization background probability from same historical input',
    thresholds=[.5,.9],
    gating='Retain complete previously accepted component if mean ink probability >= threshold',
    primary_threshold=.9,
    rendering='Same scalar original grayscale/local closed grayscale ratio on saved npj for all conditions',
    paper_kernel=17, ratio_max=1,
    paper_update=False, red_hold_extra_update=False,
    diffusion_rerun=False, model_rerun=False,
    new_pixels_from_docres=False, pixel_GT=False, adopted=False,
    selection='6 known development frames / 5 pages / same 20 ROI; no independent evaluation',
    interpretation='Probe false-gap rejection, not complete text protection or dirt-removal solution',
    threshold_selection='Conventional .5 and .9; probabilities on known examples inspected before experiment; not blind')


def main():
    assert not (OUT/'manifest.json').exists(), 'Use new version'
    OUT.mkdir(parents=True,exist_ok=True)
    selection = json.loads((TRIAL/'selection.json').read_text())
    sealed = json.loads((TRIAL/'selection_complete.json').read_text())
    assert sealed['selection_sha'] == sha(TRIAL/'selection.json')
    for rel,digest in sealed['outputs'].items(): assert sha(TRIAL/rel)==digest,rel
    deps = {}
    for method in ['docres','matched_docres']:
        check(TRIAL/method)
        for name in ['manifest.json','complete.json']:
            p=TRIAL/method/name; deps[str(p.relative_to(ROOT))]=sha(p)
    for p in [Path(__file__),Path(__file__).parent/'probe_saved_npj_role_separated_postprocess.py',
              Path(__file__).parent/'build_npj_v6_comparison.py',
              Path(__file__).parent/'package_saved_npj_inference_v6_review.py',
              TRIAL/'selection.json',TRIAL/'selection_complete.json']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    manifest=dict(settings=SETTINGS,dependencies=deps,frames=selection['frames'])
    dump(OUT/'manifest.json',manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    records=[]
    for spec in selection['frames']:
        fid=spec['id'];src=TRIAL/'inputs'/fid;dest=OUT/'native'/fid
        original=read(src/'saved_input.png');base=read(src/'saved_npj.png')
        assert np.array_equal(base,read(ROOT/spec['saved_npj']))
        assert np.array_equal(original,read(ROOT/spec['input']))
        roles={k:read(src/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        # Reuse the frozen detector; its auxiliary paper result is not used.
        _,old_masks,diagnostic=infer(original,base,roles)
        weak=old_masks['weak_candidate'];ungated=old_masks['stroke_update']
        method='docres' if spec['input_reference_exact'] else 'matched_docres'
        probability=np.load(TRIAL/method/fid/'background_probability.npy',allow_pickle=False)
        assert probability.shape==ungated.shape and np.isfinite(probability).all()
        n,labels=cv2.connectedComponents(weak.astype(np.uint8),connectivity=8)
        means=np.bincount(labels.ravel(),weights=(1-probability).ravel(),minlength=n)
        areas=np.bincount(labels.ravel(),minlength=n)
        means/=np.maximum(areas,1)
        gates={};components=[]
        accepted=np.zeros(n,bool)
        for c in diagnostic['components']:
            if c['accepted']:
                accepted[c['id']]=True
                components.append(dict(**c,docres_mean_ink_probability=float(means[c['id']]),
                    selected_05=bool(means[c['id']]>=.5),selected_09=bool(means[c['id']]>=.9)))
        assert np.array_equal(accepted[labels]&weak,ungated)
        for suffix,threshold in [('05',.5),('09',.9)]:
            gates['gate_'+suffix]=accepted[labels] & weak & (means[labels]>=threshold)
            assert not (gates['gate_'+suffix]&~ungated).any()
        # Equal channel multiplier preserves colour direction before 8bit rounding;
        # masks alone differ among the three ablation conditions.
        gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
        ratio=np.minimum(gray.astype(np.float64)/np.maximum(paper,1),1.)
        changed_value=np.clip(np.rint(base.astype(np.float64)*ratio[:,:,None]),0,255).astype(np.uint8)
        masks=dict(ungated=ungated,**gates)
        images=dict(saved_input=original,saved_npj=base)
        for key,mask in masks.items():
            output=base.copy();output[mask]=changed_value[mask]
            assert np.array_equal(output[~mask],base[~mask])
            assert not (mask&(roles['red']|roles['hold']|roles['extra'])).any()
            images[key+'_output']=output
        masks['docres_ink']=probability<.5
        masks['old_protection_disagreement']=(roles['protect']|roles['hold'])&(probability>=.9)&~(roles['red']|roles['extra'])
        masks.update({k:roles[k] for k in ['red','hold','extra','protect']})
        for k,a in images.items():write(dest/(k+'.png'),a)
        for k,m in masks.items():write(dest/(k+'_mask.png'),m)
        np.save(dest/'scalar_ratio.npy',ratio)
        if fid in ['old_upper_crop','old_full_page']:
            oldfid='upper_crop' if fid=='old_upper_crop' else 'full_page'
            prior=ROOT/'tmp_work/npj_role_separated_postprocess_pilot_20261002/native'/oldfid
            assert np.array_equal(ungated,read(prior/'stroke_update.png',True)), 'Frozen old-gap candidates exact'
        stat=dict(frame=fid,probability_source=method,components=components,
            counts={k:int(m.sum()) for k,m in masks.items()},
            changed_pixels={k:int(np.any(a!=base,axis=2).sum()) for k,a in images.items() if k.endswith('_output')})
        dump(dest/'diagnostics.json',stat);records.append(stat)
        print(fid,json.dumps({k:stat[k] for k in ['counts','changed_pixels']},ensure_ascii=False),flush=True)
    for rel,digest in deps.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'diagnostics.json',records)
    dump(OUT/'verification.json',dict(saved_input_exact=True,saved_npj_exact=True,
        ungated_prior_two_frames_exact=True,gate_subset=True,same_scalar_renderer=True,
        each_outside_mask_exact_saved_npj=True,red_hold_extra_exact_saved_npj=True,
        paper_update=False,model_rerun=False,pixel_GT=False,adopted=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file()}))


if __name__ == '__main__':main()
