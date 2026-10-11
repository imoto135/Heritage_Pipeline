"""Development: infer a lower ruled margin from original pixels; keep saved npj.

No ROI coordinates enter infer(). Rule-free/uncertain frames abstain. A second
ablation guards original coherent narrow dark ridges. Masks are hypotheses,
not text/dirt GT. No npj or diffusion rerun.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_inference_v6_review import check
from package_saved_npj_role_separated_review import figure
from probe_saved_npj_role_separated_postprocess import harmonic, frame_guard
from probe_defect_context_preprocessing import paper_estimate

OUT = ROOT/'tmp_work/saved_npj_ruled_margin_pilot_v2_20261002'
NOTES = ROOT/'tmp_work/saved_npj_ruled_margin_notes_v2_20261002.md'
PRIOR = ROOT/'tmp_work/saved_npj_paper_only_followup_pilot_20261002'
SCOPE = ROOT/'tmp_work/saved_npj_hnsd_input_scope_diagnostic_20261002'
SELECTION = ROOT/'tmp_work/saved_npj_paper_followup_selection_20261002/selection.json'
SETTINGS = dict(
    scope='Entire aligned original/reference frame; no ROI, book or page labels in infer',
    baseline='Unmodified authentic saved npj; original/reference used for geometry, not replacement outside update',
    ruled_margin=dict(local_dark_contrast_L=8, closing_kernel_native=31,
        horizontal_length='odd max(81, round(.15*width))', min_row_coverage=.65,
        row_group_gap=9, max_rule_height_fraction=.07, min_rule_y_fraction=.35,
        min_lower_margin_fraction=.10, rule_halo_px=8, require_above_ink_fraction=.01,
        require_below_less_ink=True),
    broad_shadow=dict(estimator='Unchanged prior bright-weighted paper_estimate', gaussian_sigma=8,
        soft_L_drop=5, seed_L_drop=8, min_area=128, min_seed=32, min_peak_radius=4,
        observed_npj_residual_L_drop=12, min_residual_pixels=32, min_residual_fraction=.03,
        min_paper_chroma_b=5, frame_guard_px=2),
    narrow_guard=dict(sigmas=[1.5,3.], response_min=2, coherence_min=.8,
        local_dark_drop_min=8, halo_px=2),
    mandatory='Union of existing saved-input red and v6-specific extra hold; not all protect/hold',
    fill='Same saved-npj 4-neighbour harmonic kernel; no pre-fill or model rerun',
    development=True, pixel_GT=False, automatic_adoption=False,
    limitations=['Known failure case influenced design; no independent generalization test',
                 'Rule geometry does not prove absence of margin writing',
                 'Original/reference and actual saved-input pixels differ on two frames',
                 'Inside writing frame intentionally not handled; shadows/creases may be natural'])


def infer(reference, base, mandatory):
    """Full-frame deterministic features; accept no semantic ids or ROI boxes."""
    h,w=reference.shape[:2]
    lab=cv2.cvtColor(cv2.medianBlur(reference,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    L=lab[:,:,0]
    local=cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    ink=(local-L)>=8
    length=max(81,round(.15*w))|1
    horizontal=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,length),np.uint8))>0
    rows=horizontal.sum(1)>=.65*w
    rows=cv2.morphologyEx(rows.astype(np.uint8)[:,None],cv2.MORPH_CLOSE,np.ones((9,1),np.uint8))>0
    _,_,stats,_=cv2.connectedComponentsWithStats(rows.astype(np.uint8),connectivity=8)
    rules=[]
    for st in stats[1:]:
        y0=int(st[1]); y1=y0+int(st[3]); r=max(16,round(.12*h))
        before=ink[max(0,y0-r-8):max(0,y0-8)]
        after=ink[min(h,y1+8):min(h,y1+r+8)]
        fa=float(before.mean()) if before.size else 0.
        fb=float(after.mean()) if after.size else 1.
        accepted=(y0>=.35*h and y1<=.90*h and y1-y0<=.07*h and fa>=.01 and fb<fa)
        rules.append(dict(y0=y0,y1=y1,ink_above=fa,ink_below=fb,accepted=accepted))
    accepted=[r for r in rules if r['accepted']]
    margin=np.zeros((h,w),bool)
    if accepted:
        chosen=max(accepted,key=lambda r:r['y1'])
        margin[chosen['y1']+8:]=True
    else: chosen=None
    paper=paper_estimate(lab)
    drop=cv2.GaussianBlur(paper[:,:,0]-L,(0,0),8,borderType=cv2.BORDER_REFLECT_101)
    nL=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
    nPaper=cv2.morphologyEx(nL,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(65,65)))
    residual=(nPaper-nL)>=12
    valid=margin & frame_guard((h,w),2) & (lab[:,:,2]>=5) & ~mandatory
    soft=(drop>=5)&valid; seed=(drop>=8)&soft
    n,labels,stats,_=cv2.connectedComponentsWithStats(soft.astype(np.uint8),connectivity=8)
    seeds=np.bincount(labels.ravel(),weights=seed.ravel(),minlength=n)
    residuals=np.bincount(labels.ravel(),weights=residual.ravel(),minlength=n)
    distance=cv2.distanceTransform(soft.astype(np.uint8),cv2.DIST_L2,5)
    peak=np.zeros(n,np.float32);np.maximum.at(peak,labels.ravel(),distance.ravel())
    areas=stats[:,4]
    use=(areas>=128)&(seeds>=32)&(peak>=4)&(residuals>=32)&(residuals/np.maximum(areas,1)>=.03)
    use[0]=False
    candidate=use[labels]&soft
    ridge=np.zeros((h,w),bool)
    for s in [1.5,3.]:
        sm=cv2.GaussianBlur(L,(0,0),s,borderType=cv2.BORDER_REFLECT_101)
        xx=cv2.Sobel(sm,cv2.CV_32F,2,0,ksize=3,scale=.25*s*s)
        yy=cv2.Sobel(sm,cv2.CV_32F,0,2,ksize=3,scale=.25*s*s)
        xy=cv2.Sobel(sm,cv2.CV_32F,1,1,ksize=3,scale=.25*s*s)
        disc=np.sqrt((xx-yy)**2+4*xy**2);lo=(xx+yy-disc)/2;hi=(xx+yy+disc)/2
        coherence=1-np.minimum(np.abs(lo)/np.maximum(np.abs(hi),1e-6),1)
        ridge|=(hi>=2)&(coherence>=.8)&ink
    narrow_guard=cv2.dilate(ridge.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    masks=dict(margin=margin,original_local_dark=ink,horizontal=horizontal,
               broad_soft=soft,broad_seed=seed,mandatory=mandatory,narrow_guard=narrow_guard,
               margin_update=candidate,guarded_update=candidate&~narrow_guard)
    components=[dict(id=int(i),xywh=[int(x) for x in stats[i,:4]],area=int(areas[i]),
                     seed_pixels=int(seeds[i]),npj_residual_pixels=int(residuals[i]),
                     peak_radius=float(peak[i]),accepted=bool(use[i])) for i in range(1,n)]
    return masks,dict(rules=rules,chosen_rule=chosen,components=components,
        counts={k:int(v.sum()) for k,v in masks.items()})


def main(freeze):
    assert not (OUT/'complete.json').exists(),'Frozen; use a new version'
    if freeze:
        manifest=json.loads((OUT/'manifest.json').read_text())
        for rel,digest in manifest['dependencies'].items(): assert sha(ROOT/rel)==digest,rel
        assert sha(Path(__file__))==manifest['source_sha256']
        assert sha(OUT/'source_snapshot.py')==manifest['source_sha256']
        assert NOTES.exists(),'Record visual assessment before freeze'
        shutil.copyfile(NOTES,OUT/'report.md')
        paths=sorted(OUT.glob('*.png'))
        with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in paths:z.write(p,p.name)
        with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
            assert z.testzip() is None
            for p in paths:assert z.read(p.name)==p.read_bytes()
        manifest['dependencies'][str(NOTES.relative_to(ROOT))]=sha(NOTES)
        dump(OUT/'manifest.json',manifest)
        dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
        print('Frozen',len(paths),'comparison PNGs',flush=True);return
    assert not (OUT/'manifest.json').exists(),'Draft already created'
    OUT.mkdir(parents=True,exist_ok=True)
    prior=check(PRIOR);check(SCOPE)
    selection=json.loads(SELECTION.read_text())
    deps={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SELECTION,
        Path(__file__).parent/'build_npj_v6_comparison.py',
        Path(__file__).parent/'package_saved_npj_role_separated_review.py',
        Path(__file__).parent/'package_saved_npj_inference_v6_review.py',
        Path(__file__).parent/'probe_saved_npj_role_separated_postprocess.py',
        Path(__file__).parent/'probe_defect_context_preprocessing.py']}
    for folder in [PRIOR,SCOPE]:
        for name in ['manifest.json','complete.json']:
            p=folder/name;deps[str(p.relative_to(ROOT))]=sha(p)
    manifest=dict(settings=SETTINGS,cases=selection['cases'],dependencies=deps,source_sha256=sha(Path(__file__)))
    for c in selection['cases']:
        for k in ['input','saved_npj','reference_original']:
            deps[c[k]]=sha(ROOT/c[k])
        for k in ['red','extra','protect','hold']:
            p=PRIOR/'native'/c['id']/(k+'.png');deps[str(p.relative_to(ROOT))]=sha(p)
    dump(OUT/'manifest.json',manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    all_records=[];roi_records=[];figures=[]
    for c in selection['cases']:
        fid=c['id'];ref=read(ROOT/c['reference_original'])
        if c['reference_xywh']:
            x,y,w,h=c['reference_xywh'];ref=ref[y:y+h,x:x+w]
        observed=read(ROOT/c['input']);base=read(ROOT/c['saved_npj'])
        assert ref.shape==observed.shape==base.shape
        roles={k:read(PRIOR/'native'/fid/(k+'.png'),True) for k in ['red','extra','protect','hold']}
        mandatory=roles['red']|roles['extra']
        masks,diagnostic=infer(ref,base,mandatory)
        images=dict(reference=ref,saved_input=observed,saved_npj=base)
        solves={}
        for mode in ['margin','guarded']:
            update=masks[mode+'_update']
            images[mode],solves[mode]=harmonic(base,update)
            assert not (update&~masks['margin']).any()
            assert not (update&mandatory).any()
            assert np.array_equal(images[mode][~update],base[~update])
            poisoned=base.copy();poisoned[update]=[255,0,255]
            same,_=harmonic(poisoned,update);assert np.array_equal(same,images[mode])
        assert not (masks['guarded_update']&masks['narrow_guard']).any()
        if np.array_equal(ref,observed):observed_diag=diagnostic;observed_masks=masks
        else:observed_masks,observed_diag=infer(observed,base,mandatory)
        images['roles']=overlay(ref,[('edit',masks['guarded_update']),('hold',masks['narrow_guard']),('blue',~masks['margin'])])
        native=OUT/'native'/fid
        for k,a in images.items():write(native/((k+'_output' if k in ['margin','guarded'] else k)+'.png'),a)
        for k,m in masks.items():write(native/(k+'.png'),m)
        write(native/'saved_input_guarded_update.png',observed_masks['guarded_update'])
        changes={mode:int(np.any(images[mode]!=base,axis=2).sum()) for mode in ['margin','guarded']}
        overlaps={mode:{k:int((masks[mode+'_update']&roles[k]).sum()) for k in roles} for mode in ['margin','guarded']}
        diag=dict(id=fid,geometry=diagnostic,saved_input_geometry=observed_diag,
                  saved_input_vs_reference_update_xor=int((observed_masks['guarded_update']^masks['guarded_update']).sum()),
                  solves=solves,changed_pixels=changes,old_role_overlaps=overlaps,
                  baseline_exact_outside_update=True,all_inferred_nonmargin_exact_baseline=True)
        dump(native/'diagnostics.json',diag);all_records.append(diag)
        keys=['reference','saved_npj','margin','guarded']
        labels=['参照原画像','実際の保存npj','枠外の広い候補を補修','同じ候補＋細線保持']
        if not np.array_equal(ref,observed):
            fname='input_provenance_'+fid+'.png'
            figure([ref,observed,base],['参照原画像','実際の保存入力','実際の保存npj'],OUT/fname,
                c['page']+'：原画像と実入力を区別',
                [f'保存入力との差：平均 {np.abs(ref.astype(float)-observed).mean():.3f}/255。推定は左の原画像から行う。',
                 '原画像と保存入力は同位置。復元の基準は右の実保存npj。過去の変換条件は未再現。']);figures.append(fname)
        fname='overview_'+fid+'.png'
        figure([images[k] for k in keys],labels,OUT/fname,c['page']+'：枠外候補の限定試作',
               ['原画像から枠線と広い暗部を推定。枠内・朱候補・更新外は実保存npj一致。',
                f'更新：範囲のみ {masks["margin_update"].sum():,}／細線保持 {masks["guarded_update"].sum():,}画素。',
                '開発試作。枠外にも書き込み・折れ・紙模様があり得るため、材質の正解ではない。'],overview=True);figures.append(fname)
        for roi in c['rois']:
            x,y,w,h=roi['xywh'];s=np.s_[y:y+h,x:x+w];fname=roi['id']+'.png'
            counts={mode:int(masks[mode+'_update'][s].sum()) for mode in ['margin','guarded']}
            figure([images[k][s] for k in keys],labels,OUT/fname,roi['title']+'：枠外に限定',
                   [f'更新画素：範囲のみ {counts["margin"]:,}／細線保持 {counts["guarded"]:,}。文字側を再生成しない。',
                    '中央は実保存npj。原画像と保存入力が違う例は入力対応図を別に提示。',
                    '変化なしの対照は保持の確認で、汚れ除去の成功例として数えない。']);figures.append(fname)
            roi_records.append(dict(id=roi['id'],frame=fid,xywh=roi['xywh'],counts=counts))
        if masks['margin'].any():
            fname='roles_'+fid+'.png'
            figure([ref,images['roles']],['参照原画像','枠内保持・細線保持・候補'],OUT/fname,
                   c['page']+'：推定範囲と保持帯',
                   ['青＝推定余白以外を保持、紫＝原画像の細線保持、黄＝補修候補。GTではない。',
                    '枠線下8pxまで保持。枠内の汚れも維持される。余白の書き込みは細線だけでは保証できない。'],overview=True);figures.append(fname)
    for rel,digest in deps.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'diagnostics.json',dict(cases=all_records,regions=roi_records,figures=figures))
    dump(OUT/'verification.json',dict(status='passed',figures=len(figures),cases=len(all_records),
        whole_frame_inference=True,ROI_gating=False,outside_updates_saved_npj_exact=True,
        inferred_nonmargin_saved_npj_exact=True,mandatory_saved_npj_exact=True,
        guarded_original_narrow_band_saved_npj_exact=True,harmonic_poison_invariance=True,
        model_rerun=False,development=True,pixel_GT=False,adopted=False))
    print(json.dumps([dict(id=r['id'],rule=r['geometry']['chosen_rule'],counts=r['geometry']['counts'],changed=r['changed_pixels']) for r in all_records]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true')
    main(parser.parse_args().freeze)
