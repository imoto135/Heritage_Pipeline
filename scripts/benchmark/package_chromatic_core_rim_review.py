"""Audit two fixed mask extensions, field provenance, and all77 comparisons."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import restore_chromatic_core_rims as prior
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_seam_texture_review import verify_texture
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/chromatic_core_rim_review_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.mask.OUT);check_bundle(prior.OUT);check_bundle(prior.extras.OUT)
    masks=json.loads((prior.mask.OUT/'diagnostics.json').read_text())
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(rows)==77
    originals={r['page']['page']:r['page']['original'] for r in prior.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    mask_checks=[];field_checks=[];texture=[];checked=[];unchanged={m:[] for m in prior.MODES}
    for page in masks['pages']:
        key=page['page'];folder=prior.mask.OUT/'pages'/key
        base=read(folder/'base_edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
        covered=np.zeros_like(base);previous_coverage=np.zeros_like(base)
        for row in [r for r in rows if r['page']==key]:
            x,y,w,h=row['xywh'];covered[y:y+h,x:x+w]=True
            if '_external_rim_core_' not in row['id']:previous_coverage[y:y+h,x:x+w]=True
        assert np.array_equal(previous_coverage,read(folder/'existing_ROI_coverage.png',True))
        outside=read(folder/'all6/added.png',True)&~previous_coverage
        assert np.array_equal(outside,read(folder/'outside_existing_ROIs.png',True))
        assert int(outside.sum())==page['outside_existing_ROIs']
        for level in prior.MODES:
            added=read(folder/level/'added.png',True);edit=read(folder/level/'edit.png',True)
            assert np.array_equal(edit,base|added) and not np.any(base&added) and not np.any(added&(guard|~domain))
            assert int(added.sum())==page['additions'][level]
            assert not np.any(added&~covered)
            core=read(folder/'core.png',True);strong=read(folder/'strong_core.png',True)
            seed=core if level=='all6' else strong
            assert not np.any(strong&~core)
            assert np.array_equal(seed,read(folder/level/'source_core.png',True))
            warm=read(prior.mask.diagnostic.OUT/'pages'/key/'warm_support.png',True)
            near=cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
            assert np.array_equal(added,near&~base&domain&~guard&warm)
            assert not np.any(read(folder/'strong6/added.png',True)&~read(folder/'all6/added.png',True))
            mask_checks.append(dict(page=key,level=level,additional_pixels=int(added.sum())))
    for key,path in originals.items():
        original=read(ROOT/path);page_folder=prior.mask.OUT/'pages'/key
        for row in [r for r in rows if r['page']==key]:
            rid=row['id'];native=prior.OUT/'native'/rid;old=ROOT/row['baseline_local'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            crop=original[y:y+h,x:x+w];base=read(native/'baseline.png');known=read(native/'baseline_reference_context.png',True)
            active=read(native/'baseline_eligible_context.png',True);old_applied=read(native/'baseline_applied_context.png',True)
            assert np.array_equal(read(native/'original.png'),crop) and np.array_equal(base,read(old/'output.png'))
            for label,value in [('reference_context',known),('eligible_context',active),('retained_applied_context',old_applied)]:
                assert np.array_equal(value,read(old/(label+'.png'),True))
            lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            if '_external_rim_core_' in rid:
                # The four newly cropped baselines did not exist in older verification bundles.
                fields=np.load(old/'harmonic_fields_Lab.npz',allow_pickle=False)
                if old_applied.any():
                    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
                    smooth=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                    for name,boundary in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                        error,references=verify_field(fields[name],old_applied,known,boundary)
                        assert np.array_equal(references,read(old/'used_context.png',True))
                        field_checks.append(dict(id=rid,mode='new_component_baseline',field=name,residual_max=error))
                    donor=json.loads((old/'donors.json').read_text())
                    if donor['components']:texture.append(dict(id=rid,mode='new_component_baseline',**verify_texture(lab,known,old_applied,old)))
                    else:assert len(donor['patch_xy'])<4 and not np.load(old/'texture_seam.npy',allow_pickle=False).any()
                distance=cv2.distanceTransform(old_applied.astype(np.uint8),cv2.DIST_L2,5)
                weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*old_applied
                combined=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                combined[:,:,0]+=np.load(old/'texture_seam.npy',allow_pickle=False)*np.clip((distance-1)/4.,0,1)*old_applied
                assert np.array_equal(combined,np.load(old/'result_field_Lab.npy',allow_pickle=False))
                assert np.array_equal(base,prior.fill.frozen.compose(combined,old_applied,crop,s))
            for level in prior.MODES:
                mode=level;md=native/mode;stat=row['chromatic_rim_diagnostics'][mode]
                new_known=read(md/'reference_context.png',True);new_active=read(md/'eligible_context.png',True)
                applied=read(md/'applied_context.png',True);retained=read(md/'retained_applied_context.png',True)
                used=read(md/'used_context.png',True);value=read(md/'output.png');edit=read(md/'edit.png',True);added=read(md/'added.png',True)
                whole_added=read(page_folder/mode/'added.png',True)[cs]
                assert np.array_equal(added,whole_added[s])
                assert np.array_equal(new_known,known&~whole_added) and np.array_equal(new_active,active|whole_added)
                assert np.array_equal(retained,old_applied|applied)
                assert not np.any(applied&~new_active) and not np.any(new_known&new_active)
                assert not np.any(np.any(value!=base,axis=2)&~applied[s])
                assert not np.any(np.any(value!=crop,axis=2)&~edit)
                assert stat['added_edit']==int(added.sum()) and stat['filled_added']==int((applied[s]&added).sum())
                assert stat['paper_applied']==int(retained[s].sum()) and stat['newly_filled']==int((applied[s]&~old_applied[s]).sum())
                assert stat['changed_vs_base']==int(np.any(value!=base,axis=2).sum())
                if not stat['changed_vs_base']:unchanged[level].append(rid)
                if stat['recomputed']:
                    fields=np.load(md/'harmonic_fields_Lab.npz',allow_pickle=False)
                    if applied.any():
                        mass=cv2.GaussianBlur(new_known.astype(np.float32),(0,0),6)
                        smooth=cv2.GaussianBlur(np.where(new_known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                        for name,boundary in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                            error,references=verify_field(fields[name],applied,new_known,boundary)
                            assert np.array_equal(references,used)
                            field_checks.append(dict(id=rid,mode=mode,field=name,residual_max=error))
                        notes=json.loads((md/'donors.json').read_text())
                        if notes['components']:texture.append(dict(id=rid,mode=mode,**verify_texture(lab,new_known,applied,md)))
                        else:
                            assert len(notes['patch_xy'])<4 and not np.load(md/'texture_seam.npy',allow_pickle=False).any()
                            assert np.all(np.load(md/'source_x.npy',allow_pickle=False)==-1)
                    distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
                    weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
                    combined=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                    combined[:,:,0]+=np.load(md/'texture_seam.npy',allow_pickle=False)*np.clip((distance-1)/4.,0,1)*applied
                    assert np.array_equal(combined,np.load(md/'result_field_Lab.npy',allow_pickle=False))
                    assert np.array_equal(value,prior.fill.frozen.compose(combined,applied,base,s))
                else:
                    assert not whole_added.any() or not new_active[s].any()
                    assert np.array_equal(value,base) and np.array_equal(applied,old_applied) and not used.any()
                checked.append(dict(id=rid,mode=mode,**stat))
    for folder,name in [(prior.mask.OUT/'comparisons','masks'),(prior.OUT/'comparisons','paper')]:
        (OUT/name).mkdir(exist_ok=True)
        for p in folder.glob('*.png'):shutil.copyfile(p,OUT/name/p.name)
    focus=['hole_edge_new','new_margin_hole','extra_second_branch','extra_top_branch','unseen_thin_crack','unseen_fold','extra_isolated_marks','new_brown_stamp','extra_header_strokes']
    for i,rid in enumerate(focus,1):
        stack([OUT/'masks'/(rid+'.png'),OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',
              rid+' · 暖色輪郭6px：追加源と同じ復元器',
              ['上：黄＝前回、青＝暖色外縁の追加。下：原画像・前回・全コア6px・強コア6px。',
               '紙端の反例も保存。原画像・同じ復元器で比較。一律採用はしない。'])
    # The additional old components have no fixed mask ROI PNG; paper comparisons expose all of them.
    ambiguous=[r for r in rows if r['page']=='200021712_00082_1' and '_component_' in r['id']]
    stack([OUT/'paper'/(r['id']+'.png') for r in ambiguous],OUT/'10_unknown_old_components.png',
          '原因不明の筆画状成分も全5つ保持して比較',
          ['全コアと強コアの周辺追加を比較。重複する候補数を合計しない。',
           '前回から削る5曲線自体の正誤は未確定。損傷の正解には数えない。'])
    extra=[r for r in rows if '_external_rim_core_' in r['id']]
    assert len(extra)==4
    stack([OUT/'paper'/(r['id']+'.png') for r in extra],OUT/'11_all_external_components.png',
          '固定ROI外の追加を生じた全4成分',
          ['固定ROI外の197画素を含む追加源を全て保存。原画像・同じ紙面補間で比較。',
           '追加の切り出しは出力確認後の開発診断。画素GTではなく重複数も合計しない。'])
    proof=dict(status='passed',pages=25,ROIs=77,variants=154,mask_checks=mask_checks,all_whole_page_additions_covered=True,
               direct_field_residuals=field_checks,texture_provenance=texture,regions=checked,
               unchanged_ROIs={str(k):v for k,v in unchanged.items()},outside_edit_changes=0,
               all_prior_filled_pixels_retained_on_solver_failure=True,pixel_GT=False,new_DDRM=False)
    dump(OUT/'verification.json',proof)
    notes_path=ROOT/'tmp_work/chromatic_core_rim_review_notes_20261001.md'
    report=notes_path.read_text()
    (OUT/'report.md').write_text(report)
    pngs=sorted(OUT.rglob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs) and all(n.endswith('.png') for n in z.namelist())
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str(notes_path.relative_to(ROOT)):sha(notes_path), **{str((folder/name).relative_to(ROOT)):sha(folder/name) for folder in [prior.mask.OUT,prior.OUT,prior.extras.OUT] for name in ['manifest.json','complete.json']}},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered chromatic6px rims:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
