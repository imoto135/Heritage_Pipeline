"""Audit all rim sources, both solvers, fine-L provenance and fixed boundaries."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_chromatic_rim_reference_gap_factorial as prior
from build_npj_v6_comparison import dump,read,sha
from package_known_paper_fill_review import verify_field
from package_raw_qualified_paper_colour_review import support_check
from package_paper_seam_texture_review import verify_texture
from package_paper_colour_virtual_reference_gap_review import diagnostic
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/chromatic_rim_reference_gap_factorial_review_20261001'


def operator_check(lab,known,accepted,used,fields):
    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
    smooth=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
    errors=[]
    for name,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
        error,expected=verify_field(fields[name],accepted,known,target);assert np.array_equal(expected,used);errors.append(error)
    return errors


def colour(fields,app):
    distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)
    weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*app
    return fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]


def composite(field,applied,base,s):
    converted=cv2.cvtColor(field[s].astype(np.float32),cv2.COLOR_LAB2BGR)
    converted=np.rint(np.clip(converted,0,1)*255).astype(np.uint8)
    expected=base.copy();expected[applied[s]]=converted[applied[s]];return expected


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT);data=json.loads((prior.OUT/'diagnostics.json').read_text());rows=data['regions'];checked=[];pages=[]
    unchanged={m:[] for m in prior.MODES};current=None
    for page in data['pages']:
        key=page['page'];pd=prior.OUT/'pages'/key;md=prior.rims.OUT/'pages'/key
        group=[r for r in rows if r['page']==key];original=read(ROOT/group[0]['original_source'])
        full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);added=read(pd/'added.png',True)
        if md.exists():
            core=read(prior.rims.diagnostic.OUT/'pages'/key/'core.png',True);hold=read(prior.prior.diagnostic.previous.OUT/'pages'/key/'paper_hold.png',True)
            oldadded=read(md/'all6/added.png',True);seed=core&~hold
            near=cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
            assert np.array_equal(added,oldadded&near&~hold) and np.array_equal(seed,read(pd/'unheld_source_core.png',True))
            effective=read(prior.prior.diagnostic.previous.OUT/'pages'/key/'effective_edit.png',True)
            assert np.array_equal(effective|added,read(pd/'effective_edit.png',True)) and not np.any(added&(hold|effective))
        else:
            hold=np.zeros_like(added);assert not added.any()
        covered=np.zeros_like(added)
        for row in group:
            x,y,w,h=row['xywh'];covered[y:y+h,x:x+w]=True
        assert not np.any(added&~covered) and page['held_source_filtered_added']==int(added.sum())
        pages.append(page)
        for row in group:
            rid=row['id'];native=prior.OUT/'native'/rid;source=prior.prior.OUT/'native'/rid
            x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w));lab=full_lab[cs]
            base=read(native/'baseline.png');app=read(native/'original_applied_context.png',True);known=read(native/'original_reference_context.png',True)
            add=read(native/'added_context.png',True);local_hold=read(native/'paper_hold_context.png',True)
            newknown=read(native/'rim_reference_context.png',True);active=read(native/'rim_active_context.png',True);accepted=read(native/'rim_accepted_context.png',True)
            physical=read(native/'retained_physical_context.png',True);used=read(native/'rim_used_context.png',True)
            virtual=read(native/'gap_virtual_context.png',True);ref=read(native/'gap_reference_context.png',True);gapaccepted=read(native/'gap_accepted_context.png',True)
            actual=read(native/'gap_physical_supported_context.png',True);gapused=read(native/'gap_used_context.png',True)
            assert np.array_equal(base,read(source/'baseline.png')) and np.array_equal(read(native/'original.png'),original[y:y+h,x:x+w])
            assert np.array_equal(app,read(source/'applied_context.png',True)) and np.array_equal(known,read(source/'reference_context.png',True))
            assert np.array_equal(add,added[cs]) and np.array_equal(local_hold,hold[cs]) and np.array_equal(active,app|add) and np.array_equal(newknown,known&~add)
            assert not np.any(newknown&active) and not np.any(physical&local_hold)
            assert row['added_ROI']==int(add[s].sum()) and row['retained_physical_ROI']==int(physical[s].sum())
            proof={};residuals=[];rim_support=[];gap_support=[]
            if row['recomputed_rim']:
                assert np.array_equal(physical,app|accepted) and not np.any(accepted&~active)
                rim_support=support_check(active,newknown,accepted)
                fields=np.load(native/'rim_harmonic_fields_Lab.npz',allow_pickle=False);residuals+=operator_check(lab,newknown,accepted,used,fields)
                metadata=json.loads((native/'donors.json').read_text());seam=np.load(native/'texture_seam.npy',allow_pickle=False)
                if len(metadata['patch_xy'])>=4:proof=verify_texture(lab,newknown,accepted,native)
                else:
                    assert not metadata['components'] and not seam.any()
                    assert (np.load(native/'source_x.npy',allow_pickle=False)==-1).all() and (np.load(native/'source_y.npy',allow_pickle=False)==-1).all()
                    proof=dict(donor_fallback=True,qualified_patches=len(metadata['patch_xy']))
                expected_new=colour(fields,accepted)
                expected_new[:,:,0]+=seam*np.clip((cv2.distanceTransform(accepted.astype(np.uint8),cv2.DIST_L2,5)-1)/4.,0,1)*accepted
                assert np.array_equal(expected_new,np.load(native/'rim_computed_field_Lab.npy',allow_pickle=False))
                baseline=np.load(native/'baseline_field_Lab.npy',allow_pickle=False);basecolour=np.load(native/'baseline_colour_Lab.npy',allow_pickle=False)
                if (source/'baseline_field_Lab.npy').exists():assert np.array_equal(baseline[app],np.load(source/'baseline_field_Lab.npy',allow_pickle=False)[app])
                assert np.array_equal(composite(baseline,app,base,s),base)
                expected_field=baseline.copy();expected_field[accepted]=expected_new[accepted]
                rimfield=np.load(native/'rim_field_Lab.npy',allow_pickle=False);assert np.array_equal(expected_field,rimfield)
                rimcolour=basecolour.copy();rimcolour[accepted]=colour(fields,accepted)[accepted]
                assert np.array_equal(rimcolour,np.load(native/'rim_colour_Lab.npy',allow_pickle=False))
                rimvalue=read(native/'rim6.png');assert np.array_equal(rimvalue,composite(rimfield,accepted,base,s))
                outside=cv2.distanceTransform((~physical).astype(np.uint8),cv2.DIST_L2,5)
                assert np.array_equal(virtual,physical|(newknown&(outside<=3))) and np.array_equal(ref,newknown&~virtual)
                assert np.array_equal(actual,gapaccepted&physical);gap_support=support_check(virtual,ref,gapaccepted)
                gf=np.load(native/'gap_harmonic_fields_Lab.npz',allow_pickle=False);residuals+=operator_check(lab,ref,gapaccepted,gapused,gf)
                gapcolour=colour(gf,physical);assert np.array_equal(gapcolour,np.load(native/'gap_colour_Lab.npy',allow_pickle=False))
                delta=np.zeros_like(rimfield);delta[actual]=gapcolour[actual]-rimcolour[actual];combined=rimfield+delta
                assert np.array_equal(delta,np.load(native/'gap_delta_Lab.npy',allow_pickle=False)) and np.array_equal(combined,np.load(native/'combined_field_Lab.npy',allow_pickle=False))
                assert np.array_equal(read(native/'rim6_gap3.png'),composite(combined,actual,rimvalue,s))
                assert np.allclose((combined[:,:,0]-gapcolour[:,:,0])[actual],(rimfield[:,:,0]-rimcolour[:,:,0])[actual],atol=2e-5,rtol=0)
                assert row['rim_fallback_old_ROI']==int((app&~accepted)[s].sum())
            else:
                assert not accepted.any() and not add[s].any()
                assert np.array_equal(physical,app) and np.array_equal(read(native/'rim6.png'),base)
                assert np.array_equal(read(native/'rim6_gap3.png'),read(native/'gap3.png'))
                # Unused numerical arrays are placeholders; the complete frozen gap3 arm supplies this output.
                for name,a in [('colour_reference_context',ref),('virtual_context',virtual),('physical_supported_context',actual)]:
                    assert np.array_equal(a,read(prior.gap.OUT/'native'/rid/'virtual3'/(name+'.png'),True))
            assert row['filled_added_ROI']==int((accepted&add)[s].sum()) and row['gap_fallback_ROI']==int((physical&~actual)[s].sum())
            kept_reference=newknown&~physical;assert not np.any(kept_reference&physical)
            measured={};overlap=int((known&physical).sum())
            for mode in prior.MODES:
                value=read(native/(mode+'.png'));assert np.array_equal(value[local_hold[s]],original[y:y+h,x:x+w][local_hold[s]])
                assert np.array_equal(value[~physical[s]],base[~physical[s]])
                changed=int(np.any(value!=base,axis=2).sum());assert changed==row['factorial_diagnostics'][mode]['changed_vs_base']
                if not changed:unchanged[mode].append(rid)
                measured[mode]=dict(changed_vs_base=changed,**diagnostic(value,base,physical,kept_reference,s))
            assert np.array_equal(read(native/'current.png'),base) and np.array_equal(read(native/'gap3.png'),read(prior.gap.OUT/'native'/rid/'virtual3/output.png'))
            checked.append(dict(id=rid,recomputed=row['recomputed_rim'],added_ROI=row['added_ROI'],filled_added_ROI=row['filled_added_ROI'],
                physical_ROI=row['retained_physical_ROI'],gap_fallback_ROI=row['gap_fallback_ROI'],texture=proof,residuals=residuals,
                rim_support=rim_support,gap_support=gap_support,pilot_old_reference_overlap_physical_context=overlap,
                evaluation_reference='old known minus rim added minus retained physical; disjoint and same for4 arms',fixed_boundary_diagnostics=measured))
            print('Audit rim/gap:',len(checked),'/90',rid,'old ref overlap',overlap,flush=True)
    for folder in ['comparisons','mask_comparisons']:
        (OUT/folder).mkdir(exist_ok=True)
        for p in (prior.OUT/folder).glob('*.png'):shutil.copyfile(p,OUT/folder/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','extra_brown_stain','new_brown_stamp','unseen_fold','200021712_00082_1_external_rim_core_10']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 輪郭と色参照を分けた4条件',
            ['固定方向保留、元基準／暖色輪郭6px／色参照gap3px／両方。','補修が薄くなること・輪郭回収・境界色差を別に読む。修復精度ではない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,conditions=360,page_masks=pages,records=checked,unchanged_ROIs=unchanged,
        full_page_rim_sources_holds_and_coverage_reconstructed=True,independent_harmonic_and_texture_provenance=True,
        compositing_and_colour_delta_exact=True,same_fineL_within_mask_gap_pair=True,all_held_original_pixels_exact=True,
        fixed_disjoint_reference_boundary_measurements=True,pilot_boundary_metrics_original_reference_overlap_disclosed=True,
        new_DDRM=False,pixel_GT=False))
    notes=ROOT/'tmp_work/chromatic_rim_reference_gap_factorial_review_notes_20261001.md'
    if not notes.exists():print('Rim/gap audit ready; create notes',flush=True);return
    (OUT/'report.md').write_text(notes.read_text());pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==128
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==128
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=128,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered chromatic rim/gap factorial:128 PNGs /360 conditions',flush=True)


if __name__=='__main__':main()
