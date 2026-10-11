"""Cross frozen chromatic rim6 and colour-reference gap3, with holds fixed."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_observed_absolute_paper_patches as prior
import probe_chromatic_core_rims as rims
import probe_paper_colour_virtual_reference_gap as gap
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/chromatic_rim_reference_gap_factorial_pilot_20261001'
MODES=['current','rim6','gap3','rim6_gap3']
SETTINGS=dict(pages=29,ROIs=90,mask='frozen chromatic all-core6, source core minus fixed direction hold, one pass',
    added='saved all6 added AND within frozen6 ellipse of unheld original core AND outside original hold',
    guards='saved chromatic domain and ink/warm guards unchanged; no ROI gating',
    hold='same original full-page hold; do not use held core as new rim source',
    baseline='same current effective applied and known-paper, independently full-float recomputation verified',
    rim_restorer='same16-pair raw/smooth colour plus old24px fine L quilt; new known=old known minus added',
    retain='previous physical applied/field retained when rim re-solve unsupported',
    gap3='same virtual gap3 colour restorer on the physical applied retained after that mask; this geometry fixed for the gap arm',
    fine_texture='each mask uses the same fineL before/after gap; old-field fallback disclosed',
    gap3_control='no-rim arm reuses already verified frozen gap3 output',
    new4='previous full-page accepted core0: no new chromatic additions',
    absolute_patches_or_energy_combined=False,new_DDRM=False,pixel_GT=False,development=True)


def colour(fields,app):
    weight=np.exp(-(np.maximum(cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)-1,0)/8.)**2)*app
    return fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [prior.OUT,rims.OUT,rims.diagnostic.OUT,gap.OUT]:check_bundle(folder)
    parity=ROOT/'tmp_work/directional_paper_restorer_recomputation_20261001';check_bundle(parity)
    pv=json.loads((parity/'verification.json').read_text());assert pv['all_BRG_outputs_equal'] and pv['all_float_fields_equal'] and pv['all_applied_equal']
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());deps={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in [__file__,rims.__file__,prior.fill.__file__]}
    for folder in [prior.OUT,rims.OUT,rims.diagnostic.OUT,gap.OUT,parity]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];page_records=[];poison=[]
    for key in dict.fromkeys(r['page'] for r in rows):
        group=[r for r in rows if r['page']==key];original=read(ROOT/group[0]['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        md=rims.OUT/'pages'/key;pd=OUT/'pages'/key;pd.mkdir(parents=True,exist_ok=True)
        if md.exists():
            old=rims.diagnostic.OUT/'pages'/key;seed=read(old/'core.png',True)
            hold=read(prior.diagnostic.previous.OUT/'pages'/key/'paper_hold.png',True)
            old_added=read(md/'all6/added.png',True);unheld_seed=seed&~hold
            near=cv2.dilate(unheld_seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
            added=old_added&near&~hold
            effective=read(prior.diagnostic.previous.OUT/'pages'/key/'effective_edit.png',True)
            assert not np.any(added&(hold|effective))
            for n,a in dict(unheld_source_core=unheld_seed,paper_hold=hold,added=added,effective_edit=effective|added).items():write(pd/(n+'.png'),a)
        else:
            added=np.zeros(original.shape[:2],bool);hold=added.copy();old_added=added.copy();unheld_seed=added.copy();write(pd/'added.png',added)
            dump(pd/'no_addition.json',dict(reason='new4 fixed candidate had whole-page accepted core0',page=key))
        covered=np.zeros_like(added)
        for row in group:
            x,y,w,h=row['xywh'];covered[y:y+h,x:x+w]=True
        assert not np.any(added&~covered)
        page_records.append(dict(page=key,original_added=int(old_added.sum()),held_source_filtered_added=int(added.sum()),
            hold_pixels=int(hold.sum()),unheld_core_pixels=int(unheld_seed.sum()),outside_ROIs=0))
        crop=original.copy();crop[added]=(255,120,0);crop[hold]=(190,50,190)
        figure([original,crop],['原画像','青＝暖色追加、紫＝固定保留'],OUT/'mask_comparisons'/(key+'.png'),key+' · 全ページから一段で縁を回収',
            f'追加{int(added.sum())}画素。保留のコアを追加源へ使わない。切り出しの座標は推定へ渡さない。',
            '追加は損傷GTではない。新4冊は元の受理コア0。保留は元画素を維持する。')
        for row in group:
            rid=row['id'];source=prior.OUT/'native'/rid;dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
            x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            lab=full_lab[cs];crop=read(source/'original.png');base=read(source/'baseline.png');app=read(source/'applied_context.png',True);known=read(source/'reference_context.png',True)
            add=added[cs];local_hold=hold[cs];new_known=known&~add;active=app|add
            gap_only=read(gap.OUT/'native'/rid/'virtual3/output.png')
            outputs=dict(current=base,rim6=base.copy(),gap3=gap_only,rim6_gap3=gap_only.copy())
            physical=app.copy();accepted=np.zeros_like(app);used=np.zeros_like(app);virtual=app.copy();ref=known.copy();gapaccepted=np.zeros_like(app);actual=np.zeros_like(app);gapused=np.zeros_like(app)
            notes=[];gapnotes=[];positions=[];qnotes=[];recomputed=bool(add.any() and (app[s].any() or add[s].any()))
            if recomputed:
                strict=prior.fill.field_for(lab,known,app,16);assert np.array_equal(strict[1],app)
                basefield=strict[4];basecolour=colour(strict[0],app)
                assert np.array_equal(prior.fill.frozen.compose(basefield,app,base,s),base)
                data=prior.fill.field_for(lab,new_known,active,16)
                fields,accepted,used,notes,newfield,positions,qnotes,seam,average,sx,sy=data
                assert not np.any(accepted&~active) and not np.any(accepted&local_hold)
                rimcolour=basecolour.copy();rimcolour[accepted]=colour(fields,accepted)[accepted]
                rimfield=basefield.copy();rimfield[accepted]=newfield[accepted];physical=app|accepted
                outputs['rim6']=prior.fill.frozen.compose(rimfield,accepted,base,s)
                outside=cv2.distanceTransform((~physical).astype(np.uint8),cv2.DIST_L2,5)
                virtual=physical|(new_known&(outside<=3));ref=new_known&~virtual
                gf,gapaccepted,gapused,gapnotes=prior.fill.solve_fields(lab,ref,virtual,16);actual=gapaccepted&physical
                gapcolour=colour(gf,physical);delta=np.zeros_like(rimfield);delta[actual]=gapcolour[actual]-rimcolour[actual];combined=rimfield+delta
                outputs['rim6_gap3']=prior.fill.frozen.compose(combined,actual,outputs['rim6'],s)
                bad=lab.copy();bad[~new_known]=(240,-210,140);repeat=prior.fill.field_for(bad,new_known,active,16)
                assert np.array_equal(newfield,repeat[4]) and np.array_equal(accepted,repeat[1])
                bad=lab.copy();bad[~ref]=(240,-210,140);repeat=prior.fill.solve_fields(bad,ref,virtual,16)
                for n in gf:assert np.array_equal(gf[n],repeat[0][n])
                poison.append(rid)
                np.savez_compressed(dest/'rim_harmonic_fields_Lab.npz',**fields);np.savez_compressed(dest/'gap_harmonic_fields_Lab.npz',**gf)
                for n,a in dict(baseline_field_Lab=basefield,baseline_colour_Lab=basecolour,rim_computed_field_Lab=newfield,
                    rim_field_Lab=rimfield,rim_colour_Lab=rimcolour,gap_colour_Lab=gapcolour,gap_delta_Lab=delta,combined_field_Lab=combined,
                    texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():np.save(dest/(n+'.npy'),a,allow_pickle=False)
                dump(dest/'donors.json',dict(patch_xy=positions,components=qnotes))
                assert np.array_equal(outputs['rim6'][~physical[s]],base[~physical[s]])
                assert np.array_equal(outputs['rim6_gap3'][~actual[s]],outputs['rim6'][~actual[s]])
            else:
                # No new ROI effect; preserve the frozen no-rim colour arm too.
                ref=read(gap.OUT/'native'/rid/'virtual3/colour_reference_context.png',True)
                virtual=read(gap.OUT/'native'/rid/'virtual3/virtual_context.png',True)
                actual=read(gap.OUT/'native'/rid/'virtual3/physical_supported_context.png',True)
            for n,a in dict(original=crop,baseline=base,original_applied_context=app,original_reference_context=known,added_context=add,
                paper_hold_context=local_hold,rim_reference_context=new_known,rim_active_context=active,rim_accepted_context=accepted,
                retained_physical_context=physical,rim_used_context=used,gap_virtual_context=virtual,gap_reference_context=ref,
                gap_accepted_context=gapaccepted,gap_physical_supported_context=actual,gap_used_context=gapused,**outputs).items():write(dest/(n+'.png'),a)
            dump(dest/'rim_components.json',notes);dump(dest/'gap_components.json',gapnotes)
            for m in MODES:assert np.array_equal(outputs[m][local_hold[s]],crop[local_hold[s]])
            stats={m:prior.metrics(im,base,physical,known,s) for m,im in outputs.items()}
            figure([crop]+[outputs[m] for m in MODES],['原画像','現行・輪郭0／色gap0','輪郭6のみ・保留源除外','紙色gap3のみ','輪郭6＋紙色gap3'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 輪郭と紙色参照の組み合わせ',
                f'追加{int(add[s].sum())}／追加補間{int((accepted&add)[s].sum())}画素。各マスクの模様はgap前後で固定。',
                '方向保留・本文対照を維持。模様量補正は併用なし。追加数や色変更を修復成功と数えない。')
            results.append({**row,'recomputed_rim':recomputed,'added_ROI':int(add[s].sum()),'filled_added_ROI':int((accepted&add)[s].sum()),
                'rim_fallback_old_ROI':int((app&~accepted)[s].sum()) if recomputed else 0,
                'gap_fallback_ROI':int((physical&~actual)[s].sum()),'retained_physical_ROI':int(physical[s].sum()),'factorial_diagnostics':stats})
            dump(OUT/'progress.json',dict(completed=len(results),total=90,latest=rid));print('Rim/gap factorial:',len(results),'/90',rid,int(add[s].sum()),[stats[m]['changed_vs_base'] for m in MODES],flush=True)
        del original,full_lab,hold,added,crop;gc.collect()
    assert len(results)==90 and len(page_records)==29
    dump(OUT/'diagnostics.json',dict(pages=page_records,regions=results))
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,all_whole_page_additions_covered=True,
        held_core_not_used_as_rim_source=True,held_original_pixels_exact=True,current_BRG_control_exact=True,
        rim_and_colour_nonreference_poison_invariance=poison,fine_texture_fixed_within_each_mask_gap_pair=True,new_DDRM=False,pixel_GT=False))
    for p,h in deps.items():assert sha(ROOT/p)==h
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: chromatic rim6 / colour gap3 factorial90 ROIs',flush=True)


if __name__=='__main__':main()
