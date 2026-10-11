"""Connect two frozen mask expansions to the unchanged16-pair paper restorer."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_full_upper_strong_masks_v2 as mask
import restore_anchored_pale_margin as previous
import prepare_full_upper_component_baselines_v2 as extras
import restore_unseen_thin_margin_paper as fresh3
import restore_unseen_dark_margin_paper as fresh4
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=mask.ROOT
OUT=ROOT/'tmp_work/full_upper_strong_restoration_v2_20261001'
fill=previous.fill
MODES=mask.MODES
SETTINGS=dict(mask_modes=MODES,minimum_boundary_pairs=16,component_minimum_area=64,
    reference='saved pale6 known-paper qualification minus new full-upper edits',
    restorer='same raw/smooth harmonic color, boundary8px and observed fine L seams',
    old14_fixed_baseline_elsewhere='saved DDRM unchanged',other_baseline_elsewhere='original unchanged',
    fallback='previous output retained when new eligible component is not solved',
    recent_guidance_chroma_coarse_and_pair8=False,pages=25,ROIs=73,
    fixed_original_ROIs=59,additional_old_component_ROIs=8, additional_upper_component_ROIs=6,
    development=True,pixel_GT=False,new_DDRM=False)


def regions():
    values=[]
    for row in json.loads((previous.OUT/'diagnostics.json').read_text()):
        values.append({**row,'baseline_local':str((previous.OUT/'native'/row['id']/'drop6').relative_to(ROOT))})
    values.extend(json.loads((extras.OUT/'diagnostics.json').read_text()))
    assert len(values)==73 and len({r['id'] for r in values})==73
    return values


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [mask.OUT,previous.OUT,extras.OUT]:check_bundle(folder)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [mask.OUT,previous.OUT,extras.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for mod in [fill,fill.frozen,fill.frozen.prior,fill.frozen.prior.prior]:
        dependencies[str(Path(mod.__file__).relative_to(ROOT))]=sha(Path(mod.__file__))
    inputs=mask.pale.prior.descriptors();by_page={r['page']['page']:r for r in inputs};input_rows=regions()
    for r in inputs:dependencies[r['page']['original']]=sha(ROOT/r['page']['original'])
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[];poison=[]
    for page_key,desc in by_page.items():
        original=read(ROOT/desc['page']['original']);page_folder=mask.OUT/'pages'/page_key
        edits={k:read(page_folder/k/'edit.png',True) for k in MODES}
        additions={k:read(page_folder/k/'added.png',True) for k in MODES}
        for old in [r for r in input_rows if r['page']==page_key]:
            rid=old['id'];x,y,w,h=old['xywh'];cx,cy,cw,ch=old['context_xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
            context=original[cs];crop=context[s];lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            baseline_folder=ROOT/old['baseline_local'];base=read(baseline_folder/'output.png')
            known=read(baseline_folder/'reference_context.png',True);active=read(baseline_folder/'eligible_context.png',True)
            old_applied=read(baseline_folder/'retained_applied_context.png',True)
            outputs={};stats={};cache=[];dest=OUT/'native'/rid
            for level in MODES:
                name=level;md=dest/name;added=additions[level][cs];new_known=known&~added;new_active=active|added
                recompute=bool(added.any() and new_active[s].any())
                reused=next((d for d in cache if np.array_equal(d['known'],new_known) and np.array_equal(d['active'],new_active)),None)
                if recompute:
                    if reused is None:
                        data=fill.field_for(lab,new_known,new_active,16)
                        cache.append(dict(known=new_known,active=new_active,data=data))
                    else:data=reused['data']
                    fields,applied,used,notes,field,positions,qnotes,seam,average,sx,sy=data
                    value=fill.frozen.compose(field,applied,base,s)
                    md.mkdir(parents=True,exist_ok=True);np.savez_compressed(md/'harmonic_fields_Lab.npz',**fields)
                    for label,item in dict(result_field_Lab=field,texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():
                        np.save(md/(label+'.npy'),item,allow_pickle=False)
                    dump(md/'donors.json',dict(context_xywh=old['context_xywh'],patch_xy=positions,components=qnotes))
                    if applied.any() and reused is None:
                        poisoned=lab.copy();poisoned[~new_known]=(240,-220,132)
                        repeat=fill.field_for(poisoned,new_known,new_active,16)
                        assert np.array_equal(field,repeat[4]) and np.array_equal(applied,repeat[1])
                        poison.append(dict(id=rid,mode=name))
                else:
                    value=base.copy();applied=old_applied.copy();used=np.zeros_like(known);notes=[];positions=[]
                assert not np.any(np.any(value!=base,axis=2)&~applied[s])
                assert not np.any(np.any(value!=crop,axis=2)&~edits[level][cs][s])
                assert not np.any(applied&~new_active) and not np.any(new_known&new_active)
                retained=old_applied|applied
                for label,item in dict(output=value,edit=edits[level][cs][s],added=added[s],reference_context=new_known,
                    eligible_context=new_active,applied_context=applied,retained_applied_context=retained,used_context=used).items():
                    write(md/(label+'.png'),item)
                dump(md/'components.json',notes);outputs[name]=value
                stats[name]=dict(added_edit=int(added[s].sum()),filled_added=int((applied[s]&added[s]).sum()),
                    newly_filled=int((applied[s]&~old_applied[s]).sum()),paper_applied=int(retained[s].sum()),
                    changed_vs_base=int(np.any(value!=base,axis=2).sum()),recomputed=recompute,
                    solver_applied=int(applied[s].sum()),donor_patches=len(positions),reused=reused is not None)
            for label,item in dict(original=crop,baseline=base,baseline_applied_context=old_applied,
                baseline_reference_context=known,baseline_eligible_context=active).items():write(dest/(label+'.png'),item)
            elsewhere='旧14ページの固定ROI：紙面外は保存DDRM。' if old['base_elsewhere']=='saved_DDRM' else '紙面外・未適用画素は原画像。'
            figure([crop,base]+[outputs[k] for k in MODES],
                ['原画像','前回の差6回収＋紙面','上方強コア＋同じ紙面','横長の新コア除外＋紙面'],
                OUT/'comparisons'/(rid+'.png'),old['title']+' · 旧帯より上の強コアを追加回収',
                '同じ16境界組・紙色・観測明度模様。編集範囲を拡張し参照／場を再計算。',
                elsewhere+' 新しいDDRM・色／模様／境界組の変更なし。')
            row={**old,'full_upper_diagnostics':stats};rows.append(row)
            dump(OUT/'progress.json',dict(completed=len(rows),total=73,latest=rid))
            print('Upper strong paper:',len(rows),'/73',rid,{k:v['filled_added'] for k,v in stats.items()},flush=True)
        del original,edits,additions;gc.collect()
    assert len(rows)==73
    for p,d in dependencies.items():assert sha(ROOT/p)==d
    dump(OUT/'diagnostics.json',rows)
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=73,old_base_kept_on_solver_failure=True,
        outside_edit_changes=0,all_nonreference_poison_cases=poison,minimum_boundary_pairs=16,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.py','.npy','.npz'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete:73 ROI paper comparisons for full-upper strong cores',flush=True)


if __name__=='__main__':main()
