"""Paper-only frozen baselines for every chromatic rim outside previous ROIs."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_chromatic_core_rims as mask
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=mask.ROOT
OUT=ROOT/'tmp_work/chromatic_external_baselines_20261001'
prior=mask.diagnostic.prior
fill=prior.fill


def full_reference(desc,current_edit):
    key=desc['page']['page'];source=desc['source'];thin=prior.mask.pale.prior.anchor.prior.mask
    if source in thin.prior.MARGIN.SOURCES:path=fill.frozen.prior.prior.OUT/'reference_pages'/key/'paper.png'
    elif source==thin.new.INPUT:path=fill.new.OUT/'reference_pages'/key/'rim_paper.png'
    elif source==prior.fresh3.INPUT:path=prior.fresh3.OUT/'reference_pages'/key/'near8_paper.png'
    elif source==prior.fresh4.INPUT:path=prior.fresh4.OUT/'reference_pages'/key/'near8_paper.png'
    else:raise AssertionError(('Unknown reference source',source))
    return read(path,True)&~current_edit,path


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(mask.OUT);check_bundle(prior.OUT)
    rows=json.loads((mask.OUT/'diagnostics.json').read_text())['external_regions']
    inputs={r['page']['page']:r for r in prior.mask.pale.prior.descriptors()}
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [mask.OUT,prior.OUT]:
        for name in ['manifest.json','complete.json']:deps[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for mod in [fill,fill.frozen,fill.frozen.prior,fill.frozen.prior.prior]:deps[str(Path(mod.__file__).relative_to(ROOT))]=sha(Path(mod.__file__))
    for row in rows:
        deps[row['original']]=sha(ROOT/row['original'])
        desc=inputs[row['page']];edit=read(mask.OUT/'pages'/row['page']/'base_edit.png',True)
        _,path=full_reference(desc,edit);deps[str(path.relative_to(ROOT))]=sha(path)
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,settings=dict(
        baseline='same frozen full_upper short mask; pure-paper only, original elsewhere',
        additional_ROIs=len(rows),selected_after_mask_outputs=True,pixel_GT=False,minimum_boundary_pairs=16,new_DDRM=False))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];current=None;poison=[]
    for row in rows:
        key=row['page'];rid=row['id'];desc=inputs[key]
        if current!=key:
            original=read(ROOT/row['original']);current=key;folder=mask.OUT/'pages'/key
            edit=read(folder/'base_edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
            known_full,_=full_reference(desc,edit);active_full=edit&domain&~guard
        x,y,w,h=row['xywh'];cx,cy=max(0,x-64),max(0,y-64)
        cw,ch=min(original.shape[1],x+w+64)-cx,min(original.shape[0],y+h+64)-cy
        cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w));context=original[cs];crop=context[s]
        known=known_full[cs];active=active_full[cs];lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        fields,applied,used,notes,field,positions,qnotes,seam,average,sx,sy=fill.field_for(lab,known,active,16)
        value=fill.frozen.compose(field,applied,crop,s);dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
        assert not np.any(np.any(value!=crop,axis=2)&~edit[cs][s])
        for name,item in dict(output=value,original=crop,reference_context=known,eligible_context=active,
            applied_context=applied,retained_applied_context=applied,used_context=used,edit=edit[cs][s]).items():write(dest/(name+'.png'),item)
        np.savez_compressed(dest/'harmonic_fields_Lab.npz',**fields)
        for name,item in dict(result_field_Lab=field,texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():np.save(dest/(name+'.npy'),item,allow_pickle=False)
        dump(dest/'donors.json',dict(context_xywh=[cx,cy,cw,ch],patch_xy=positions,components=qnotes));dump(dest/'components.json',notes)
        if applied.any():
            bad=lab.copy();bad[~known]=(230,-180,130);repeat=fill.field_for(bad,known,active,16)
            assert np.array_equal(field,repeat[4]) and np.array_equal(applied,repeat[1]);poison.append(rid)
        figure([crop,value],['原画像','固定マスク・従来紙面の基準'],OUT/'comparisons'/(rid+'.png'),
            row['title']+' · 候補ROI外も基準から保存','元の16組・紙色・観測細かいL模様。新しい暖色外縁はまだ追加していない。',
            '未補間・紙面適用外は原画像。原因未確定の成分で、損傷GTではない。')
        results.append({**row,'context_xywh':[cx,cy,cw,ch],'baseline_local':str(dest.relative_to(ROOT)),
            'fixed_ROI':False,'base_elsewhere':'original','baseline_applied':int(applied[s].sum())})
        print('External chromatic baseline:',len(results),'/',len(rows),rid,int(applied[s].sum()),flush=True);gc.collect()
    for path,value in deps.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',additional_ROIs=len(rows),all_nonreference_poison_cases=poison,outside_edit_changes=0,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:external chromatic baselines',len(rows),flush=True)


if __name__=='__main__':main()
