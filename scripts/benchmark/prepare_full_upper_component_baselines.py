"""Paper-only baselines for every new unanchored strong component, including ROI-external page edges."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_full_upper_strong_masks_v2 as masks
import restore_anchored_pale_margin as previous
import restore_unseen_margin_paper as first4
import restore_unseen_dark_margin_paper as last4
from build_npj_v6_comparison import dump,read,sha,write
from package_paper_blending_review import check_bundle

ROOT=masks.ROOT
OUT=ROOT/'tmp_work/full_upper_component_baselines_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [masks.OUT,previous.OUT,first4.OUT,last4.OUT]:check_bundle(folder)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [masks.OUT,previous.OUT,first4.OUT,last4.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=dependencies,
        fixed_full_page_mask='previous pale6',ROIs='all five new unanchored accepted strong components, 45px border/64px context',
        reference='same qualified full-page paper minus pale6 additions',restorer='same16-pair harmonic and observed fine L seams',
        outside_applied='original',GT=False,selection_after_mask_outputs=True)
    dump(OUT/'manifest.json',manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    desc={r['page']['page']:r for r in masks.pale.prior.descriptors()};rows=[];poison=[]
    for page in json.loads((masks.OUT/'diagnostics.json').read_text())['pages']:
        key=page['page'];components=[n for n in page['strong_components'] if n['accepted'] and n['old_core_overlap']==0]
        if not components:continue
        pg=desc[key]['page'];original=read(ROOT/pg['original']);H,W=original.shape[:2];folder=masks.OUT/'pages'/key
        edit=read(folder/'base_edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
        if key=='200021802_00015_1':
            known=read(first4.OUT/'reference_pages'/key/'rim_paper.png',True)&~edit
        else:
            assert key=='200014685_00002_2'
            known=read(last4.OUT/'reference_pages'/key/'near8_paper.png',True)&~edit
        active=edit&domain&~guard
        assert not np.any(known&active)
        for n in components:
            x,y,w,h=n['xywh'];rx,ry=max(0,x-45),max(0,y-45);rw,rh=min(W,x+w+45)-rx,min(H,y+h+45)-ry
            cx,cy=max(0,rx-64),max(0,ry-64);cw,ch=min(W,rx+rw+64)-cx,min(H,ry+rh+64)-cy
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(ry-cy,ry-cy+rh),slice(rx-cx,rx-cx+rw))
            context=original[cs];crop=context[s];lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            local_known=known[cs];local_active=active[cs]
            data=previous.fill.field_for(lab,local_known,local_active,16)
            fields,applied,used,notes,field,positions,qnotes,seam,average,sx,sy=data
            value=previous.fill.frozen.compose(field,applied,crop,s)
            rid=key+'_upper_core_'+str(n['id']);dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
            assert not np.any(np.any(value!=crop,axis=2)&~applied[s])
            for label,item in dict(original=crop,output=value,edit=edit[cs][s],reference_context=local_known,
                eligible_context=local_active,applied_context=applied,retained_applied_context=applied,used_context=used).items():write(dest/(label+'.png'),item)
            np.savez_compressed(dest/'harmonic_fields_Lab.npz',**fields)
            for label,item in dict(result_field_Lab=field,texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():np.save(dest/(label+'.npy'),item,allow_pickle=False)
            dump(dest/'donors.json',dict(context_xywh=[cx,cy,cw,ch],patch_xy=positions,components=qnotes));dump(dest/'components.json',notes)
            if applied.any():
                changed=lab.copy();changed[~local_known]=(240,-220,132)
                repeat=previous.fill.field_for(changed,local_known,local_active,16)
                assert np.array_equal(field,repeat[4]) and np.array_equal(applied,repeat[1]);poison.append(rid)
            rows.append(dict(id=rid,page=key,title='追加強コア '+str(n['id'])+'：未判定の成分を全て比較',
                xywh=[rx,ry,rw,rh],context_xywh=[cx,cy,cw,ch],fixed_ROI=False,component=n,
                base_elsewhere='original',baseline_local=str(dest.relative_to(ROOT)),selection_after_mask_outputs=True))
            print('Extra full-upper baseline:',rid,flush=True)
        del original,edit,domain,guard,known,active;gc.collect()
    assert len(rows)==5
    for p,d in dependencies.items():assert sha(ROOT/p)==d
    dump(OUT/'diagnostics.json',rows);dump(OUT/'verification.json',dict(status='passed',all_new_components=5,
        original_elsewhere=True,nonreference_poison_cases=poison,GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.py','.npz','.npy'] and p.name not in ['manifest.json','complete.json']}))


if __name__=='__main__':main()
