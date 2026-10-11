"""Fixed ridge guard on another saved page and new ranges of known pages.

Saved full-page outputs only. Selection precedes policy output. No fresh DDRM,
no threshold tuning, and no claim of independent books or material ground truth.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump
from finalize_npj_full_page_layout import render
from validate_npj_dirt_transfer import verified_bundle
from probe_saved_npj_role_separated_postprocess import frame_guard
import probe_npj_contact_dirt_guard as contact
import probe_npj_contact_dirt_guard_v2 as reference
import probe_npj_colour_independent_ridge_guard as ridge
import probe_npj_full_page_layout as broad
import probe_npj_full_page_layout_v4 as layout
import probe_npj_dirt_clean_reference as paper

OUT=ROOT/'tmp_work/npj_ridge_saved_oldfull_20261004'
SPEC=[('oldfull_upper_band','200018243_00003_2',[1100,300,768,768],'saved_black_frame_band_and_writing_control')]


def prepare():
    assert not OUT.exists(), 'Use a new output name'
    verified_bundle(ridge.OUT)
    literature=json.loads((ROOT/'tmp_work/npj_literature_trial_20261002/selection.json').read_text())
    saved={c['page']:c for c in literature['frames'] if c['input_reference_exact'] and
           (c['box'] is None or c['box']==[0,0,c['shape'][1],c['shape'][0]])}
    inventory={r['page']:r for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
    new_page='200018243_00002_2'
    saved[new_page]=dict(input='tmp_work/bt_200018243_00002_2/input/'+new_page+'.jpg',
        saved_npj='tmp_work/bt_200018243_00002_2/ddrm_extref_out/'+new_page+'/result/'+new_page+'.png',
        roles=str((OUT/'classification/v6_masks'/new_page).relative_to(ROOT)),
        baseline_label='Actual saved external-reference local npj v2; historical source snapshot unavailable')
    cases=[];deps={}
    for cid,page,box,role in SPEC:
        c=dict(inventory[page]);c.update(saved[page]);c.update(id=cid,page=page,xywh=box,role=role,
            rois=[],contains_illustration=False,new_page_for_ridge=page==new_page)
        original=read(ROOT/c['original']);actual=read(ROOT/c['input']);base=read(ROOT/c['saved_npj'])
        assert np.array_equal(original,actual),cid
        assert base.shape==original.shape,cid
        x,y,w,h=box;assert x>=0 and y>=0 and x+w<=original.shape[1] and y+h<=original.shape[0]
        for k in ['original','input','saved_npj','npj_red']:
            deps[c[k]]=sha(ROOT/c[k])
        if page!=new_page:
            for k in ['edit','protect','hold','red','extra']:
                p=ROOT/c['roles']/(k+'.png');assert read(p,True).shape==original.shape[:2]
                deps[str(p.relative_to(ROOT))]=sha(p)
        cases.append(c)
    for p in [Path(__file__),Path(contact.__file__),Path(reference.__file__),Path(ridge.__file__),
              Path(broad.__file__),Path(layout.__file__),Path(paper.__file__),ridge.OUT/'complete.json',
              ROOT/'tmp_work/bt_200018243_00002_2/run_4patterns.sh',
              ROOT/'tmp_work/bt_200018243_00002_2/log_ddrm_extref.txt']:
        deps[str(p.relative_to(ROOT))]=sha(p)
    OUT.mkdir()
    dump(OUT/'selection.json',dict(cases=cases,dependencies=deps,
        selection_before_policy_output=True,all_actual_saved_full_page_outputs=True,
        all_actual_inputs_match_original=True,no_new_diffusion_inference=True,
        one_new_page_three_known_pages=True,new_books=0,pixel_GT=False,adopted=False,
        novelty='New page for this fixed ridge policy plus new ranges in known pages; prior studies may include them',
        excluded_sources={'trial_run':'actual input differs from original',
            'fidelity_pilot':'different masks/recipe and synthetic intensity variants; not pooled with these actual saved baselines'}))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    # Original/saved-baseline sheets are screening evidence, before any tested policy result.
    for c in cases:
        x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        render(OUT/(c['id']+'_baseline.png'),[read(ROOT/c['original'])[sl],read(ROOT/c['saved_npj'])[sl]],
            ['原画像','実保存npj・全ページ出力から同位置'],c['page']+'：固定条件の適用前',
            ['実推論入力と原画像は全画素一致。原画像と保存結果の同じ範囲。',
             '別ページ1枚と既知3ページの別範囲。汚れの画素GTはない。'],overview=True)


def classify():
    import compare_saved_npj_v6 as classifier
    assert not (OUT/'classification/complete.json').exists(), 'Already classified'
    cases=json.loads((OUT/'selection.json').read_text())['cases']
    row=next(c for c in cases if c['new_page_for_ridge'])
    classifier.OUT=OUT/'classification';classifier.render=lambda:None
    dump(classifier.OUT/'baseline_review.json',dict(cases=[row],scope='One writing-only original page; fixed v6'))
    classifier.compute()
    dump(classifier.OUT/'complete.json',dict(outputs={str(p.relative_to(classifier.OUT)):sha(p) for p in classifier.OUT.rglob('*.png')},thresholds_unchanged=True))


def run():
    assert not (OUT/'run_complete.json').exists(), 'Already processed'
    manifest=json.loads((OUT/'selection.json').read_text())
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    verified_bundle(OUT/'classification')
    cache={};records=[]
    for c in manifest['cases']:
        if c['page'] not in cache:
            original=read(ROOT/c['original'])
            full_roles={k:read(ROOT/c['roles']/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
            f,_=contact.features(original,full_roles);geom,diag=layout.detect_frame(original)
            new_guard=ridge.full_geometry(original,full_roles['protect'])
            cache[c['page']]=(f,geom,diag,new_guard)
        f,geom,diag,new_guard=cache[c['page']]
        x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        original=read(ROOT/c['original'])[sl];base=read(ROOT/c['saved_npj'])[sl]
        roles={k:read(ROOT/c['roles']/(k+'.png'),True)[sl] for k in ['edit','protect','hold','red','extra']}
        mandatory=roles['red']|roles['extra']|(cv2.imread(str(ROOT/c['npj_red']),0)[sl]<127)
        L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        candidate,_=broad.select_update(base,f['drop'][sl],f['b'][sl],f['footprint'][sl],mandatory)
        residual=cv2.dilate((L<30).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
        old_update=candidate&residual&~f['guard'][sl]&~mandatory&frame_guard((h,w),2)
        domain=f['footprint'][sl]&~f['guard'][sl]&frame_guard((h,w),2)
        frame_update,_=broad.select_update(base,f['drop'][sl],f['b'][sl],geom['exterior'][sl],mandatory)
        keep=roles['protect']|roles['hold']|mandatory
        keepguard=cv2.dilate(keep.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
        v6update=roles['edit']&~keepguard&frame_guard((h,w),1)
        modes=dict(frame=(frame_update,geom['exterior'][sl],paper.paper_reference),
            v6=(v6update,~keep&frame_guard((h,w),1),paper.paper_reference),
            old_contact=(old_update,domain,reference.paper_reference))
        for mode in ridge.MODES:
            guard=new_guard[mode][sl];modes[mode]=(old_update&~guard,domain&~guard,reference.paper_reference)
        n=OUT/'native'/c['id'];results={};applied_maps={};metrics={}
        for k,a in dict(original=original,baseline=base,mandatory=mandatory,
                        old_guard=f['guard'][sl],frame_interior=geom['interior'][sl],
                        all_colour_ridge=new_guard['all_colour_ridge'][sl],
                        protect_anchored_ridge=new_guard['protect_anchored_ridge'][sl]).items():write(n/(k+'.png'),a)
        for mode,(update,ref_domain,fn) in modes.items():
            known,t=fn(base,update,ref_domain,mandatory)
            result,applied,used,solves=paper.fill(base,update,known)
            poison=base.copy();poison[~known]=[255,0,255]
            trial,pa,pu,_=paper.fill(poison,update,known)
            assert np.array_equal(trial[applied],result[applied]) and np.array_equal(pa,applied) and np.array_equal(pu,used)
            assert np.array_equal(result[~applied|mandatory],base[~applied|mandatory])
            for k,a in dict(update=update,domain=ref_domain,reference=known,repaired=result,
                            applied=applied,used_reference=used).items():write(n/mode/(k+'.png'),a)
            results[mode]=result;applied_maps[mode]=applied
            metrics[mode]=dict(update=int(update.sum()),applied=int(applied.sum()),
                changed=int(np.any(result!=base,2).sum()),reference_L_threshold=t,
                poison_invariant=True,solves=solves)
        for mode in ['old_contact',*ridge.MODES]:
            added=applied_maps[mode]&~applied_maps['frame']
            combined=results['frame'].copy();combined[added]=results[mode][added]
            assert np.array_equal(combined[applied_maps['frame']],results['frame'][applied_maps['frame']])
            lab=cv2.cvtColor(combined.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            metrics[mode].update(additional=int(added.sum()),
                additional_changed=int(np.any(combined!=results['frame'],2).sum()),
                additional_dark_to_bright=int((added&(L<30)&(lab[:,:,0]>60)).sum()),
                additional_bright_green=int((added&(lab[:,:,0]>60)&(lab[:,:,1]<-8)).sum()))
            write(n/mode/'combined.png',combined);write(n/mode/'additional.png',added)
        render(OUT/(c['id']+'_comparison.png'),[original,base,results['v6'],
            *[read(n/m/'combined.png') for m in ['old_contact',*ridge.MODES]]],
            ['原画像','実保存npj','固定v6紙面補修','従来の線保持＋枠外','色を問わず線保持＋枠外','保護接続線の保持＋枠外'],
            c['page']+'：固定した形状条件を別範囲へ',
            ['保存済み全ページ出力を使用。追加推論・文字生成なし。条件変更なし。',
             '汚れの材質GTはない。変更0も黒画素減少も精度と数えない。'],overview=True)
        records.append(dict(id=c['id'],page=c['page'],role=c['role'],layout=diag,methods=metrics))
        print(c['id'],{m:(d.get('additional',d['applied']),d['changed']) for m,d in metrics.items()},flush=True)
    dump(OUT/'diagnostics.json',dict(cases=records))
    dump(OUT/'run_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.png')},
        no_new_diffusion=True,conditions_unchanged=True))


def freeze():
    assert not (OUT/'complete.json').exists()
    assert (OUT/'report.md').exists() and (OUT/'verification.json').exists()
    selection=json.loads((OUT/'selection.json').read_text())
    for rel,digest in selection['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(OUT/'source_snapshot.py')==sha(Path(__file__))
    for rel,digest in json.loads((OUT/'run_complete.json').read_text())['outputs'].items():assert sha(OUT/rel)==digest,rel
    figures=sorted(OUT.glob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_png.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in figures:z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_png.zip') as z:
        assert sorted(z.namelist())==sorted(p.name for p in figures)
        for p in figures:assert z.read(p.name)==p.read_bytes()
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        comparison_png_count=len(figures),adopted=False,independent_books=False,pixel_GT=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','classify','run','freeze'])
    {'prepare':prepare,'classify':classify,'run':run,'freeze':freeze}[parser.parse_args().stage]()
