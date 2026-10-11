"""Fixed geometric guard ablation on known dirt and stroke regression cases.

Use the existing ridge response and thresholds without its neutral-colour gate.
An ablation keeps only ridge components touching fixed-v6 protect. All features
come from full originals; display regions never enter the feature function.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT,read,write,sha,dump,overlay
from finalize_npj_full_page_layout import render
from validate_npj_dirt_transfer import verified_bundle
import probe_defect_context_preprocessing as geometry
import probe_npj_contact_dirt_guard_v2 as reference
import probe_npj_dirt_clean_reference as paper

OUT=ROOT/'tmp_work/npj_colour_independent_ridge_20261004'
OLD=ROOT/'tmp_work/npj_contact_dirt_guard_v2_20261004'
NEW=ROOT/'tmp_work/npj_dirt_new_book_20261004'
VETO=ROOT/'tmp_work/npj_contact_component_veto_20261004'
FRAME=ROOT/'tmp_work/npj_full_page_layout_v4_20261004'
MODES=['all_colour_ridge','protect_anchored_ridge']


def full_geometry(original,protect):
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    drop=geometry.paper_estimate(lab)[:,:,0]-lab[:,:,0]
    response=geometry.ridge_response(lab[:,:,0])
    ridge=(response>=1.5)&(drop>=6)
    count,labels=cv2.connectedComponents(ridge.astype(np.uint8),connectivity=8)
    touch=np.bincount(labels.ravel(),weights=protect.ravel(),minlength=count)>0;touch[0]=False
    anchored=touch[labels]&ridge
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    all_guard=cv2.dilate(ridge.astype(np.uint8),kernel)>0
    anchor_guard=cv2.dilate(anchored.astype(np.uint8),kernel)>0
    assert not (anchor_guard&~all_guard).any()
    return dict(ridge=ridge,anchored_ridge=anchored,all_colour_ridge=all_guard,
                protect_anchored_ridge=anchor_guard)


def prepare():
    assert not OUT.exists(), 'Use a new output name'
    for folder in [OLD,NEW,VETO,FRAME]:verified_bundle(folder)
    old={c['id']:c for c in json.loads((OLD/'manifest.json').read_text())['cases']}
    new={c['id']:c for c in json.loads((NEW/'selection.json').read_text())['cases']}
    cases=json.loads((VETO/'manifest.json').read_text())['cases']
    deps={}
    for p in [Path(__file__),Path(geometry.__file__),Path(reference.__file__),Path(paper.__file__),
              *[d/'complete.json' for d in [OLD,NEW,VETO,FRAME]]]:deps[str(p.relative_to(ROOT))]=sha(p)
    for c in cases:
        source=(old if c['id'].startswith('old_') else new)[c['source_case']]
        c['original']=source['original']
        c['prior_applied']=None
        if c['has_prior']:
            p=(FRAME/'native'/c['source_case']/'applied.png') if c['id'].startswith('old_') else (ROOT/c['native']/'full_page_layout/applied.png')
            c['prior_applied']=str(p.relative_to(ROOT));deps[c['prior_applied']]=sha(p)
        for p in [ROOT/c['original'],ROOT/c['roles']/'protect.png']:
            deps[str(p.relative_to(ROOT))]=sha(p)
    OUT.mkdir()
    dump(OUT/'manifest.json',dict(cases=cases,dependencies=deps,settings=dict(
        geometry='Existing Hessian response sigmas1.5,3,6; response>=1.5; paper_L_drop>=6; halo4px',
        all_colour_ridge='Remove only the neutral delta_a<=4 and a<=6 restrictions from the ridge centres',
        protect_anchored_ridge='Keep 8-connected ridge-centre components touching fixed-v6 protect; then the same halo4',
        existing_guard='Always keep previous neutral-ridge/warm-network guard',
        update='Previous contact update minus added geometric guard',
        reference='Previous domain minus added geometric guard; same brightness/closing/ab4 conditions',
        renderer='Same known-only harmonic fill, no model inference',
        combination='Keep all prior full-page layout applied pixels; add contact result outside them'),
        adopted=False,independent_test=False,pixel_GT=False,all_cases_development=True,
        no_stroke_generation=True,thresholds_recorded_before_outputs=True))
    # Frozen witnesses, selected from previous reports before this ablation is executed.
    witnesses=[dict(id='old_saved_hnsd',name='inside_frame_residual',xywh=[250,240,350,150]),
               dict(id='old_saved_hnsd',name='left_writing_control',xywh=[30,50,160,260]),
               dict(id='old_wormhole',name='visible_defect_rim',xywh=[230,150,400,420])]
    for c in json.loads((NEW/'independent_verification.json').read_text())['all_combination_changed_components']:
        witnesses.append(dict(id='new_'+c['id'],name='previous_change_%02d'%c['component'],
                              xywh=c['display_xywh'],change_bbox=c['bbox']))
    for i,box in enumerate([[168,10,78,78],[295,182,75,75],[132,447,75,76],[0,647,72,82]],1):
        witnesses.append(dict(id='old_old_full_page',name='thin_control_%02d'%i,xywh=box))
    dump(OUT/'witnesses.json',dict(regions=witnesses,
         scope='Previously observed development controls, not independent or pixel material GT; no ROI supplied to geometry'))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')


def run():
    assert not (OUT/'run_complete.json').exists(), 'Already run'
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    cache={};records=[]
    for c in manifest['cases']:
        key=(c['original'],c['roles'])
        if key not in cache:
            original=read(ROOT/c['original']);protect=read(ROOT/c['roles']/'protect.png',True)
            assert protect.shape==original.shape[:2]
            cache[key]=len(cache)
            folder=OUT/'full_geometry'/str(cache[key])
            for name,a in full_geometry(original,protect).items():write(folder/(name+'.png'),a)
            del original,protect;gc.collect()
        full=OUT/'full_geometry'/str(cache[key]);x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
        source=ROOT/c['native'];n=OUT/'native'/c['id'];v=VETO/'native'/c['id']
        original=read(source/'original.png');base=read(source/'baseline.png');prior=read(v/'prior.png')
        assert np.array_equal(original,read(ROOT/c['original'])[sl])
        before=read(v/'previous_combined.png');veto=read(v/'protect_component_veto/repaired.png')
        old_update=read(source/c['contact_dir']/'update.png',True)
        dp=source/c['contact_dir']/'domain.png'
        domain=read(dp if dp.exists() else source/'domain.png',True)
        mandatory=read(source/'mandatory.png',True);old_guard=read(source/'guard.png',True)
        prior_applied=read(ROOT/c['prior_applied'],True) if c['prior_applied'] else np.zeros((h,w),bool)
        for k,a in dict(original=original,baseline=base,prior=prior,before=before,veto=veto,
                        old_update=old_update,old_domain=domain,mandatory=mandatory,
                        old_guard=old_guard,prior_applied=prior_applied).items():write(n/(k+'.png'),a)
        methods={};results={};L=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        for mode in MODES:
            guard=read(full/(mode+'.png'),True)[sl]
            update=old_update&~guard;ref_domain=domain&~guard
            known,t=reference.paper_reference(base,update,ref_domain,mandatory)
            repaired,applied,used,solves=paper.fill(base,update,known)
            poison=base.copy();poison[~known]=[255,0,255]
            trial,pa,pu,_=paper.fill(poison,update,known)
            assert np.array_equal(trial[applied],repaired[applied])
            assert np.array_equal(pa,applied) and np.array_equal(pu,used)
            assert np.array_equal(repaired[guard|old_guard|mandatory|~applied],base[guard|old_guard|mandatory|~applied])
            added=applied&~prior_applied;combined=prior.copy();combined[added]=repaired[added]
            assert np.array_equal(combined[prior_applied],prior[prior_applied])
            assert not (added&guard).any()
            for k,a in dict(guard=guard,update=update,domain=ref_domain,reference=known,
                            repaired=repaired,applied=applied,used_reference=used,
                            additional=added,combined=combined).items():write(n/mode/(k+'.png'),a)
            lab=cv2.cvtColor(combined.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            methods[mode]=dict(update=int(update.sum()),applied=int(applied.sum()),additional=int(added.sum()),
                additional_changed=int(np.any(combined!=prior,2).sum()),
                additional_dark_to_bright=int((added&(L<30)&(lab[:,:,0]>60)).sum()),
                additional_bright_green=int((added&(lab[:,:,0]>60)&(lab[:,:,1]<-8)).sum()),
                reference_L_threshold=t,solves=solves,poison_invariant=True,
                old_guard_and_mandatory_exact=True,prior_layout_exact=True)
            results[mode]=combined
        render(OUT/(c['id']+'_comparison.png'),[original,base,before,veto,*results.values()],
            ['原画像','同じnpj基準','前回の線保持','前回：保護接触なら見送る','今回：色を問わず線保持','今回：保護に接続する線保持'],
            c['page']+'：本文形状による保持と汚れ残留の開発比較',
            ['全原ページから固定条件で形状を算出。追加補修にだけ適用し、前回枠外補修を維持。',
             '既知17範囲。新しいモデル推論・筆画生成なし。保持や黒減少は汚れ精度ではない。'],overview=True)
        records.append(dict(id=c['id'],page=c['page'],full_geometry=str(full.relative_to(OUT)),methods=methods))
        print(c['id'],{k:(d['additional'],d['additional_dark_to_bright']) for k,d in methods.items()},flush=True)
    dump(OUT/'diagnostics.json',dict(cases=records))
    witnesses=json.loads((OUT/'witnesses.json').read_text())['regions'];metrics=[]
    for witness in witnesses:
        n=OUT/'native'/witness['id'];x,y,w,h=witness['xywh'];cut=np.s_[y:y+h,x:x+w]
        arrays=[read(n/(k+'.png')) for k in ['original','baseline','before','veto']]
        arrays.extend(read(n/m/'combined.png') for m in MODES)
        assert x>=0 and y>=0 and x+w<=arrays[0].shape[1] and y+h<=arrays[0].shape[0]
        name=witness['id']+'_'+witness['name']+'.png'
        render(OUT/name,[a[cut] for a in arrays],
            ['原画像','同じnpj基準','前回の線保持','成分の見送り','色を問わず線保持','保護に接続する線保持'],
            witness['id']+'：'+witness['name'],
            ['前回から固定した汚れ候補・本文対照。色補正なし、元画素を整数倍表示。',
             '選定済みの開発範囲で、画素単位の材質GTではない。'])
        prior=read(n/'prior.png')
        metrics.append(dict(**witness,filename=name,additional_changes={m:int(np.any(read(n/m/'combined.png')[cut]!=prior[cut],2).sum()) for m in MODES}))
    dump(OUT/'witness_metrics.json',dict(regions=metrics))
    dump(OUT/'run_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*.png')},
        full_original_feature_sets=len(cache),cases=len(records),comparison_png=len(list(OUT.glob('*.png')))))


def freeze():
    assert not (OUT/'complete.json').exists(), 'Already frozen'
    assert (OUT/'report.md').exists(), 'Visual review required before freeze'
    manifest=json.loads((OUT/'manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(OUT/'source_snapshot.py')==sha(Path(__file__))
    for rel,digest in json.loads((OUT/'run_complete.json').read_text())['outputs'].items():assert sha(OUT/rel)==digest,rel
    figures=sorted(OUT.glob('*.png'))
    with zipfile.ZipFile(OUT/'comparison_png.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in figures:z.write(p,p.name)
    with zipfile.ZipFile(OUT/'comparison_png.zip') as z:
        assert sorted(z.namelist())==sorted(p.name for p in figures)
        for p in figures:assert z.read(p.name)==p.read_bytes()
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        comparison_png_count=len(figures),adopted=False,independent_test=False,pixel_GT=False))
    print('Frozen',len(figures),'comparison PNG',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','freeze'])
    {'prepare':prepare,'run':run,'freeze':freeze}[parser.parse_args().stage]()
