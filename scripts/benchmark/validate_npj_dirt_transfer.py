"""Fixed-condition transfer of dirt-only restoration to additional books.

Restore only where an actual npj image is available. Missing control crops are
inferred with the frozen Oct 2 local npj source, not labelled historical output.
No stroke repair, model training, threshold search or illustration pages.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from build_npj_v6_comparison import ROOT, read, write, sha, dump, overlay
from package_saved_npj_role_separated_review import figure
import probe_npj_dirt_clean_reference as paper
import probe_saved_npj_ruled_margin_v2 as geometry
from probe_saved_npj_role_separated_postprocess import harmonic, frame_guard

OUT = ROOT / 'tmp_work/npj_dirt_transfer_20261004'
FROZEN = ROOT / 'tmp_work/npj_dirt_control_20261002'
SPECS = [
    dict(id='fresh_lower_paper', page='100249537_00003_2', xywh=[520,2250,768,768],
         reason='Lower writing frame and brown paper; stain material unconfirmed',
         role='paper_pattern_and_text_control', new_v6_book=False, compute_v6_here=True),
    dict(id='fresh_binding_mark', page='200021851_00024_1', xywh=[1732,900,768,768],
         reason='Dark mark on binding margin adjacent to writing frame',
         role='damage_candidate_material_unconfirmed', new_v6_book=False, compute_v6_here=True),
    dict(id='wormhole', page='200021712_00082_1', xywh=[1070,0,768,768],
         reason='Visibly missing paper and dark rim in upper margin',
         role='visible_paper_loss', new_v6_book=False,
         cache='tmp_work/npj_failure_collection_20261001'),
    dict(id='seal_control', page='200021644_00003_2', xywh=[1235,0,768,768],
         reason='Upper red seal and writing; preserve npj output',
         role='red_seal_and_text_control', new_v6_book=False,
         cache='tmp_work/unseen_thin_margin_validation_20261001'),
    dict(id='fresh_printed_text', page='200008003_00004_2', xywh=[850,2400,768,768],
         reason='Clear black printed text as a preservation control',
         role='text_control', new_v6_book=True),
]


def verified_bundle(folder):
    m = json.loads((folder/'complete.json').read_text())
    for p, digest in m['outputs'].items():
        assert sha(folder/p) == digest, p
    return m


def prepare():
    assert not (OUT/'selection.json').exists(), 'Selection already recorded'
    verified_bundle(FROZEN)
    verified_bundle(paper.OUT)
    rows = {a['page']:a for a in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
    selected = []
    dependencies = {}
    for s in SPECS:
        row = rows[s['page']]
        original = read(ROOT/row['original'])
        x,y,w,h = s['xywh']; crop=original[y:y+h,x:x+w]
        assert crop.shape == (h,w,3)
        item = dict({**row, **s}, contains_illustration=False,
                    novelty='All frames new to clean-reference renderer; only new_v6_book=true is a book absent from earlier fixed-v6 manifests/reviews',
                    baseline_kind='Fresh frozen-code crop control, not a historical saved page')
        selected.append(item)
        write(OUT/'input'/(s['id']+'.png'), crop)
        for key,name in [('npj_black','02_text_black.png'),('npj_red','03_text_red.png')]:
            p=ROOT/row[key]; mask=cv2.imread(str(p),0)[y:y+h,x:x+w]
            assert mask.shape == (h,w)
            write(OUT/'masks'/s['id']/name,mask)
            assert np.array_equal(cv2.imread(str(OUT/'masks'/s['id']/name),0),mask)
        for key in ['original','npj_input','npj_black','npj_red','npj_degradation','npj_background']:
            dependencies[row[key]]=sha(ROOT/row[key])
    for p in [Path(__file__), FROZEN/'complete.json', FROZEN/'manifest.json', paper.OUT/'complete.json',
              Path(paper.__file__),Path(geometry.__file__),ROOT/'scripts/benchmark/package_saved_npj_role_separated_review.py']:
        dependencies[str(p.relative_to(ROOT))]=sha(p)
    dump(OUT/'selection.json',dict(cases=selected,dependencies=dependencies,
         selection_before_inference=True, original_screening='original_screening.png',
         illustration_review='All five full original pages reviewed as text pages; no illustration',
         baseline_scope='Native 768px crop controls using full-page stored CSC v1 crop; not whole-page DDRM',
         no_text_repair=True, threshold_changes=False))
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    print('Selection recorded:', len(selected), 'frames /', len({c['book'] for c in selected}), 'books', flush=True)


def infer():
    selection=json.loads((OUT/'selection.json').read_text())
    assert not (OUT/'inference_complete.json').exists(), 'Already inferred'
    verified_bundle(FROZEN)
    snapshot=FROZEN/'source_snapshot/ddrm_npj_v2'
    old=json.loads((FROZEN/'manifest.json').read_text())
    command=[sys.executable,'-B',str(snapshot/'scripts/pipeline_ddrm.py'),
             '--input_dir',str(OUT/'input'),'--output_dir',str(OUT/'npj_output'),
             '--mask_dir',str(OUT/'masks'),'--ckpt',str(ROOT/'modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth'),
             '--back_ground',str(ROOT/'modules/ddrm_npj_v2/background/isemonogatari.png'),
             '--device','cuda:0','--patch_size','128','--overlap','32','--no_wandb']
    for rel in ['modules/stain_removal/experiments/model_char/model_char/ckpt_epoch0350.pth',
                'modules/ddrm_npj_v2/background/isemonogatari.png']:
        assert sha(ROOT/rel)==old['dependencies'][rel]
    dump(OUT/'inference_command.json',dict(command=command, seed=999, timesteps=20,masking=True,
         mask_timesteps=50,shared_noise=False, source= str(snapshot.relative_to(ROOT)),
         source_bundle_sha256=sha(FROZEN/'complete.json')))
    with (OUT/'inference.log').open('w') as f:
        subprocess.run(command,cwd=snapshot,stdout=f,stderr=subprocess.STDOUT,check=True)
    products={}
    for c in selection['cases']:
        p=OUT/'npj_output'/c['id']/'result'/(c['id']+'.png')
        a=read(p);assert a.shape==read(OUT/'input'/(c['id']+'.png')).shape
        products[str(p.relative_to(OUT))]=sha(p)
    dump(OUT/'inference_complete.json',dict(outputs=products))
    print('Verified',len(selection['cases']),'fresh npj crop controls',flush=True)


def classify():
    import compare_saved_npj_v6 as classifier
    selection=json.loads((OUT/'selection.json').read_text())
    target=OUT/'classification'
    assert not (target/'complete.json').exists(), 'Already classified'
    new=[c for c in selection['cases'] if c.get('compute_v6_here',c['new_v6_book'])]
    # The frozen classifier accepts full original pages, never these display crops.
    dump(target/'baseline_review.json',dict(cases=[dict(c,rois=[]) for c in new],
         selection_protocol='Original-only full-page review; no restoration output used to select'))
    classifier.OUT=target
    classifier.render=lambda: None  # This stage produces masks only; comparison PNGs are authored below.
    classifier.compute()
    for c in selection['cases']:
        if c.get('compute_v6_here',c['new_v6_book']):continue
        old=ROOT/c['cache']; expected=json.loads((old/'comparison_complete.json').read_text())['outputs']
        oldcases=json.loads((old/'baseline_review.json').read_text())['cases']
        recorded=next(a for a in oldcases if a['page']==c['page'])
        assert recorded['original']==c['original']
        cm=json.loads((old/'comparison_manifest.json').read_text())
        assert sha(ROOT/c['original'])==cm['dependencies'][c['original']]
        for key in ['edit','protect','hold','red','extra']:
            p=old/'v6_masks'/c['page']/(key+'.png')
            assert sha(p)==expected[str(p.relative_to(old))]
            write(target/'v6_masks'/c['page']/(key+'.png'),read(p,True))
    dump(target/'complete.json',dict(outputs={str(p.relative_to(target)):sha(p)
         for p in sorted(target.rglob('*.png'))},thresholds_unchanged=True,full_original_scope=True))
    print('Full-page frozen v6 ready for',len(selection['cases']),'pages',flush=True)


def compare():
    assert not (OUT/'comparison_complete.json').exists(), 'Comparison already generated'
    selected=json.loads((OUT/'selection.json').read_text())
    verified_bundle(OUT/'classification')
    for rel,h in selected['dependencies'].items():assert sha(ROOT/rel)==h,rel
    rows=[]
    for c in selected['cases']:
        fid=c['id']; original=read(OUT/'input'/(fid+'.png'))
        base=read(OUT/'npj_output'/fid/'result'/(fid+'.png'))
        x,y,w,h=c['xywh'];s=np.s_[y:y+h,x:x+w]
        roles={k:read(OUT/'classification/v6_masks'/c['page']/(k+'.png'),True)[s]
               for k in ['edit','hold','protect','red','extra']}
        stored_red=cv2.imread(str(OUT/'masks'/fid/'03_text_red.png'),0)<127
        keep=roles['hold']|roles['protect']|roles['red']|roles['extra']|stored_red
        guard=cv2.dilate(keep.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
        v6update=roles['edit']&~guard&frame_guard(base.shape[:2],1)
        v6naive,_=harmonic(base,v6update)
        known,threshold=paper.paper_reference(base,v6update,~keep&frame_guard(base.shape[:2],1),keep)
        v6clean,applied,used,notes=paper.fill(base,v6update,known)
        # Separately retain the exact Oct 2 lower-rule detector/renderer settings.
        masks,geom=geometry.infer(original,base,roles['red']|roles['extra']|stored_red)
        known_margin,tmargin=paper.paper_reference(base,masks['margin_update'],masks['margin'],masks['mandatory'])
        marginclean,ma,mu,mnotes=paper.fill(base,masks['margin_update'],known_margin)
        for a,u in [(v6clean,applied),(marginclean,ma)]:assert np.array_equal(a[~u],base[~u])
        assert np.array_equal(v6clean[keep],base[keep])
        assert np.array_equal(marginclean[~masks['margin']|masks['mandatory']],base[~masks['margin']|masks['mandatory']])
        poison=base.copy();poison[~known]=[255,0,255]
        poisoned,pa,pu,_=paper.fill(poison,v6update,known)
        assert np.array_equal(poisoned[applied],v6clean[applied]) and np.array_equal(pa,applied) and np.array_equal(pu,used)
        native=OUT/'native'/fid
        for key,a in dict(original=original,baseline=base,v6_all_boundary=v6naive,v6_clean_reference=v6clean,
                          ruled_clean_reference=marginclean,v6_update=v6update,v6_applied=applied,
                          v6_reference=known,v6_used_reference=used,ruled_update=masks['margin_update'],
                          ruled_applied=ma,keep=keep).items():write(native/(key+'.png'),a)
        figure([original,base,v6naive,v6clean,marginclean],
               ['原画像','固定npjの対照','v6範囲・全境界参照','v6範囲・紙面参照','下端枠外・紙面参照'],
               OUT/(fid+'_comparison.png'),c['page']+'：固定した汚れ処理を追加冊へ適用',
               ['768px原画素の切り出し。通常の黒文字マスク＋既存紙背景による今回の対照。',
                'v6は全原ページで分類。v6列は同じ候補、右端は既存の下端枠外検出。筆画補完なし。',
                '全境界／紙面だけの差は参照条件。新対照は過去の保存復元や全ページ推論ではない。'],overview=True)
        rendered=overlay(original,[('edit',applied),('blue',used),('hold',keep)])
        figure([original,rendered],['原画像','黄＝適用、青＝使用紙面、紫＝保持'],OUT/(fid+'_scope.png'),c['page']+'：文字保護と紙面参照',
               ['表示は元画素。分類・復元へ表示位置や症例名を渡さない。',
                '紙面の明るさ条件は前回と固定。v6列は保護／保留を除いた参照領域を使う。'])
        row=dict(id=fid,page=c['page'],role=c['role'],new_v6_book=c['new_v6_book'],
                 v6_update=int(v6update.sum()),v6_applied=int(applied.sum()),
                 v6_changed=int(np.any(v6clean!=base,2).sum()),
                 ruled_update=int(masks['margin_update'].sum()),ruled_applied=int(ma.sum()),
                 ruled_changed=int(np.any(marginclean!=base,2).sum()),
                 v6_reference_L_threshold=threshold,ruled_reference_L_threshold=tmargin,
                 v6_solves=notes,ruled_geometry=geom,ruled_solves=mnotes,
                 outside_update_exact=True,v6_keep_exact=True,ruled_keep_exact=True,
                 excluded_reference_poison_invariant=True)
        rows.append(row)
        print(fid, 'v6 applied',row['v6_applied'],'rule applied',row['ruled_applied'],flush=True)
    dump(OUT/'diagnostics.json',dict(cases=rows,mask_inference_scope='Full original page for v6, aligned 768px raw frame for lower-rule detector',
         fixed_numeric_reference_settings=paper.SETTINGS,ruled_geometry_settings=geometry.SETTINGS,
         v6_reference_domain_extension='Use v6 non-keep pixels as paper domain because these crops lack a lower margin; NOT unchanged Oct2 geometry'))
    dump(OUT/'comparison_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
         for folder in [OUT/'native'] for p in sorted(folder.rglob('*.png'))},
         comparisons={p.name:sha(p) for p in sorted(OUT.glob('*_comparison.png'))},thresholds_changed=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','classify','compare'])
    stage=parser.parse_args().stage
    {'prepare':prepare,'infer':infer,'classify':classify,'compare':compare}[stage]()
