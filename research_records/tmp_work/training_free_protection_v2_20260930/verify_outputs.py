"""Verify native masks, independent decisions, regression sites and artifacts."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

import cv2
import numpy as np

cv2.setNumThreads(1)
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path,flag=cv2.IMREAD_COLOR):
    value=cv2.imread(str(path),flag)
    assert value is not None,path
    return value


manifest=json.loads((OUT/'manifest.json').read_text())
complete=json.loads((OUT/'complete.json').read_text())
metrics=json.loads((OUT/'diagnostics.json').read_text())
for path,expected in {**manifest['inputs'],**manifest['dependencies']}.items():
    assert sha(ROOT/path)==expected,path
for name,expected in complete['outputs'].items():
    assert sha(OUT/name)==expected,name
assert sha(OUT/'source_snapshot.py')==sha(ROOT/'scripts/benchmark/refine_training_free_protection.py')==manifest['source_sha256']
assert sha(OUT/'viewer_snapshot.html')==sha(ROOT/'scripts/benchmark/training_free_protection_viewer.html')==manifest['viewer_template_sha256']
ast.parse((OUT/'source_snapshot.py').read_text())
keys=['red','protect','edit_allowed','hold','ink_color_uncertain','stain_candidate','black','robust_seed',
      'seed_supported_red','contextual_red','thin_extension','protection_guard','dense_stain','dense_hold',
      'legacy_red','legacy_ink','legacy_stain']
regression={};overlaps=dict(stain_protect=0,protect_hold=0)
for case in manifest['cases']:
    stem=case['id'];folder=OUT/stem
    original=read(folder/'original.png')
    assert np.array_equal(original,read(ROOT/case['original']))
    h,w=original.shape[:2];masks={}
    for key in keys:
        raw=read(folder/(key+'.png'),0)
        assert raw.shape==(h,w) and set(np.unique(raw)).issubset({0,255}),key
        masks[key]=raw>0
    m=masks
    assert np.array_equal(m['protect'],m['black']|m['red']|m['ink_color_uncertain'])
    assert not (m['black']&m['red']).any()
    assert not (m['black']&m['ink_color_uncertain']).any()
    assert not (m['red']&m['ink_color_uncertain']).any()
    assert not (m['red']&~m['legacy_red']).any()
    assert not (m['red']&~m['seed_supported_red']).any()
    assert np.array_equal(m['ink_color_uncertain'],m['protect']&m['legacy_red']&~m['red'])
    assert not ((m['legacy_red']&~m['red'])&~m['hold']).any()
    assert not (m['edit_allowed']&(m['protect']|m['hold'])).any()
    assert np.array_equal(m['edit_allowed'],m['stain_candidate']&~m['protection_guard']&~m['hold'])
    distance=cv2.distanceTransform((~m['protect']).astype(np.uint8),cv2.DIST_L2,5)
    assert np.array_equal(m['protection_guard'],distance<=3.)
    pilot=ROOT/('tmp_work/training_free_preprocess_20260930' if case['group']=='initial_development'
               else 'tmp_work/training_free_fixed_check_20260930/pilot')
    fg=read(pilot/stem/'sauvola_local/foreground.png',0)>0
    votes=read(pilot/stem/'scale_votes.png',0)//85
    assert not ((fg&~m['dense_stain']&~m['dense_hold'])&~m['protect']).any()
    assert not (m['thin_extension']&(votes==0)).any()
    # Removing the color label never authorizes editing a prior ink candidate.
    assert not (m['legacy_ink']&m['edit_allowed']).any()
    assert not ((m['legacy_ink']&~m['protect'])&~m['hold']).any()
    overlaps['stain_protect']+=int((m['stain_candidate']&m['protect']).sum())
    overlaps['protect_hold']+=int((m['protect']&m['hold']).sum())
    for key in ['red','protect','edit_allowed','hold','ink_color_uncertain','stain_candidate']:
        im=read(folder/('overlay_'+key+'.png'))
        assert np.array_equal(im[~m[key]],original[~m[key]])
    old_overlay=read(folder/'overlay_legacy_red.png')
    assert np.array_equal(old_overlay[~m['legacy_red']],original[~m['legacy_red']])
    decision=read(folder/'decision_overlay.png')
    union=m['protect']|m['red']|m['edit_allowed']|m['hold']|m['ink_color_uncertain']
    assert np.array_equal(decision[~union],original[~union])
    for row in [r for r in metrics if r['page']==stem]:
        x,y,rw,rh=row['xywh'];assert 0<=x<x+rw<=w and 0<=y<y+rh<=h
        sl=(slice(y,y+rh),slice(x,x+rw));s={k:v[sl] for k,v in m.items()}
        for key in keys:assert row[key+'_pixels']==int(s[key].sum()),(stem,row['roi'],key)
        expected=dict(previous_red_no_longer_typed=(s['legacy_red']&~s['red']),
                      previous_red_still_protected=(s['legacy_red']&s['protect']),
                      previous_ink_no_longer_protected=(s['legacy_ink']&~s['protect']),
                      new_protected_pixels=(s['protect']&~s['legacy_ink']),
                      stain_protection_overlap=(s['stain_candidate']&s['protect']),
                      protection_hold_overlap=(s['protect']&s['hold']),
                      previous_stain_near_new_protection=(s['legacy_stain']&s['protection_guard']),
                      previous_stain_edit_allowed=(s['legacy_stain']&s['edit_allowed']))
        for key,value in expected.items():assert row[key]==int(value.sum()),(stem,key)
    if stem=='200010454_00002_2':
        sl=(slice(1810,2290),slice(1370,1730));old=m['legacy_red'][sl]
        assert int(old.sum())==20027
        assert not (old&~m['protect'][sl]).any()
        assert not (old&m['edit_allowed'][sl]).any()
        regression['seal_roi']=dict(previous_red_pixels=int(old.sum()),previous_red_protected_pixels=int((old&m['protect'][sl]).sum()),
                                    confirmed_red_pixels=int(m['red'][sl].sum()),manual_recall=False)
    if stem=='hnsd001_011':
        assert int(m['dense_stain'].sum())==3131
        assert not (m['dense_stain']&~m['edit_allowed']).any()
        regression['known_dense_stain']=dict(candidate_pixels=3131,edit_allowed_pixels=3131)
    if case['group']=='regression_development':
        assert not m['red'].any()
    del masks,m,distance
assert overlaps['stain_protect']>0 and overlaps['protect_hold']>0
html=(OUT/'index.html').read_text();script=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
data=json.loads(re.search(r'const DATA=(.*);\nconst \$=',script).group(1))
assert len(data['cases'])==12
for case in data['cases']:
    for roi in case['rois']:assert (OUT/roi['image']).exists()
    for filename in ['original.png','overlay_legacy_red.png','overlay_red.png','decision_overlay.png']+['overlay_'+key+'.png' for key in ['red','protect','edit_allowed','hold','ink_color_uncertain','stain_candidate']]:
        assert (OUT/case['id']/filename).exists()
for href in re.findall(r'href="([^"]+)"',html):
    if href!='verification.json':assert (OUT/href).resolve().exists(),href
js_path=OUT/'viewer_syntax_check.js';js_path.write_text(script)
subprocess.run(['node','--check',str(js_path)],check=True)
result=dict(status='verified',pages=12,fixed_rois=38,all_review_crops=complete['all_review_crops'],
            png_hashes_checked=len(complete['outputs']),regression=regression,actual_mask_overlaps=overlaps,
            checks=['source/input/dependency/output hashes','native binary masks',
                    'original decoded pixels unchanged; overlays unchanged outside selected masks',
                    'red subset of prior red; uncertain color protected or held',
                    'protection equals black/red/unknown-color partition',
                    'edit_allowed excludes protection, hold and 3-pixel protection guard',
                    'local foreground retained except previously logged dense candidates',
                    'no prior ink candidate is edit-allowed; unprotected prior ink is held',
                    'whole-page and review-crop metrics recomputed',
                    'known seal and dense-stain regression checks',
                    'viewer file links and JavaScript syntax'],
            browser_ui_tested=False,manual_ground_truth_used=False,independent_accuracy_test=False)
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
