"""Frozen exploratory paper-margin gate; original images plus frozen v6 only.

This does not certify that a margin contains no writing. Brown seals above the
body rule are deliberately included as a negative example in the new selection.
Never pass ROI coordinates or semantic labels to infer().
"""
import csv
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_defect_context_preprocessing as first
import probe_defect_context_anchors as second
from build_npj_v6_comparison import dump, sha, read, write
from collect_saved_npj_failures import saved_masks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/paper_margin_defect_pilot_20261001'
OLD_V2 = ROOT / 'tmp_work/defect_context_preprocessing_pilot_v2_20261001'
SOURCES = first.SOURCES + [OUT]
SETTINGS = dict(
    rule=dict(scale=4, min_L_drop=8, a_max=6, kernel_width_fraction=.28,
              min_width_fraction=.55, max_height_fraction=.03, y_min_fraction=.075,
              y_max_fraction=.55, top_guard_px=32),
    margin=dict(x_margin_fraction=.06, y_min_fraction=.04),
    support=dict(min_L_drop=10, L_max=80, closing_kernel=3,
                 min_area=128, max_area_fraction=.01, min_brown_seed_pixels=32,
                 min_peak_radius=6, max_side_fraction=.20,
                 grow_radius=3, grow_L_drop=4),
    conservation='preserve v2 warm-anchored networks plus v6 red and extra',
    edit='fixed v6 edit OR top-margin candidate; not v2 edit union',
    ablation='brown-only versus neutral dark support, with identical margin and anchors',
    full_page=True, ROI_gating=False, pixel_GT=False, training=False,
    purpose='exploratory top-margin heuristic, not a no-text certificate or adopted method')


def top_rule(lab, drop):
    h, w = drop.shape
    bw = ((drop >= 8) & (lab[:, :, 1] <= 6)).astype(np.float32)
    small = cv2.resize(bw, ((w+3)//4, (h+3)//4), interpolation=cv2.INTER_AREA) >= .35
    opened = cv2.morphologyEx(small.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((1, max(3, round(small.shape[1]*.28))), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    lines = []
    for i in range(1, n):
        x, y, rw, rh, area = map(int, stats[i])
        if rw*4 >= .55*w and rh*4 <= .03*h and .075*h <= y*4 <= .55*h:
            lines.append(dict(x=x*4, y=y*4, width=rw*4, height=rh*4, area_small=area))
    lines.sort(key=lambda r: (r['y'], -r['width']))
    return (lines[0]['y'] if lines else None), lines


def infer(original, v6, features):
    h, w = original.shape[:2]
    median = cv2.medianBlur(original, 3)
    lab = cv2.cvtColor(median.astype(np.float32)/255., cv2.COLOR_BGR2LAB)
    drop = features['light_drop']
    rule_y, lines = top_rule(lab, drop)
    margin = np.zeros((h, w), bool)
    if rule_y is not None:
        margin[round(.04*h):max(round(.04*h), rule_y-32), round(.06*w):round(.94*w)] = True
    # Region boundaries are barriers; reject components that touch them.
    support = (drop >= 10) & (lab[:, :, 0] <= 80) & margin
    connected = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    connected &= margin
    n, labels, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), connectivity=8)
    seed = features['color_seed'] & margin
    seed_counts = np.bincount(labels.ravel(), weights=seed.ravel(), minlength=n)
    distance = cv2.distanceTransform(connected.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = np.zeros(n, np.float32)
    np.maximum.at(peaks, labels.ravel(), distance.ravel())
    valid = np.zeros(n, bool); components = []
    inner = cv2.erode(margin.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    boundary_counts = np.bincount(labels.ravel(), weights=(connected & ~inner).ravel(), minlength=n)
    for i in range(1, n):
        x, y, rw, rh, area = map(int, stats[i])
        accept = (128 <= area <= .01*h*w and seed_counts[i] >= 32 and peaks[i] >= 6
                  and max(rw, rh) <= .20*min(h, w) and boundary_counts[i] == 0)
        valid[i] = accept
        if seed_counts[i] >= 1:
            components.append(dict(id=i, xywh=[x,y,rw,rh], area=area,
                brown_seed_pixels=int(seed_counts[i]), peak_radius=float(peaks[i]),
                touches_margin_boundary=bool(boundary_counts[i]), accepted=bool(accept)))
    core = valid[labels] & connected
    halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))) > 0
    anchored = features['warm_anchor_guard'] | v6['red'] | v6['extra']
    # Matched ablation: the same component seeds, margin, halo and anchors.
    dark_candidate = halo & (drop >= 4) & margin & ~anchored
    brown_candidate = dark_candidate & features['color_soft']
    modes = {}
    for name, candidate in [('brown_only', brown_candidate), ('dark_support', dark_candidate)]:
        added = candidate & ~v6['edit']
        modes[name] = dict(edit=v6['edit'] | added, added=added,
            protect=v6['protect'] & ~added, hold=v6['hold'] & ~added, red=v6['red'])
        assert not np.any(modes[name]['edit'] & (modes[name]['protect'] | modes[name]['hold'] | v6['red'] | v6['extra']))
        assert not np.any(added & ~margin)
    return modes, dict(top_margin=margin, dark_support=connected, accepted_core=core,
                      warm_anchor_guard=anchored), dict(rule_y=rule_y, rules=lines, components=components)


def load_features(page, original, v6):
    path = OLD_V2 / 'pages' / page
    if path.exists():
        features = {k:read(path/'features'/(k+'.png'),True)
                    for k in ['color_seed', 'color_soft', 'warm_anchor_guard']}
        features['light_drop'] = np.load(path/'features/light_drop.npy')
        previous = {k:read(path/'context_stroke'/(k+'.png'),True)
                    for k in ['edit', 'added', 'protect', 'hold', 'red']}
    else:
        modes, features, _ = second.infer(original, v6)
        previous = modes['context_stroke']
    return features, previous


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []; dependencies = {str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for source in SOURCES:
        path = source / 'baseline_review.json'; dependencies[str(path.relative_to(ROOT))] = sha(path)
        for row in json.loads(path.read_text())['cases']:
            for key in ['original','npj_black','npj_red','npj_degradation']:
                dependencies[row[key]] = sha(ROOT / row[key])
            for key in ['edit','protect','hold','red','extra']:
                p = source/'v6_masks'/row['page']/(key+'.png'); dependencies[str(p.relative_to(ROOT))] = sha(p)
            cases.append((source,row))
    for p in [Path(first.__file__), Path(second.__file__), OLD_V2/'manifest.json', OLD_V2/'complete.json']:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    prior = json.loads((OLD_V2/'complete.json').read_text())['outputs']
    for source,row in cases:
        path=OLD_V2/'pages'/row['page']
        if path.exists():
            for p in path.rglob('*'):
                if p.suffix in ['.png','.npy']:
                    assert sha(p)==prior[str(p.relative_to(OLD_V2))]
                    dependencies[str(p.relative_to(ROOT))]=sha(p)
    manifest=dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)),
        pages=len(cases), regions=sum(len(c['rois']) for _,c in cases),
        previous_regions=15,new_regions=7,parameter_tuning_after_new_output=False,
        pixel_GT=False,standalone_original_npj_reproduction=False)
    p=OUT/'candidate_manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Freeze changed; use a new output directory'
    else:dump(p,manifest)
    shutil.copyfile(__file__,OUT/'candidate_source_snapshot.py')
    diagnostics=[]
    for source,row in cases:
        page=row['page'];original=read(ROOT/row['original']);npj=saved_masks(row)
        v6={k:read(source/'v6_masks'/page/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        features, previous=load_features(page,original,v6)
        modes,maps,details=infer(original,v6,features)
        dest=OUT/'candidate_pages'/page
        for mode,masks in dict(previous_v2=previous,**modes).items():
            for key,value in masks.items():write(dest/mode/(key+'.png'),value)
        for key,value in maps.items():write(dest/'features'/(key+'.png'),value)
        dump(dest/'components.json',details)
        regions=[dict(id=page+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]],baseline_failure=False)]+row['rois']
        for roi in regions:
            id=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
            npj_paint=first.overlay(crop,[('edit',npj['edit'][s]),('blue',npj['black'][s]),('red',npj['red'][s])])
            images=[crop,npj_paint,first.paint(crop,{k:v[s] for k,v in v6.items()}),first.paint(crop,{k:v[s] for k,v in modes['dark_support'].items()})]
            count={name:int(m['added'][s].sum()) for name,m in modes.items()}
            note=f'追加候補：茶色のみ {count["brown_only"]:,} / 暗い縁へ接続 {count["dark_support"]:,}画素。余白は文字なしの保証ではない。'
            first.figure(images,['原画像','保存済みnpj','固定v6','試作：上余白＋暗い縁'],OUT/'candidate_comparisons'/(id+'.png'),roi['title']+' · '+page,note)
            first.figure([crop,first.paint(crop,{k:v[s] for k,v in previous.items()}),
                          first.paint(crop,{k:v[s] for k,v in modes['brown_only'].items()}),images[-1]],
                         ['原画像','前回試作v2','上余白＋茶色のみ','上余白＋暗い縁'],OUT/'candidate_ablation'/(id+'.png'),roi['title']+' · '+page,note)
            if not id.endswith('_whole'):
                write(OUT/'candidate_native'/id/'original.png',crop)
                for mode,masks in dict(previous_v2=previous,**modes).items():
                    for key,value in masks.items():write(OUT/'candidate_native'/id/(mode+'_'+key+'.png'),value[s])
                write(OUT/'candidate_native'/id/'top_margin.png',maps['top_margin'][s])
            diagnostics.append(dict(id=id,page=page,xywh=roi['xywh'],baseline_failure=roi['baseline_failure'],
                rule_y=details['rule_y'],previous_v2_added=int(previous['added'][s].sum()),
                brown_only_added=count['brown_only'],dark_support_added=count['dark_support'],
                neutral_extension_added=int((modes['dark_support']['added'][s]&~modes['brown_only']['added'][s]).sum()),
                v6_protect_released=int((modes['dark_support']['added'][s]&v6['protect'][s]).sum())))
        print('Margin probe:',page,'rule',details['rule_y'],'added',int(modes['dark_support']['added'].sum()),flush=True)
        del original,npj,v6,features,previous,modes,maps;gc.collect()
    dump(OUT/'candidate_diagnostics.json',diagnostics)
    with (OUT/'candidate_diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'candidate_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
        for folder in ['candidate_pages','candidate_comparisons','candidate_ablation','candidate_native']
        for p in (OUT/folder).rglob('*') if p.is_file()}))
    dump(OUT/'candidate_verification.json',dict(status='passed',pages=14,regions=22,
        full_page_inference=True,ROI_gating=False,sources_and_hashes_checked=True,
        edit_protect_hold_disjoint=True,red_extra_anchored_networks_preserved=True,
        candidate_additions_only_in_detected_top_margin=True,thresholds_frozen_before_outputs=True,
        pixel_GT=False,independent_accuracy=False,adopted=False,restoration=False))
    print('Margin prototype complete: 14 pages / 22 ROIs',flush=True)


if __name__=='__main__':main()
