"""Full-page top-boundary relaxation; frozen bands/anchors and old edits retained."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_margin_defects as old
from build_npj_v6_comparison import dump, sha, read, write

ROOT = old.ROOT
OUT = ROOT/'tmp_work/margin_top_completion_pilot_20261001'
MODES = ['in_band', 'completed']
SETTINGS = dict(scope='only previously rejected top-touching components, full-page inference',
    old_top_band_fixed=True, original_top_only_touch=True, area_min=128, area_max_fraction=.01,
    seed_min=32, peak_radius_min=6, max_side_fraction=.20,
    complete_top_guard_px=16, bottom_rule_guard_px=32, side_margin_fraction=.06,
    long_horizontal_width_fraction=.55, long_horizontal_guard_px=8,
    support_L_drop=10, support_L_max=80, support_close_px=3,
    grow_radius_px=3, grow_L_drop=4,
    conservation='frozen warm anchor guard plus v6 red/extra; old edit union always retained',
    ablation='release top touch inside old band vs trace same selected components into safe upper domain',
    ROI_gating=False, training=False, pixel_GT=False, development=True)


def infer(original, features, old_masks, maps):
    H, W = original.shape[:2]
    lab = cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    drop = features['light_drop']; margin = maps['top_margin']; connected = maps['dark_support']
    anchor = maps['warm_anchor_guard']
    n, labels, stats, _ = cv2.connectedComponentsWithStats(connected.astype(np.uint8), connectivity=8)
    seed_count = np.bincount(labels.ravel(), weights=(features['color_seed']&margin).ravel(), minlength=n)
    inner = cv2.erode(margin.astype(np.uint8), np.ones((3,3),np.uint8))>0
    border = connected & ~inner
    allowed_top = np.zeros_like(margin); rule_y = None
    if margin.any():
        yy, xx = np.nonzero(margin); top_y = int(yy.min()); bottom = int(yy.max())+1
        allowed_top[top_y, int(xx.min())+1:int(xx.max())] = True
        rule_y = bottom+32
    top_count = np.bincount(labels.ravel(), weights=(border&allowed_top).ravel(), minlength=n)
    other_count = np.bincount(labels.ravel(), weights=(border&~allowed_top).ravel(), minlength=n)
    distance = cv2.distanceTransform(connected.astype(np.uint8), cv2.DIST_L2,5)
    peak = np.zeros(n,np.float32); np.maximum.at(peak,labels.ravel(),distance.ravel())
    selected = np.zeros(n,bool); top_notes = []
    for i in range(1,n):
        x,y,w,h,area = map(int,stats[i])
        eligible = (128<=area<=.01*H*W and seed_count[i]>=32 and peak[i]>=6
                    and max(w,h)<=.20*min(H,W) and top_count[i]>0 and other_count[i]==0)
        selected[i] = eligible
        if top_count[i]>0 and seed_count[i]>=1:
            top_notes.append(dict(id=i,xywh=[x,y,w,h],area=area,seeds=int(seed_count[i]),
                peak=float(peak[i]),top_contacts=int(top_count[i]),other_contacts=int(other_count[i]),eligible=bool(eligible)))
    selected_core = selected[labels]&connected
    raw_support = (drop>=10)&(lab[:,:,0]<=80)
    long_lines = cv2.morphologyEx(raw_support.astype(np.uint8),cv2.MORPH_OPEN,
        np.ones((1,max(3,round(.55*W))),np.uint8))>0
    line_guard = cv2.dilate(long_lines.astype(np.uint8),np.ones((17,17),np.uint8))>0
    domain = np.zeros_like(margin)
    if rule_y is not None:
        domain[16:max(16,rule_y-32),round(.06*W):round(.94*W)] = True
    support = raw_support&domain&~line_guard
    extended = cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    extended &= domain&~line_guard
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(extended.astype(np.uint8),connectivity=8)
    selected_pixels = np.bincount(ll.ravel(),weights=selected_core.ravel(),minlength=nn)
    seeds = np.bincount(ll.ravel(),weights=(features['color_seed']&margin).ravel(),minlength=nn)
    dist = cv2.distanceTransform(extended.astype(np.uint8),cv2.DIST_L2,5)
    peaks = np.zeros(nn,np.float32);np.maximum.at(peaks,ll.ravel(),dist.ravel())
    domain_inner = cv2.erode(domain.astype(np.uint8),np.ones((3,3),np.uint8))>0
    outer = np.bincount(ll.ravel(),weights=(extended&~domain_inner).ravel(),minlength=nn)
    valid = np.zeros(nn,bool); full_notes = []
    for i in range(1,nn):
        if selected_pixels[i]==0:continue
        x,y,w,h,area = map(int,ss[i])
        accepted = (128<=area<=.01*H*W and seeds[i]>=32 and peaks[i]>=6
                    and max(w,h)<=.20*min(H,W) and outer[i]==0)
        valid[i] = accepted
        full_notes.append(dict(id=i,xywh=[x,y,w,h],area=area,selected_overlap=int(selected_pixels[i]),
            seeds=int(seeds[i]),peak=float(peaks[i]),outer_contacts=int(outer[i]),accepted=bool(accepted)))
    full_core = valid[ll]&extended
    outputs = {}
    for name,core,allowed in [('in_band',selected_core,margin),('completed',full_core,domain&~line_guard)]:
        halo = cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0
        candidate = halo&(drop>=4)&allowed&~anchor
        added = candidate&~old_masks['edit']
        outputs[name] = dict(edit=old_masks['edit']|added,added=added,
            protect=old_masks['protect']&~added,hold=old_masks['hold']&~added,red=old_masks['red'])
        assert not np.any(outputs[name]['edit']&(outputs[name]['protect']|outputs[name]['hold']|outputs[name]['red']))
        assert not np.any(added&anchor)
        assert not np.any(added&~allowed)
    return (outputs,dict(top_margin=margin,safe_domain=domain,long_line_guard=line_guard,
        selected_top_core=selected_core,completed_core=full_core,warm_anchor_guard=anchor),
        dict(rule_y=rule_y,top_components=top_notes,completed_components=full_notes))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(old.__file__).relative_to(ROOT)):sha(Path(old.__file__))}
    for name in ['candidate_manifest.json','candidate_complete.json']:
        dependencies[str((old.OUT/name).relative_to(ROOT))]=sha(old.OUT/name)
    previous = json.loads((old.OUT/'candidate_manifest.json').read_text())
    for rel,digest in previous['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in json.loads((old.OUT/'candidate_complete.json').read_text())['outputs'].items():assert sha(old.OUT/rel)==digest,rel
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),pages=14,regions=22,
        selection='same all 22 development ROIs; no output-based selection',DDRM=False)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[]
    for source in old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key=page['page'];original=read(ROOT/page['original'])
            v6={k:read(source/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
            features,_=old.load_features(key,original,v6)
            frozen=old.OUT/'candidate_pages'/key
            old_masks={k:read(frozen/'dark_support'/(k+'.png'),True) for k in ['edit','protect','hold','red']}
            old_maps={k:read(frozen/'features'/(k+'.png'),True) for k in ['top_margin','dark_support','warm_anchor_guard']}
            modes,maps,details=infer(original,features,old_masks,old_maps)
            dest=OUT/'pages'/key
            for mode,values in modes.items():
                for name,value in values.items():write(dest/mode/(name+'.png'),value)
            for name,value in maps.items():write(dest/'features'/(name+'.png'),value)
            dump(dest/'components.json',details)
            regions=[dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+page['rois']
            for roi in regions:
                rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
                paints=[old.first.paint(crop,{k:v[s] for k,v in masks.items()}) for masks in [old_masks]+list(modes.values())]
                added={m:int(v['added'][s].sum()) for m,v in modes.items()}
                old.first.figure([crop]+paints,['原画像','前回の余白マスク','上端解除・帯内のみ','上端解除・全成分'],
                    OUT/'comparisons'/(rid+'.png'),roi['title']+' · '+key,
                    f'前回からの追加：帯内 {added["in_band"]:,} / 全成分 {added["completed"]:,}画素。画素精度ではない。')
                if not rid.endswith('_whole'):
                    for name,value in dict(original=crop,old_edit=old_masks['edit'][s],
                        top_margin=maps['top_margin'][s],safe_domain=maps['safe_domain'][s],
                        **{m+'_edit':v['edit'][s] for m,v in modes.items()},
                        **{m+'_added':v['added'][s] for m,v in modes.items()}).items():write(OUT/'native'/rid/(name+'.png'),value)
                rows.append(dict(id=rid,page=key,title=roi['title'],xywh=roi['xywh'],added_pixels=added))
            print('Top completion:',key,'additional',int(modes['in_band']['added'].sum()),int(modes['completed']['added'].sum()),flush=True)
            del original,v6,features,old_masks,old_maps,modes,maps;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=14,regions=22,full_page_inference=True,ROI_gating=False,
        old_edit_preserved=True,anchors_preserved=True,edit_protect_hold_disjoint=True,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json'] and p.name not in ['manifest.json','complete.json']}))
    print('Mask comparison complete: 14 pages / 22 ROIs',flush=True)


if __name__=='__main__':main()
