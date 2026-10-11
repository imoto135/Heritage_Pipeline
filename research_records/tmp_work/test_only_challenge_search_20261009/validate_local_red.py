"""Exploratory source-relative red support, anchored to CSC candidates.

Howe and frozen DDRM stay fixed. Original experiment artifacts stay immutable.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse,json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import check_red_v1_followup as old
import validate_red_retention as review
import run_ddrm_pilot as driver

ROOT=driver.ROOT
BASE=Path(__file__).resolve().parent
OUT=BASE/'red_local_color_validation'
VARIANTS={
    'strict':dict(radius=0,weak_a=3,strong_a=8,strong_fraction=.10,min_seed_pixels=8),
    'grow5':dict(radius=5,weak_a=2,strong_a=8,strong_fraction=.05,min_seed_pixels=8),
    'grow9':dict(radius=9,weak_a=2,strong_a=6,strong_fraction=.05,min_seed_pixels=8),
}

def selected_cases():
    controls={c[0]:c[4] for c in review.SELECTION}
    cases=[dict(id=c,page=p,roi_xywh=r,controls_xywh=controls.get(c,{})) for c,p,r in old.CASES]
    extra=json.loads((BASE/'fixed_holdout_validation/protocol.json').read_text())['cases']
    cases.extend({k:c[k] for k in ['id','page','roi_xywh','controls_xywh']} for c in extra if c['id']!='F04')
    return cases

def color_fields(raw):
    bgr=raw[:,:,::-1].copy();hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    a=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)[:,:,1]
    delta=a.astype(np.int16)-cv2.medianBlur(a,51).astype(np.int16)
    r,g,b=raw.astype(np.int16).transpose(2,0,1)
    chroma=(((hsv[:,:,0]<=15)|(hsv[:,:,0]>=165))&(hsv[:,:,1]>=25)&(r-g>=12)&(r-2*g+b>=6))
    return delta,chroma

def detect(anchor,delta,chroma,params):
    radius=params['radius']
    allowed=cv2.dilate(anchor.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0 if radius else anchor
    weak=allowed&chroma&(delta>=params['weak_a'])
    # Only strong pixels inside CSC's previously supported mask can seed growth.
    seed=weak&anchor&(delta>=params['strong_a'])
    n,labels=cv2.connectedComponents(weak.astype(np.uint8),connectivity=8)
    counts=np.bincount(labels.ravel(),minlength=n)
    seed_counts=np.bincount(labels[seed],minlength=n)
    keep=(seed_counts>=params['min_seed_pixels'])&(seed_counts/np.maximum(counts,1)>=params['strong_fraction'])
    keep[0]=False
    red=keep[labels]
    # Pixel constraints remain after closing: morphology cannot protect neutral ink.
    red=(cv2.morphologyEx(red.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0)&weak
    return red,dict(seed_pixels=int(seed.sum()),weak_pixels=int(weak.sum()),components=int(n-1),kept_components=int(keep.sum()))

def rgbmask(mask):return np.repeat(np.where(mask,0,255).astype(np.uint8)[:,:,None],3,2)

def roi_count(mask,roi):
    x,y,w,h=roi;return int(mask[y:y+h,x:x+w].sum())

def anchor_for(case,v2):
    p=case['page'];cached=old.OUT/p/'red_without_global_gate.png'
    if cached.exists():return review.load(cached),dict(reused=driver.relative(cached),sha256=driver.digest(cached))
    folder=ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p
    ace=cv2.imread(str(folder/'00_input.png'))
    result,trace,candidate=old.run_traced(v2,old.V2,ace)
    saved=review.load(folder/'03_text_red.png');assert np.array_equal(result.red_text_mask>0,saved)
    anchor=v2._apply_morphology(old.component_filter_fast(ace,candidate,.05).astype(np.uint8)*255,3,1)>0
    old.save(OUT/'anchors'/p/'red_without_global_gate.png',anchor)
    return anchor,dict(recomputed=True,trace=trace,ace_sha256=driver.digest(folder/'00_input.png'))

def prepare():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True)
    cases=selected_cases()
    for c in cases:
        src=ROOT/'data/test'/c['page'].split('_')[0]/(c['page']+'.jpg')
        c.update(source=driver.relative(src),source_sha256=driver.digest(src),input_size=list(Image.open(src).size))
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=cases,variants=VARIANTS,
        shared=dict(local_median_kernel=51,hue='<=15 or >=165',saturation_min=25,r_minus_g_min=12,r_minus_2g_plus_b_min=6,
                    csc_anchor='Previous mean-a-gate-off mask; unchanged ACE HSV component support >=.05',post='close3 intersect original weak support'),
        howe=dict(c=160,sigma=.6,low=.1,high=.4),final_test=False,
        scope='Development on observed test-derived pages; not dense-GT accuracy or unseen final test',
        script_sha256=driver.digest(Path(__file__)),frozen_csc_sha256=driver.digest(old.V2))
    path=OUT/'protocol.json';assert not path.exists(), 'Use infer/render; prepare is immutable'
    driver.save_json(path,protocol)
    v2=old.module(old.V2,'local_red_csc');v2._filter_red_candidate_by_component=old.component_filter_fast
    records=[]
    for c in cases:
        p=c['page'];raw=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
        saved=review.load(ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p/'03_text_red.png')
        anchor,meta=anchor_for(c,v2);delta,chroma=color_fields(raw)
        arrays=[raw];titles=['原画像'];row=dict(id=c['id'],page=p,anchor=meta,saved_pixels=int(saved.sum()),variants={})
        for name,params in VARIANTS.items():
            red,stats=detect(anchor,delta,chroma,params);dest=OUT/'red_masks'/name/p/'03_text_red.png';old.save(dest,red)
            overlay=raw.copy();overlay[red]=[20,235,20];arrays.append(overlay);titles.append(name+' 赤保持（緑）')
            stats.update(page_pixels=int(red.sum()),roi_pixels=roi_count(red,c['roi_xywh']),
                controls={k:roi_count(red,r) for k,r in c['controls_xywh'].items()},sha256=driver.digest(dest),
                removed_saved_pixels=int((saved&~red).sum()),added_saved_pixels=int((red&~saved).sum()))
            row['variants'][name]=stats
        baseline=raw.copy();baseline[saved]=[20,235,20];arrays.insert(1,baseline);titles.insert(1,'現行CSC赤（緑）')
        review.figure(OUT/'mask_figures'/(c['id']+'_局所赤比較.png'),arrays,titles,c['roi_xywh'],c['id'])
        for label,roi in c['controls_xywh'].items():
            review.figure(OUT/'control_figures'/(c['id']+'_'+label+'_局所赤比較.png'),arrays,titles,roi,c['id'])
        records.append(row);driver.save_json(OUT/'mask_diagnostics.json',records)
        print(c['id'],json.dumps(row['variants'],ensure_ascii=False),flush=True)
    # Overview uses four confirmed front-red cases and one uncertain ghost.
    sheet=Image.new('RGB',(2000,5*480),'#f4f2ec')
    for i,c in enumerate(cases[:5]):
        im=Image.open(OUT/'mask_figures'/(c['id']+'_局所赤比較.png')).convert('RGB');im.thumbnail((1980,465));sheet.paste(im,(10,480*i+10))
    sheet.save(OUT/'5例_局所赤判定_候補比較.png')

def raw_howe(case):
    p=case['page'];source=BASE/'red_retention_validation/scratch'/p/'graph_output.bin'
    if case['id'] in ('R01','R02','R03'):
        w,h=case['input_size'];b=np.fromfile(source,dtype=np.uint8).reshape(h-1,w-1).astype(bool)
        expanded=np.pad(b,((0,1),(0,1)),mode='edge')
        above=np.concatenate([expanded[:1],expanded[:-1]],axis=0);left=np.concatenate([expanded[:,:1],expanded[:,:-1]],axis=1)
        final=expanded&~(expanded&~above&~left)
        saved_red=review.load(ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p/'03_text_red.png')
        saved_black=review.load(BASE/'red_retention_validation/masks/howe'/p/'02_text_black.png')
        assert np.array_equal(final&~saved_red,saved_black)
        return final,dict(source=driver.relative(source),sha256=driver.digest(source),saved_black_after_priority_match=True)
    source=BASE/'fixed_holdout_validation/masks/howe'/p/'02_text_black.png'
    return review.load(source),dict(source=driver.relative(source),sha256=driver.digest(source),saved_red_was_empty=True)

def infer(variant,ids):
    protocol=json.loads((OUT/'protocol.json').read_text());assert variant in protocol['variants']
    cases=[c for c in protocol['cases'] if c['id'] in ids]
    assert len(cases)==len(ids)
    driver.OUT=OUT;method='howe_'+variant;records=[]
    for c in cases:
        p=c['page'];red=review.load(OUT/'red_masks'/variant/p/'03_text_red.png');howe,meta=raw_howe(c)
        old.save(OUT/'masks'/method/p/'02_text_black.png',howe&~red);old.save(OUT/'masks'/method/p/'03_text_red.png',red)
        stage=OUT/'inputs'/p;stage.mkdir(parents=True,exist_ok=True)
        dest=stage/(p+'.jpg')
        if not dest.exists():dest.symlink_to(ROOT/c['source'])
        records.append(dict(id=c['id'],page=p,howe=meta,red_pixels=int(red.sum()),removed_howe_pixels=int((howe&red).sum()),
            masks_sha256={driver.relative(q):driver.digest(q) for q in (OUT/'masks'/method/p).glob('*.png')}))
    driver.save_json(OUT/('ddrm_protocol_'+variant+'.json'),dict(variant=variant,cases=records,
        source_protocol_sha256=driver.digest(OUT/'protocol.json'),pipeline_sha256=driver.digest(driver.PIPELINE),
        checkpoint_sha256=driver.digest(driver.CHECKPOINT),background_sha256=driver.digest(driver.BACKGROUND),
        parameters=dict(seed=999,steps=20,patch=128,overlap=32,shared_initial_noise=True),
        red='Copy original RGB after inference',stepwise_noise_matched_between_black_masks=False))
    def group(cs,gpu):return [driver.run(c,method,gpu) for c in cs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(group,cases[i::2],i) for i in range(2)]
        results=[r for f in fs for r in f.result()]
    driver.save_json(OUT/('ddrm_results_'+variant+'.json'),results)
    audits=[]
    for c in cases:
        p=c['page'];red=review.load(OUT/'masks'/method/p/'03_text_red.png')
        raw=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
        result=next(r for r in results if r['id']==c['id']);new=np.asarray(Image.open(ROOT/result['output']).convert('RGB'))
        assert new.shape==raw.shape and not np.any(new[red]!=raw[red])
        assert '[ERROR]' not in (OUT/'logs'/(c['id']+'_'+method+'.log')).read_text()
        prevbase=BASE/('fixed_holdout_validation' if c['id']=='F04' else 'red_retention_validation')
        prev=np.asarray(Image.open(prevbase/'outputs/howe'/p/'result'/(p+'.png')).convert('RGB'))
        review.figure(OUT/'ddrm_figures'/(c['id']+'_局所赤_DDRM比較.png'),[raw,prev,new],
            ['原画像','Howe＋現行CSC赤','Howe＋局所赤 '+variant],c['roi_xywh'],c['id'])
        for k,r in c['controls_xywh'].items():
            review.figure(OUT/'ddrm_controls'/(c['id']+'_'+k+'_DDRM比較.png'),[raw,prev,new],
                ['原画像','Howe＋現行CSC赤','Howe＋局所赤 '+variant],r,c['id'])
        audits.append(dict(id=c['id'],output_size_verified=True,red_source_copy_mismatch_pixels=0,log_errors=False))
    driver.save_json(OUT/('ddrm_audit_'+variant+'.json'),audits)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer']);parser.add_argument('--variant',default='grow5');parser.add_argument('--ids',nargs='+',default=['F04','R02','R01','R03'])
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else infer(args.variant,args.ids)
