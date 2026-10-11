"""Frozen four-page transfer probe; no parameter search on these outputs."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import argparse
import importlib.util
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import run_ddrm_pilot as driver
import probe_howe_csc_fusion as probe

BASE = Path(__file__).resolve().parent
ROOT = driver.ROOT
OUT = BASE/'new_material_validation'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
SELECTION = [
    ('V01', '200019865_00029_2', [400,850,750,950], '細い曲線・小さな濁点を含む本文',
     [[300,650,1600,2200]]),
    ('V02', '200017458_00008_2', [650,900,850,900], '染みと淡い像が重なる密な本文',
     [[170,500,1850,2420]]),
    ('V03', '200018243_00004_1', [1100,140,800,780], '上部余白の点と枠に近い本文',
     [[1050,580,900,2200]]),
    ('V04', '200018243_00038_2', [50,800,750,1600], '本文が疎なページの独立した小さな印字',
     [[35,840,160,1610],[1530,500,480,2250]]),
]


def frozen_candidate(howe, csc, degradation, red):
    _, labels, _, _ = cv2.connectedComponentsWithStats(howe.astype(np.uint8),8)
    _, _, meta = probe.components(howe,csc,degradation,red)
    pad = int(np.ceil(3*meta['stroke_width']))
    selected = []
    H,W = howe.shape
    for c in meta['removed_components']:
        if c['reason'] != 'isolated_round_mark' or c['csc_fraction'] < .5:
            continue
        x,y,w,h = c['xywh']
        sl = np.s_[max(0,y-pad):min(H,y+h+pad),max(0,x-pad):min(W,x+w+pad)]
        if np.any(howe[sl] & (labels[sl] != c['component'])):
            continue
        selected.append(c)
    removed = np.isin(labels,[c['component'] for c in selected])
    assert not np.any(removed & red)
    return howe & ~removed, removed, dict(stroke_width=meta['stroke_width'],
                                         removed_components=selected)


def regression():
    old=BASE/'howe_csc_fusion/isolated_csc_v2'
    cases=json.loads((BASE/'test_candidates.json').read_text())['cases']
    for case in cases:
        p=case['name'];csc=ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p
        h=probe.load(old/'masks/howe'/p/'02_text_black.png',True)
        candidate,_,_=frozen_candidate(h,probe.load(csc/'02_text_black.png',True),
            probe.load(csc/'04_degradation.png',False),probe.load(csc/'03_text_red.png',True))
        assert np.array_equal(candidate,probe.load(old/'masks/isolated'/p/'02_text_black.png',True)),p
    return 'all six original masks reproduced exactly'


def panel(path, arrays, titles, roi, cid):
    x,y,w,h=roi;gap=12
    canvas=Image.new('RGB',(len(arrays)*(w+gap)+gap,h+70),'#f4f2ec')
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(FONT,24)
    for i,(a,t) in enumerate(zip(arrays,titles)):
        px=gap+i*(w+gap);draw.text((px,10),cid+' '+t,font=font,fill='black')
        tile=Image.fromarray(a[y:y+h,x:x+w]).convert('RGB')
        canvas.paste(tile,(px,58))
        assert np.array_equal(np.asarray(canvas.crop((px,58,px+w,58+h))),np.asarray(tile))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return canvas


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=OUT/'protocol.json'
    assert not protocol.exists(), 'Use saved protocol; do not silently reselect pages'
    prior=json.loads((BASE/'ddrm_pilot/protocol.json').read_text())
    old_books={c['page'].split('_')[0] for c in prior['cases']}
    sources=[]
    for cid,p,roi,reason,protect in SELECTION:
        assert p.split('_')[0] not in old_books
        source=ROOT/'data/test'/p.split('_')[0]/(p+'.jpg')
        size=Image.open(source).size
        for x,y,w,h in [roi]+protect:
            assert x>=0 and y>=0 and x+w<=size[0] and y+h<=size[1]
        sources.append(dict(id=cid,page=p,source=driver.relative(source),source_sha256=driver.digest(source),
                            input_size=list(size),roi_xywh=roi,reason=reason,protect_rectangles_xywh=protect))
    frozen=BASE/'howe_csc_fusion/isolated_csc_v2/protocol.json'
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),cases=sources,
        conditions_fixed_before_outputs=True,ddrm=prior['ddrm'],
        sauvola=dict(window=61,k=.2,R=.5),howe=dict(c=160,sigma=.6,low=.1,high=.4),
        isolated_rule=json.loads(frozen.read_text())['candidate_rule'],
        frozen_rule_protocol_sha256=driver.digest(frozen),regression=regression(),
        selection_scope='three books absent from six development cases; source-only purpose selection, not random final test',
        prior_exposure='Original thumbnails and other source-screening scores existed before this run; not a wholly unseen corpus',
        ground_truth='No dense pixel GT. Protection rectangles are source-only review regions, not labels for every pixel',
        shared_stepwise_noise_guaranteed=False,script_sha256=driver.digest(Path(__file__)))
    driver.save_json(protocol,provenance)
    spec=importlib.util.spec_from_file_location('classical_transfer',ROOT/'tmp_work/classical_deep_review_20261008/run.py')
    deep=importlib.util.module_from_spec(spec);spec.loader.exec_module(deep)
    cv2.setNumThreads(2);diagnostics=[]
    for case in sources:
        p=case['page'];cscdir=ROOT/'data/split_dataset_csc_mask_v2/test'/p.split('_')[0]/p
        raw=np.asarray(Image.open(ROOT/case['source']).convert('RGB'))
        gray=deep.base.gray(raw)
        deep.SCRATCH=OUT/'scratch'/p;deep.SCRATCH.mkdir(parents=True,exist_ok=True)
        terms,term_meta=deep.howe_terms(gray);howe,cut_meta=deep.howe(terms,160)
        red=probe.load(cscdir/'03_text_red.png',True)
        overlap=int((howe & red).sum())
        # Fixed red priority for the pipeline's mutually exclusive ink channels.
        howe &= ~red
        csc=probe.load(cscdir/'02_text_black.png',True)
        isolated,removed,meta=frozen_candidate(howe,csc,probe.load(cscdir/'04_degradation.png',False),red)
        masks={'csc':csc,'howe':howe,'isolated':isolated}
        for method,m in masks.items():
            probe.save(OUT/'masks'/method/p/'02_text_black.png',m)
            probe.save(OUT/'masks'/method/p/'03_text_red.png',red)
        inp=OUT/'inputs'/p;inp.mkdir(parents=True,exist_ok=True)
        link=inp/Path(case['source']).name
        if not link.exists():link.symlink_to(ROOT/case['source'])
        overlay=raw.copy();overlay[removed]=[255,40,40]
        def rgb_mask(m):return np.repeat(np.where(m,0,255).astype(np.uint8)[:,:,None],3,axis=2)
        panel(OUT/'mask_figures'/(case['id']+'_マスク比較.png'),
              [raw,rgb_mask(csc),rgb_mask(howe),rgb_mask(isolated),overlay],
              ['原画像','CSC v2','Howe 固定','孤立点条件後','除去候補（赤）'],case['roi_xywh'],case['id'])
        protection=[]
        for r in case['protect_rectangles_xywh']:
            x,y,w,h=r;sl=np.s_[y:y+h,x:x+w]
            protection.append(dict(xywh=r,removed_pixels=int(removed[sl].sum())))
        # Show every selected component, not just positive examples.
        for i,c in enumerate(meta['removed_components']):
            x,y,w,h=c['xywh'];cx,cy=x+w//2,y+h//2
            x0,y0=max(0,cx-110),max(0,cy-110)
            cr=[x0,y0,min(220,raw.shape[1]-x0),min(220,raw.shape[0]-y0)]
            panel(OUT/'component_figures'/f"{case['id']}_{i:03d}.png",
                  [raw,overlay,rgb_mask(howe),rgb_mask(isolated)],
                  ['原画像','除去（赤）','Howe','条件後'],cr,case['id'])
        deps={driver.relative(q):driver.digest(q) for q in [ROOT/case['source'],
              cscdir/'02_text_black.png',cscdir/'03_text_red.png',cscdir/'04_degradation.png']}
        hashes={driver.relative(q):driver.digest(q) for q in (OUT/'masks').glob('* /nothing')}
        hashes={driver.relative(q):driver.digest(q) for method in masks
                for q in (OUT/'masks'/method/p).glob('*.png')}
        row=dict(id=case['id'],page=p,howe_terms=term_meta,howe_cut=cut_meta,
                 red_pixels=int(red.sum()),howe_red_overlap_before_priority=overlap,
                 removed_pixels=int(removed.sum()),protection_regions=protection,
                 dependencies=deps,masks_sha256=hashes,**meta)
        diagnostics.append(row)
        print(case['id'],len(meta['removed_components']),'components',int(removed.sum()),'removed pixels',
              'protected region changes',protection,flush=True)
        driver.save_json(OUT/'mask_diagnostics.json',diagnostics)
    driver.save_json(OUT/'masks_complete.json',dict(cases=len(sources),diagnostics_sha256=driver.digest(OUT/'mask_diagnostics.json')))


def infer():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert (OUT/'masks_complete.json').exists()
    assert driver.digest(driver.PIPELINE)==protocol['ddrm']['code_sha256']
    assert driver.digest(driver.CHECKPOINT)==protocol['ddrm']['checkpoint_sha256']
    assert driver.digest(driver.BACKGROUND)==protocol['ddrm']['background_sha256']
    diagnostics=json.loads((OUT/'mask_diagnostics.json').read_text())
    for row in diagnostics:
        for rel,sha in {**row['dependencies'],**row['masks_sha256']}.items():assert driver.digest(ROOT/rel)==sha
    driver.OUT=OUT
    def group(cases,gpu):
        results=[]
        for case in cases:
            for method in ['csc','howe','isolated']:
                p=case['page'];same=method=='isolated' and np.array_equal(
                    np.asarray(Image.open(OUT/'masks/howe'/p/'02_text_black.png')),
                    np.asarray(Image.open(OUT/'masks/isolated'/p/'02_text_black.png')))
                if same:
                    src=OUT/'outputs/howe'/p/'result'/(p+'.png');dst=OUT/'outputs/isolated'/p/'result'/(p+'.png')
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():dst.symlink_to(src)
                    assert driver.digest(src)==driver.digest(dst)
                    result=dict(id=case['id'],method=method,reused=True,reason='identical black and red masks',
                                output=driver.relative(dst),sha256=driver.digest(dst))
                else:
                    result=driver.run(case,method,gpu)
                    log=OUT/'logs'/f"{case['id']}_{method}.log"
                    assert '[ERROR]' not in log.read_text(errors='replace')
                results.append(result)
        return results
    cases=protocol['cases']
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(group,cases[::2],0);b=pool.submit(group,cases[1::2],1)
        results=a.result()+b.result()
    figures=[];measurements=[]
    for case,row in zip(cases,diagnostics):
        p=case['page'];raw=np.asarray(Image.open(ROOT/case['source']).convert('RGB'))
        arrays=[np.asarray(Image.open(OUT/'outputs'/m/p/'result'/(p+'.png')).convert('RGB')) for m in ['csc','howe','isolated']]
        figure=panel(OUT/'ddrm_figures'/(case['id']+'_DDRM比較.png'),[raw]+arrays,
                     ['原画像','CSC → DDRM','Howe → DDRM','孤立点条件 → DDRM'],case['roi_xywh'],case['id'])
        figures.append(figure)
        protection=[]
        for r in case['protect_rectangles_xywh']:
            x,y,w,h=r;sl=np.s_[y:y+h,x:x+w]
            protection.append(dict(xywh=r,changed_output_pixels=int(np.any(arrays[1][sl]!=arrays[2][sl],axis=2).sum())))
        components=[]
        for i,c in enumerate(row['removed_components']):
            x,y,w,h=c['xywh'];sl=np.s_[y:y+h,x:x+w]
            components.append(dict(xywh=c['xywh'],area=c['area'],
                baseline_dark_pixels=int((arrays[1][sl].mean(axis=2)<=128).sum()),
                candidate_dark_pixels=int((arrays[2][sl].mean(axis=2)<=128).sum())))
            cx,cy=x+w//2,y+h//2;x0,y0=max(0,cx-110),max(0,cy-110)
            roi=[x0,y0,min(220,raw.shape[1]-x0),min(220,raw.shape[0]-y0)]
            panel(OUT/'ddrm_component_figures'/f"{case['id']}_{i:03d}.png",[raw]+arrays,
                  ['原画像','CSC','Howe','孤立点条件'],roi,case['id'])
        measurements.append(dict(id=case['id'],protection_regions=protection,removed_components=components))
    width=max(f.width for f in figures);height=sum(f.height for f in figures)+18*(len(figures)-1)
    sheet=Image.new('RGB',(width,height),'#ddd');y=0
    for f in figures:sheet.paste(f,(0,y));y+=f.height+18
    path=OUT/'別資料4ページ_DDRM比較.png';sheet.save(path)
    sheet.resize((1800,round(height*1800/width)),Image.Resampling.LANCZOS).save(OUT/'別資料4ページ_DDRM比較_preview.png')
    for row in diagnostics:
        for rel,sha in {**row['dependencies'],**row['masks_sha256']}.items():assert driver.digest(ROOT/rel)==sha
    driver.save_json(OUT/'ddrm_complete.json',dict(results=results,measurements=measurements,
        figure=driver.relative(path),figure_sha256=driver.digest(path),inputs_unchanged=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else infer()
