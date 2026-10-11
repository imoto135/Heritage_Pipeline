"""Additional fixed-condition evaluation; no output-driven parameter selection.

Pages were screened as sources and have historical Otsu outputs. This is an
additional transfer evaluation, not an untouched final test set.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import threading
import cv2
import numpy as np
from PIL import Image
import run_ddrm_pilot as driver
import probe_howe_csc_fusion as probe
import validate_new_materials as transfer
import validate_red_retention as red_review

ROOT = driver.ROOT
BASE = Path(__file__).resolve().parent
OUT = BASE / 'fixed_holdout_validation'
METHODS = ['csc', 'howe', 'isolated']
SELECTION = [
    ('F01', '200004107_00006_1', [650,900,850,1000], '白黒資料の細い本文と小さな傍記',
     {'text':[500,900,1300,1700], 'margin':[150,300,700,500]}),
    ('F02', '200021063_00007_1', [650,900,800,1000], 'カラー資料の密な本文と青灰色の背面像候補',
     {'text':[250,750,1650,2300], 'paper':[100,100,700,550]}),
    ('F03', '200021063_00009_2', [100,100,1000,1000], '上部の枝状像・紙色の濃淡と本文',
     {'text':[250,950,1500,1950], 'paper':[300,2850,1350,280]}),
    ('F04', '200021063_00002_2', [1650,2500,400,620], '本文枠外の表側朱印',
     {'text':[650,900,800,1000]}),
    ('F05', '200005798_00036_1', [700,500,850,1100], '白黒資料の余白の細点と本文',
     {'text':[450,1100,1150,1500], 'margin':[350,200,1250,650]}),
    ('F06', '200008003_00006_1', [550,900,1050,2400], '離れた日付・署名と疎な本文',
     {'small_text':[650,1300,700,2000], 'body':[1700,1050,780,2700]}),
]


def load(path):
    return probe.load(path, True)


def snapshot(paths):
    records = []
    for path in paths:
        rel = driver.relative(path); dst = OUT / 'source_snapshot' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        assert driver.digest(dst) == driver.digest(path)
        records.append(dict(source=rel, sha256=driver.digest(path), snapshot=driver.relative(dst)))
    driver.save_json(OUT / 'implementation_manifest.json', dict(files=records))


def verify_inputs():
    for row in json.loads((OUT / 'mask_diagnostics.json').read_text()):
        for rel, sha in {**row['dependencies'], **row['masks_sha256']}.items():
            assert driver.digest(ROOT / rel) == sha, rel
    for row in json.loads((OUT / 'implementation_manifest.json').read_text())['files']:
        assert driver.digest(ROOT / row['source']) == row['sha256'], row['source']


def prepare():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Selection is frozen once saved'
    prior = json.loads((BASE / 'ddrm_pilot/protocol.json').read_text())
    used_pages = set(); used_books = set()
    for rel in ['ddrm_pilot/protocol.json', 'new_material_validation/protocol.json', 'red_retention_validation/protocol.json']:
        for case in json.loads((BASE / rel).read_text())['cases']:
            used_pages.add(case['page']); used_books.add(case['page'].split('_')[0])
    cases = []; split = []
    historical = json.loads((ROOT / 'tmp_work/npj_reproduction_audit_20261007/saved_image_dates_rows.json').read_text())
    for cid, page, roi, reason, controls in SELECTION:
        assert page not in used_pages and page.split('_')[0] not in used_books
        src = ROOT / 'data/test' / page.split('_')[0] / (page + '.jpg')
        size = list(Image.open(src).size)
        for x,y,w,h in [roi] + list(controls.values()):
            assert x >= 0 and y >= 0 and x+w <= size[0] and y+h <= size[1], (page, [x,y,w,h])
        split_src = ROOT / 'data/split_dataset/test' / src.parent.name / src.name
        record = dict(id=cid, page=page, raw_split_test_identical=driver.digest(src)==driver.digest(split_src),
            book_in_train=(ROOT / 'data/split_dataset/train' / src.parent.name).exists(),
            book_in_val=(ROOT / 'data/split_dataset/val' / src.parent.name).exists())
        assert record['raw_split_test_identical'] and not record['book_in_train'] and not record['book_in_val']
        split.append(record)
        cases.append(dict(id=cid,page=page,source=driver.relative(src),source_sha256=driver.digest(src),
            input_size=size,roi_xywh=roi,controls_xywh=controls,reason=reason,
            historical_otsu_outputs=[r['path'] for r in historical if page in r['path']]))
    frozen = BASE / 'howe_csc_fusion/isolated_csc_v2/protocol.json'
    driver.save_json(OUT / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        cases=cases,conditions_fixed_before_new_masks_and_outputs=True,ddrm=prior['ddrm'],
        howe=dict(c=160,sigma=.6,low=.1,high=.4),
        isolated_rule=json.loads(frozen.read_text())['candidate_rule'],
        isolated_rule_protocol_sha256=driver.digest(frozen),isolated_regression=transfer.regression(),
        methods=METHODS,red_priority='Both CSC and Howe black &= ~saved CSC red; isolated derived from priority-resolved Howe',
        isolated_component_support='Use original saved CSC black for fixed component-support fractions',
        selection_scope='Four books absent from previous full-page CSC/Howe comparisons in this workflow; six source-selected pages',
        prior_exposure='Original contact sheets and source-screening scores existed; historical Otsu outputs existed but were not inspected for selection; two grayscale books had other mask-only stress pages',
        final_test=False,ground_truth='No dense GT; source-only review rectangles do not label every pixel',
        shared_stepwise_noise_guaranteed_between_black_masks=False,script_sha256=driver.digest(Path(__file__))))
    driver.save_json(OUT / 'split_membership.json', split)
    snapshot([Path(__file__), BASE/'run_ddrm_pilot.py', BASE/'probe_howe_csc_fusion.py',
        BASE/'validate_new_materials.py', BASE/'validate_red_retention.py',
        ROOT/'tmp_work/classical_deep_review_20261008/run.py',
        ROOT/'tmp_work/classical_deep_review_20261008/graphcut_cli',
        ROOT/'tmp_work/classical_black_comparison_20261008/compare.py', driver.PIPELINE])
    spec = importlib.util.spec_from_file_location('classical_fixed', ROOT/'tmp_work/classical_deep_review_20261008/run.py')
    deep = importlib.util.module_from_spec(spec); spec.loader.exec_module(deep)
    cv2.setNumThreads(2); rows = []
    for case in cases:
        page = case['page']; src = ROOT/case['source']; raw = np.asarray(Image.open(src).convert('RGB'))
        cscdir = ROOT/'data/split_dataset_csc_mask_v2/test'/page.split('_')[0]/page
        csc_raw = load(cscdir/'02_text_black.png'); red = load(cscdir/'03_text_red.png')
        deg = probe.load(cscdir/'04_degradation.png', False)
        deep.SCRATCH = OUT/'scratch'/page; deep.SCRATCH.mkdir(parents=True,exist_ok=True)
        terms, term_meta = deep.howe_terms(deep.base.gray(raw)); howe, cut_meta = deep.howe(terms,160)
        assert howe.shape == red.shape == csc_raw.shape == raw.shape[:2]
        overlaps = dict(csc=int((csc_raw & red).sum()), howe=int((howe & red).sum()))
        csc = csc_raw & ~red; howe &= ~red
        isolated, removed, meta = transfer.frozen_candidate(howe,csc_raw,deg,red)
        assert not (removed & red).any()
        for method, mask in zip(METHODS,[csc,howe,isolated]):
            assert not (mask & red).any()
            probe.save(OUT/'masks'/method/page/'02_text_black.png',mask)
            probe.save(OUT/'masks'/method/page/'03_text_red.png',red)
        inp=OUT/'inputs'/page;inp.mkdir(parents=True,exist_ok=True)
        if not (inp/src.name).exists():(inp/src.name).symlink_to(src)
        overlay=raw.copy();overlay[removed]=[255,40,40]
        rgb=lambda m:np.repeat(np.where(m,0,255).astype(np.uint8)[:,:,None],3,2)
        red_review.figure(OUT/'mask_figures'/(case['id']+'_マスク比較.png'),
            [raw,rgb(csc),rgb(howe),rgb(isolated),overlay],
            ['原画像','CSC黒','Howe黒','孤立点条件後','除去候補（赤）'],case['roi_xywh'],case['id'])
        control_counts={}
        for key,roi in case['controls_xywh'].items():
            x,y,w,h=roi;sl=np.s_[y:y+h,x:x+w]
            control_counts[key]=dict(xywh=roi,removed_pixels=int(removed[sl].sum()),red_pixels=int(red[sl].sum()))
        for i,comp in enumerate(meta['removed_components']):
            x,y,w,h=comp['xywh'];x0=max(0,x+w//2-100);y0=max(0,y+h//2-100)
            roi=[x0,y0,min(200,raw.shape[1]-x0),min(200,raw.shape[0]-y0)]
            red_review.figure(OUT/'component_mask_figures'/f"{case['id']}_{i:03d}.png",
                [raw,overlay,rgb(howe),rgb(isolated)],['原画像','除去候補（赤）','Howe黒','条件後'],roi,case['id'])
        deps={driver.relative(p):driver.digest(p) for p in [src,cscdir/'02_text_black.png',cscdir/'03_text_red.png',cscdir/'04_degradation.png']}
        hashes={driver.relative(p):driver.digest(p) for method in METHODS for p in (OUT/'masks'/method/page).glob('*.png')}
        row=dict(id=case['id'],page=page,red_pixels=int(red.sum()),removed_pixels=int(removed.sum()),
            black_red_overlap_before_priority=overlaps,control_regions=control_counts,
            howe_terms=term_meta,howe_cut=cut_meta,dependencies=deps,masks_sha256=hashes,**meta)
        rows.append(row);driver.save_json(OUT/'mask_diagnostics.json',rows)
        print(case['id'],'red',row['red_pixels'],'removed',len(meta['removed_components']),row['removed_pixels'],
              'controls',control_counts,flush=True)
    driver.save_json(OUT/'masks_complete.json',dict(cases=len(cases)))
    verify_inputs()


def infer():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert (OUT/'masks_complete.json').exists();verify_inputs()
    for attr,key in [('PIPELINE','code_sha256'),('CHECKPOINT','checkpoint_sha256'),('BACKGROUND','background_sha256')]:
        assert driver.digest(getattr(driver,attr))==protocol['ddrm'][key]
    driver.OUT=OUT;locks=[threading.Semaphore(2),threading.Semaphore(2)]
    def group(case,gpu):
        results=[]
        with locks[gpu]:
            for method in METHODS:
                page=case['page']
                same=method=='isolated' and np.array_equal(load(OUT/'masks/howe'/page/'02_text_black.png'),
                    load(OUT/'masks/isolated'/page/'02_text_black.png'))
                if same:
                    src=OUT/'outputs/howe'/page/'result'/(page+'.png');dst=OUT/'outputs/isolated'/page/'result'/(page+'.png')
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():dst.symlink_to(src)
                    assert driver.digest(src)==driver.digest(dst)
                    result=dict(id=case['id'],method=method,reused=True,reason='Identical whole-page black and red masks',
                        output=driver.relative(dst),sha256=driver.digest(dst))
                else:
                    result=driver.run(case,method,gpu)
                    assert '[ERROR]' not in (OUT/'logs'/f"{case['id']}_{method}.log").read_text(errors='replace')
                results.append(result)
        return results
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(group,case,i%2) for i,case in enumerate(protocol['cases'])]
        results=[r for job in jobs for r in job.result()]
    verify_inputs()
    driver.save_json(OUT/'inference_complete.json',dict(results=results,inputs_unchanged=True))
    render()


def render():
    protocol=json.loads((OUT/'protocol.json').read_text());complete=json.loads((OUT/'inference_complete.json').read_text())
    diagnostics=json.loads((OUT/'mask_diagnostics.json').read_text());boards=[];audits=[]
    for case,row in zip(protocol['cases'],diagnostics):
        page=case['page'];raw=cv2.imread(str(ROOT/case['source']))[:,:,::-1]
        arrays=[np.asarray(Image.open(OUT/'outputs'/method/page/'result'/(page+'.png')).convert('RGB')) for method in METHODS]
        red=load(OUT/'masks/howe'/page/'03_text_red.png');removed=load(OUT/'masks/howe'/page/'02_text_black.png') & ~load(OUT/'masks/isolated'/page/'02_text_black.png')
        output_checks={}
        for method,arr in zip(METHODS,arrays):
            result=next(r for r in complete['results'] if r['id']==case['id'] and r['method']==method)
            assert list(Image.open(ROOT/result['output']).size)==case['input_size']
            assert driver.digest(ROOT/result['output'])==result['sha256']
            difference=int(np.any(arr[red]!=raw[red],axis=1).sum());assert difference==0
            output_checks[method]=dict(source_red_changed_pixels=difference,output_sha256=result['sha256'])
        board=red_review.figure(OUT/'ddrm_figures'/(case['id']+'_DDRM比較.png'),[raw]+arrays,
            ['原画像','CSC黒＋CSC赤','Howe黒＋CSC赤','孤立点条件＋CSC赤'],case['roi_xywh'],case['id'])
        boards.append(board);controls={}
        for key,roi in case['controls_xywh'].items():
            x,y,w,h=roi;sl=np.s_[y:y+h,x:x+w]
            controls[key]=dict(xywh=roi,removed_mask_pixels=int(removed[sl].sum()),
                changed_output_pixels=int(np.any(arrays[1][sl]!=arrays[2][sl],axis=2).sum()))
            red_review.figure(OUT/'control_figures'/(case['id']+'_'+key+'.png'),[raw]+arrays,
                ['原画像','CSC黒＋CSC赤','Howe黒＋CSC赤','孤立点条件＋CSC赤'],roi,case['id'])
        components=[]
        for i,comp in enumerate(row['removed_components']):
            x,y,w,h=comp['xywh'];sl=np.s_[y:y+h,x:x+w]
            components.append(dict(xywh=comp['xywh'],area=comp['area'],
                baseline_dark_pixels=int((arrays[1][sl].mean(2)<=128).sum()),
                candidate_dark_pixels=int((arrays[2][sl].mean(2)<=128).sum())))
            x0=max(0,x+w//2-100);y0=max(0,y+h//2-100)
            roi=[x0,y0,min(200,raw.shape[1]-x0),min(200,raw.shape[0]-y0)]
            red_review.figure(OUT/'component_ddrm_figures'/f"{case['id']}_{i:03d}.png",[raw]+arrays,
                ['原画像','CSC黒','Howe黒','孤立点条件'],roi,case['id'])
        audits.append(dict(id=case['id'],methods=output_checks,control_regions=controls,removed_components=components))
    width=max(b.width for b in boards);height=sum(b.height for b in boards)+16*(len(boards)-1)
    sheet=Image.new('RGB',(width,height),'#ddd');y=0
    for board in boards:sheet.paste(board,(0,y));y+=board.height+16
    path=OUT/'追加6ページ_固定条件DDRM比較.png';sheet.save(path)
    sheet.resize((1800,round(height*1800/width)),Image.Resampling.LANCZOS).save(OUT/'追加6ページ_固定条件DDRM比較_preview.png')
    verify_inputs()
    driver.save_json(OUT/'ddrm_complete.json',dict(results=complete['results'],audits=audits,
        figure=driver.relative(path),figure_sha256=driver.digest(path),inputs_unchanged=True))
    print('All six pages / three fixed conditions verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','render']);args=parser.parse_args()
    {'prepare':prepare,'infer':infer,'render':render}[args.stage]()
