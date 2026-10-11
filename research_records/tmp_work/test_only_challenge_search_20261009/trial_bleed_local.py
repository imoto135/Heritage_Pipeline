"""Single-page, manually reviewed exclusion experiment; not a general filter."""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
from PIL import Image
import run_ddrm_pilot as driver
import review_bleed_local as review

BASE = Path(__file__).resolve().parent
ROOT = driver.ROOT
OUT = review.OUT/'local_exclusion_trial'
PAGE = review.PAGE
METHOD = 'local_review'

def main():
    annotation = json.loads((review.OUT/'局所分類.json').read_text())
    prior = json.loads((BASE/'ddrm_pilot/protocol.json').read_text())
    for path, expected in annotation['dependencies'].items():
        assert driver.digest(ROOT/path)==expected, path
    for p,key in [(driver.PIPELINE,'code_sha256'),(driver.CHECKPOINT,'checkpoint_sha256'),(driver.BACKGROUND,'background_sha256')]:
        assert driver.digest(p)==prior['ddrm'][key]
    case=next(c for c in prior['cases'] if c['id']=='07')
    olddir=BASE/'six_challenge_fixed_review/masks/howe'/PAGE
    # Frozen masks in the six-case review; same input as the original Howe pilot.
    black_path=olddir/'02_text_black.png'; red_path=olddir/'03_text_red.png'
    old=np.asarray(Image.open(black_path).convert('L'))<127
    red=np.asarray(Image.open(red_path).convert('L'))<127
    csc=np.asarray(Image.open(ROOT/'data/split_dataset_csc_mask_v2/test/200021071'/PAGE/'02_text_black.png').convert('L'))<127
    _,labels,stats,_=cv2.connectedComponentsWithStats(old.astype(np.uint8),8)
    selected=[r for r in annotation['components'] if r.get('local_shape_match')]
    assert [r['label'] for r in selected]==[44,50,51,53]
    remove=np.zeros_like(old)
    for row in selected:
        i=row['label']; assert stats[i].tolist()==row['xywh']+[row['area']]
        component=labels==i
        assert not red[component].any() and not csc[component].any()
        remove|=component
    new=old&~remove
    assert int(remove.sum())==515 and not np.any(new&~old)
    assert np.array_equal(new[~remove],old[~remove])
    dest=OUT/'masks'/METHOD/PAGE;dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.where(new,0,255).astype(np.uint8)).save(dest/'02_text_black.png')
    shutil.copyfile(red_path,dest/'03_text_red.png')
    Image.fromarray(np.where(remove,255,0).astype(np.uint8)).save(OUT/'diagnostic_excluded_components.png')
    inp=OUT/'inputs'/PAGE;inp.mkdir(parents=True,exist_ok=True)
    link=inp/Path(case['source']).name
    if not link.exists():link.symlink_to(ROOT/case['source'])
    protocol=dict(status='prepared',page=PAGE,manually_reviewed_component_ids=[44,50,51,53],
        removed_pixels=515,reason='next-page mirrored shape correspondence in an upper glyph neighborhood',
        physical_cause_confirmed=False,automated_general_rule=False,development_only=True,
        original_source_sha256=driver.digest(ROOT/case['source']),
        black_before_sha256=driver.digest(black_path),black_after_sha256=driver.digest(dest/'02_text_black.png'),
        red_before_sha256=driver.digest(red_path),red_after_sha256=driver.digest(dest/'03_text_red.png'),
        unreviewed_components_preserved=True,ddrm=prior['ddrm'],
        shared_stepwise_noise_guaranteed=False,script_sha256=driver.digest(Path(__file__)),
        annotation_sha256=driver.digest(review.OUT/'局所分類.json'))
    driver.save_json(OUT/'protocol.json',protocol)
    driver.OUT=OUT
    result=driver.run(case,METHOD,0)
    log=OUT/'logs'/f'07_{METHOD}.log'
    assert '[ERROR]' not in log.read_text(errors='replace')
    source=Image.open(ROOT/case['source']).convert('RGB')
    baseline_path=BASE/'six_challenge_fixed_review/outputs/howe'/PAGE/'result'/f'{PAGE}.png'
    baseline=Image.open(baseline_path).convert('RGB')
    candidate=Image.open(ROOT/result['output']).convert('RGB')
    overlay=np.asarray(source).copy();overlay[remove]=[255,40,40]
    controls=annotation['controls']
    previous_out=review.OUT;review.OUT=OUT
    review.board(controls,[source,baseline,candidate,Image.fromarray(overlay)],
                 ['原画像','固定Howe → DDRM','局所除外 → DDRM','除外画素（赤）'],
                 '07_局所除外_DDRM比較.png',(430,500))
    full=[dict(caption='全ページ：局所除外515画素の実験／基本設定は変更しない',context_xyxy=[0,0,source.width,source.height])]
    review.board(full,[source,baseline,candidate],['原画像','固定Howe → DDRM','局所除外 → DDRM'],
                 '07_局所除外_全ページ比較.png',(620,1040))
    review.OUT=previous_out
    before=np.asarray(baseline);after=np.asarray(candidate)
    diagnostic=[]
    for row in annotation['components']:
        comp=labels==row['label']
        bd=cv2.cvtColor(before,cv2.COLOR_RGB2GRAY)[comp]<100
        ad=cv2.cvtColor(after,cv2.COLOR_RGB2GRAY)[comp]<100
        diagnostic.append(dict(label=row['label'],excluded=bool(remove[comp].any()),
             mask_area=int(comp.sum()),before_dark_pixels=int(bd.sum()),after_dark_pixels=int(ad.sum())))
    protocol.update(status='inference_complete_pending_visual_review',result=result,
        baseline_output_sha256=driver.digest(baseline_path),
        component_diagnostics=diagnostic,
        changed_output_pixels=int(np.any(before!=after,axis=2).sum()),
        removed_mask_pixels=int(remove.sum()))
    driver.save_json(OUT/'complete.json',protocol)
    print(json.dumps(diagnostic,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
