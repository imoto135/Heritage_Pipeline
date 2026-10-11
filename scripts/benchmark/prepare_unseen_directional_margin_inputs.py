"""Freeze original-only four-book controls before any method output."""
import csv
import json
from pathlib import Path
import shutil
import cv2
from build_npj_v6_comparison import dump,read,sha
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle
import frozen_margin_preprocessing_candidate as candidate

ROOT=Path(__file__).resolve().parents[2]
SCREEN=ROOT/'tmp_work/unseen_directional_margin_screening_20261001'
OUT=ROOT/'tmp_work/unseen_directional_margin_validation_20261001'
SELECTION={
 '200015843_00003_2':[
  ('directional_fibrous_paper','対照：繊維模様と淡い上余白',[270,50,1500,410]),
  ('directional_faint_ink','対照：淡い本文と紙模様',[260,550,1100,750]),
  ('directional_lower_seal','対照：右下の印影',[1780,2670,260,420])],
 '200021637_00019_2':[
  ('directional_diagonal_paper','対照：斜め模様の上部と貼付票',[70,70,1480,230]),
  ('directional_pattern_left','対照：上部左側の紙模様',[160,185,160,140]),
  ('directional_attached_seal','対照：上部の朱印と貼付票',[365,30,725,265]),
  ('directional_pattern_body','対照：模様と重なる本文',[230,480,1100,700])],
 '200005798_00092_2':[
  ('directional_gray_top','候補：白黒画像の上余白の濃淡',[200,120,1400,470]),
  ('directional_gray_fold','候補：左上の折れ状線',[140,80,360,420]),
  ('directional_gray_body','対照：白黒画像の本文',[350,850,1200,1100])],
 '200005598_00003_2':[
  ('directional_black_spot','候補：上余白の孤立黒い跡・原因不明',[1000,190,450,400]),
  ('directional_black_edge','対照：右枠付近の線',[1450,130,400,450]),
  ('directional_black_body','対照：白黒本文の筆画',[350,720,1150,1200])]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'baseline_review.json').exists() and not (OUT/'v6_masks').exists()
    check_bundle(SCREEN,'screening.json','complete.json')
    screen=json.loads((SCREEN/'screening.json').read_text())
    rejected_illustrations=['200004107_00033_1','200004107_00028_1','200004148_00043_2','200004148_00043_1',
        '200004148_00084_2','200005598_00005_1','200005798_00085_1','200015843_00002_2','200015843_00003_1',
        '200015843_00116_1','200015843_00093_1','200021637_00010_1','200021637_00003_1',
        '200021637_00013_2','200021637_00005_1']
    covers=['200005798_00002_1','200015843_00002_1','200021637_00001']
    review={r['page']:('selected_text_page' if r['page'] in SELECTION else 'excluded_illustration' if r['page'] in rejected_illustrations
        else 'excluded_cover_or_title' if r['page'] in covers else 'reviewed_original_not_selected') for r in screen['candidates']}
    assert len(review)==48
    inventory={r['page']:r for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
    inputs={};cases=[]
    frozen={str(Path(candidate.__file__).relative_to(ROOT)):sha(Path(candidate.__file__))}
    for module in [candidate.rim,candidate.rim.MARGIN,candidate.rim.MARGIN.first,candidate.rim.MARGIN.second,candidate.top,candidate.thin,candidate.pale,candidate.upper,candidate.direction]:
        frozen[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    for key,specs in SELECTION.items():
        pg=inventory[key].copy();assert pg['book'] not in screen['excluded_books']
        original=read(ROOT/pg['original']);H,W=original.shape[:2];rois=[]
        for rid,title,box in specs:
            x,y,w,h=box;assert 0<=x<x+w<=W and 0<=y<y+h<=H
            rois.append(dict(id=rid,title=title,xywh=box,baseline_failure=None))
            figure([original[y:y+h,x:x+w]],['原画像：出力前に固定'],OUT/'original_ROIs'/(rid+'.png'),title+' · '+key,
                'npj・v6・候補出力を見ずに選定。領域名は原因の正解ラベルではない。','画素GT・学習時独立性は未確定。')
        pg.update(rois=rois,contains_illustration=False,baseline_failure=None,selection_reason='original texture and writing controls, gray fold and isolated mark candidates')
        cases.append(pg)
        for n in ['original','npj_input','npj_black','npj_red','npj_degradation','npj_background']:inputs[pg[n]]=sha(ROOT/pg[n])
    data=dict(cases=cases,selected_pages=4,selected_books=4,selected_ROIs=13,inputs=inputs,
        original_only=True,selection_before_new_v6_and_candidate=True,excluded_books=screen['excluded_books'],
        illustration_and_decorative_filter=review,parameters_tuned_after_new_outputs=False,pixel_GT=False,
        training_independence_unknown=True,candidate_sources=frozen,candidate_settings=candidate.SETTINGS,
        limitations='gray pages may have no warm seed; no confirmed positive of the exact two prior directional patterns')
    dump(OUT/'baseline_review.json',data)
    dump(OUT/'visual_selection.json',dict(review=review,selection_sha256=sha(OUT/'baseline_review.json'),
        screening_sha256=sha(SCREEN/'screening.json'),original_only=True))
    shutil.copyfile(__file__,OUT/'selection_runner_snapshot.py')
    dump(OUT/'selection_complete.json',dict(source_sha256=sha(Path(__file__)),outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='selection_complete.json'}))
    print('Fixed4 original-first pages /4 unused books /13 ROIs',flush=True)


if __name__=='__main__':main()
