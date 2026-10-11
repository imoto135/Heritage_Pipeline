"""Freeze four original-selected pages from three further unused book groups."""
import csv
import json
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw, ImageFont
from build_npj_v6_comparison import dump, read, sha, FONT_PATH
from package_defect_context_review import stack

ROOT=Path(__file__).resolve().parents[2]
SCREEN=ROOT/'tmp_work/unseen_dark_margin_screening_20261001'
OUT=ROOT/'tmp_work/unseen_dark_margin_validation_20261001'
SELECTION={
 '200014685_00004_2':[
  ('extra_top_branch','上端の枝分かれした損傷候補',[1550,0,440,420]),
  ('extra_branch_blank','対照：損傷候補左の余白',[670,85,560,320]),
  ('extra_branch_body','対照：同じページの本文',[380,900,700,900])],
 '200014685_00002_2':[
  ('extra_brown_stain','上余白の広い黄褐色のシミ候補',[620,35,900,430]),
  ('extra_second_branch','同じ冊の別ページの上端損傷候補',[1540,0,440,350]),
  ('extra_small_seal','対照：右下の小さな朱印',[1740,2880,260,330]),
  ('extra_stain_body','対照：シミのページの本文',[420,850,700,900])],
 'umgy004_013':[
  ('extra_isolated_marks','上余白の孤立した茶色い跡：原因未確定',[530,80,640,230]),
  ('extra_mark_blank','対照：跡の左の余白',[160,90,300,210]),
  ('extra_mark_body','対照：茶色い紙の本文',[380,600,600,850])],
 '200008316_00020_1':[
  ('extra_header_strokes','対照：上余白内の大きな実筆画',[560,210,1200,430]),
  ('extra_header_blank','対照：見出し右の余白',[1850,220,650,370]),
  ('extra_header_body','対照：白黒の本文',[650,1250,700,950])]}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'baseline_review.json').exists() and not (OUT/'v6_masks').exists()
    screen=json.loads((SCREEN/'screening.json').read_text());assert sha(SCREEN/'source_snapshot.py')==screen['source_sha256']
    for p,d in screen['dependencies'].items():assert sha(ROOT/p)==d
    inventory={r['page']:r for r in csv.DictReader((ROOT/'tmp_work/npj_failure_collection_20261001/page_inventory.csv').open())}
    prior=json.loads((ROOT/'tmp_work/unseen_thin_margin_validation_20261001/baseline_review.json').read_text())
    review={key:'selected_text_page' for key in SELECTION}
    for key in ['200004107_00033_1','200004107_00028_1','umgy002_006','200004148_00043_2',
                '200005798_00085_1','200014685_00009_1','200014685_00028_2','200014685_00006_1',
                '200014685_00007_1','umgy004_009','200014685_00008_1','umgy001_011',
                '200021637_00010_1','200021637_00003_1']:
        review[key]='excluded_illustration'
    for key in ['200005798_00092_2','200014685_00015_2','200004148_00016_1','200003967_00024_2',
                '200003967_00036_2','200005598_00003_2','200005598_00048_2','200008316_00126_2',
                'umgy012_013','umgy002_025','umgy010_022','200014685_00005_2','umgy006_021','umgy003_007']:
        review[key]='reviewed_original_not_selected'
    cases=[];inputs={};figures=[]
    font=ImageFont.truetype(FONT_PATH,23)
    for key,specs in SELECTION.items():
        pg=inventory[key].copy();assert pg['book'] not in screen['excluded_books']
        image=read(ROOT/pg['original']);H,W=image.shape[:2];rois=[]
        for rid,title,box in specs:
            x,y,w,h=box;assert 0<=x<x+w<=W and 0<=y<y+h<=H
            rois.append(dict(id=rid,title=title,xywh=box,baseline_failure=None))
            pil=Image.fromarray(cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2RGB));pil.thumbnail((1000,900))
            canvas=Image.new('RGB',(1400,pil.height+145),(250,248,243));draw=ImageDraw.Draw(canvas)
            draw.text((14,8),title+' · '+key,font=font,fill='black');canvas.paste(pil,(14,48))
            draw.text((14,pil.height+65),'原画像のみ。npj/v6と候補処理の出力前に選定を固定。',font=font,fill='black')
            draw.text((14,pil.height+100),'損傷の形に見えるものも物理原因・画素GTは未確定。',font=font,fill='black')
            p=OUT/'original_ROIs'/(rid+'.png');p.parent.mkdir(exist_ok=True);canvas.save(p)
        top=Image.fromarray(cv2.cvtColor(image[:round(.25*H)],cv2.COLOR_BGR2RGB));top.thumbnail((1400,900))
        canvas=Image.new('RGB',(1400,top.height+100),(250,248,243));draw=ImageDraw.Draw(canvas)
        draw.text((14,8),key+' 原画像の上部：出力を見ずに選定',font=font,fill='black');canvas.paste(top,(0,45))
        p=OUT/'original_details'/(key+'.png');p.parent.mkdir(exist_ok=True);canvas.save(p);figures.append(p)
        pg.update(rois=rois,contains_illustration=False,baseline_failure=None,
                  selection_reason='original-only dark/warm screening and visual damage/stain/header controls')
        cases.append(pg)
        for k in ['original','npj_input','npj_black','npj_red','npj_degradation','npj_background']:inputs[pg[k]]=sha(ROOT/pg[k])
    data=dict(cases=cases,selected_pages=4,selected_books=3,selected_ROIs=13,
              selection_before_new_v6_and_candidate=True,original_only=True,screened_pages=screen['screened_pages'],
              excluded_books=screen['excluded_books'],pixel_GT=False,training_independence_unknown=True,
              parameters_tuned_after_new_outputs=False,frozen_rim_source_sha256=prior['frozen_rim_source_sha256'],
              frozen_candidate_sources=prior['frozen_candidate_sources'],candidate=prior['candidate'],
              restoration=prior['restoration'],illustration_and_decorative_filter=review,inputs=inputs,
              same_book_pages=['200014685_00004_2','200014685_00002_2'],
              selection_limits='same book group counted once; umgy volume prefix not claimed independent books; no confirmed pixel GT')
    assert sum(len(p['rois']) for p in cases)==13
    dump(OUT/'baseline_review.json',data)
    dump(OUT/'visual_selection.json',dict(review=review,selection_sha256=sha(OUT/'baseline_review.json'),
        original_only=True,screening_sha256=sha(SCREEN/'screening.json'),screening_source_sha256=screen['source_sha256']))
    stack(figures,OUT/'selected_originals.png','次の固定検証：原画像から選んだ3冊群・4ページ',
          ['上端の枝状損傷・広いシミの候補、孤立した茶色い跡、大きな実筆画の対照。',
           '2つの枝状損傷ページは同じ冊。学習・画素GTからの独立性は不明。'])
    shutil.copyfile(__file__,OUT/'selection_runner_snapshot.py')
    dump(OUT/'selection_complete.json',dict(source_sha256=sha(Path(__file__)),
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='selection_complete.json'},
        frozen_candidate_sources=data['frozen_candidate_sources']))
    print('Fixed original-first selection:4 pages /3 book groups /13 ROIs',flush=True)


if __name__=='__main__':main()
